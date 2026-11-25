"""VICL evaluation (Zhou et al., "Visual In-Context Learning for Large Vision-Language
Models", ACL Findings 2024), stages 2-3, on top of the stage-1 v2 datalists.

Stage 1 (visual demonstration retrieval + CLIP cross-modal rerank with a query
caption, ascending order) is done OFFLINE by tools/retrieval/build_retrieval.py
and stored in datalists/**/dataset-*_vicl_v2.json (ORDERS=vicl_v2).  This script
then, per test item:
  2) intent-oriented image summarization: for every demonstration, the evaluated
     MLLM sees ONLY that demo's image(s) plus the demo's text (question + answer,
     i.e. the task intent) and writes a short intent-focused description;
  3) intent-oriented demonstration composition: every demo's <image> placeholder(s)
     is replaced by "[Image: <summary>]" while ALL other prompt text (instruction
     prefix, demo questions/answers, separators, the query segment) is kept
     verbatim; only the query keeps its real image(s).
  4) the MLLM answers this composed prompt; scoring mirrors eval_VLM.py.

Ported from /tmp/LMM-ICL/eval_VLM_vicl.py with these fixes:
  * the old parser treated the instruction prefix as a demo (image misalignment for
    TrueMICL / VL-ICL) and lost MIC demo answers; the composed prompt dropped the
    demos' questions/answers and the instruction prefix;
  * the final prompt is sent as ONE user turn through the model's own chat template
    (the Qwen dataset path re-splits prompts on "<image>"/"Answer:" and would drop
    the text-only demos);
  * checkpoints are read from $CKPT_ROOT with the same strict key check as eval_VLM.py;
  * EVAL_LIMIT=<n> evaluates the first n items (smoke tests).
Use it with VARIANT=vicl ORDERS=vicl_v2 bash eval.sh (MODEL_SUFFIX _vicl).
"""
import datetime
import json
import logging.config
import os
import random
import re
import time
import warnings

import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import BitsAndBytesConfig

from configuration.VLM_config_new import DataArguments, ModelArguments, TrainingArguments
from utils.train_utils import get_VLMmodel
from utils.eval_metrics import NLPEvaluator, matching_token_num
from eval_VLM import can_infer, get_datalists, load_image, parse_choice_list, CHOICES

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

SUMMARIZE_TEMPLATE = (
    "Here is one demonstration of a task (instruction, question and its correct answer):\n"
    "\"\"\"\n{demo}\n\"\"\"\n"
    "First work out what the task is asking about the image (the intent of the question), then "
    "describe ONLY the visual content of the image that is relevant to this intent and that "
    "justifies the given answer, so that someone who cannot see the image could solve similar "
    "questions. Reply with one short paragraph and nothing else."
)
SUMMARY_MAX_NEW_TOKENS = 128
IMG_GROUP = re.compile(r"((?:<image>\s*)+)")


def resolve_image(p):
    rel = p[2:] if p.startswith("./") else p
    if not os.path.exists(rel):
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0] == "data":
            alt = os.path.join("data", parts[1], parts[1], *parts[2:])
            if os.path.exists(alt):
                return alt
    return rel


def parse_prompt(text, image_files):
    """-> prefix, [ (images, demo_text) ], (query_images, query_text_with_tags)"""
    toks = IMG_GROUP.split(text)
    prefix, tags, bodies = toks[0], toks[1::2], toks[2::2]
    n = [t.count("<image>") for t in tags]
    if sum(n) != len(image_files) or not tags:
        return None
    pos, demos = 0, []
    for t, b, k in zip(tags[:-1], bodies[:-1], n[:-1]):
        demos.append((image_files[pos:pos + k], t, b))
        pos += k
    return prefix, demos, (image_files[pos:], tags[-1] + bodies[-1])


def compose(prefix, demos, summaries, query_tail):
    out = prefix
    for (imgs, tag, body), s in zip(demos, summaries):
        s = s.replace("<image>", " ").strip()
        trail = tag[len(tag.rstrip()):]  # keep the whitespace that followed the tag(s)
        out += f"[Image: {s}]" + trail + body
    return out + query_tail


# ----------------------------------------------------------------- model runners
def run_chat(text, image_files, model, tokenizer, processor, model_args, training_args, device, max_new_tokens):
    """One user turn = interleaved text and images in the order of <image> tags."""
    name = model_args.model_name_or_path
    image_files = [resolve_image(p) for p in image_files]
    do_sample = training_args.mode != "dara"
    if "intern" in name.lower():
        pvs, npl = [], []
        for p in image_files:
            pv = load_image(p, max_num=12).to(torch.bfloat16).to(device)
            pvs.append(pv)
            npl.append(pv.size(0))
        q = text
        gen = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        with torch.inference_mode():
            if pvs:
                resp = model.chat(tokenizer, torch.cat(pvs, 0), q, gen, num_patches_list=npl,
                                  history=None, return_history=False)
            else:
                resp = model.chat(tokenizer, None, q.replace("<image>", ""), gen, history=None, return_history=False)
        return resp.strip()
    pieces = re.split(r"(<image>)", text)
    content, ims = [], []
    k = 0
    for pc in pieces:
        if pc == "<image>":
            content.append({"type": "image"})
            ims.append(Image.open(image_files[k]).convert("RGB"))
            k += 1
        elif pc.strip():
            content.append({"type": "text", "text": pc})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kw = dict(text=[prompt], return_tensors="pt", padding=True)
    if ims:
        kw["images"] = ims
    inputs = processor(**kw)
    inputs = {kk: v.to(device) for kk, v in inputs.items()}
    gen = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        gen["temperature"] = training_args.eval_temp
    with torch.inference_mode():
        out = model.generate(**inputs, **gen)
    out = out[:, inputs["input_ids"].shape[1]:]
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else tokenizer
    return tok.batch_decode(out, skip_special_tokens=True)[0].strip()


def vicl_predict(d, model, tokenizer, processor, model_args, training_args, device):
    text = d["conversations"][0]["value"]
    imgs = d.get("image", [])
    imgs = [imgs] if isinstance(imgs, str) else list(imgs)
    parsed = parse_prompt(text, imgs)
    if parsed is None:
        raise RuntimeError(f"cannot parse VICL prompt (tags vs {len(imgs)} images): {text[:200]}")
    prefix, demos, (q_imgs, q_tail) = parsed
    summaries = []
    for dimgs, tag, body in demos:
        demo_txt = (prefix.strip() + "\n" if prefix.strip() else "") + body.strip()
        demo_txt = demo_txt.replace("Answer:", "Answer -").replace("Question:", "Question -")
        stext = tag.strip() + "\n" + SUMMARIZE_TEMPLATE.format(demo=demo_txt)
        summaries.append(run_chat(stext, dimgs, model, tokenizer, processor, model_args, training_args,
                                  device, SUMMARY_MAX_NEW_TOKENS))
    composed = compose(prefix, demos, summaries, q_tail)
    pred = run_chat(composed, q_imgs, model, tokenizer, processor, model_args, training_args, device,
                    model_args.max_new_tokens)
    return pred, summaries, composed


def score_and_save(records, preds_meta, task_type, dataname, tokenizer, training_args, model_args, logger):
    predictions = []
    acc, correct, total = [], 0, 0
    nw, ngw, nwc = 0, 1, 1
    for (d, pred, summaries, composed) in records:
        prompt = d["conversations"][0]["value"]
        gold = d["conversations"][1]["value"].strip().lower()
        p = pred.strip().lower()
        rec = {"image_file": d.get("image"), "sentence": p, "gt_sentence": gold, "summaries": summaries,
               "composed_prompt": composed}
        if task_type == "multi-choice":
            choices = CHOICES["classification"] if "classification" in dataname else (
                CHOICES["nlvr2"] if "nlvr2" in dataname else parse_choice_list(prompt))
            opt = can_infer(p, choices)
            ok = isinstance(opt, str) and gold == opt.lower()
            correct += ok
            total += 1
            rec["status"] = "correct" if ok else ("wrong" if isinstance(opt, str) else "unkown")
        elif task_type == "exact-match":
            m = re.search(r"\d+", p)
            ok = (m.group() == gold) if m else (gold == p or gold == p.replace(".", ""))
            acc.append(int(ok))
            rec["result"] = "correct" if ok else "wrong"
        else:
            il, oi = tokenizer.encode(gold), tokenizer.encode(p)
            nw += len(set(il))
            ngw += len(set(oi))
            nwc += matching_token_num(oi, il)
        predictions.append(rec)
    if task_type == "multi-choice":
        scores = {"accuracy": correct / total if total else 0.0}
    elif task_type == "exact-match":
        scores = {"accuracy": float(np.mean(acc)) if acc else 0.0}
    else:
        scores = NLPEvaluator(predictions).evaluate()
        scores["precision"] = nwc / nw if nw else 0.0
        scores["recall"] = nwc / ngw
    logger.info(f"Test (Server) VICL | {dataname} | {scores}")
    predictions.append(scores)
    out_dir = (f"./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/"
               f"{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as fp:
        json.dump(predictions, fp, indent=4)
    print("saved at", out_dir, scores)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    bnb = {}
    if training_args.bits in [4, 8]:
        bnb.update(dict(device_map={"": training_args.device}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4, load_in_8bit=training_args.bits == 8,
            llm_int8_skip_modules=["mm_projector"], llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type)))
    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()
    out_dir = (f"./final2_eval_results/{training_args.ordering}/{training_args.eval_shot}/"
               f"{training_args.dataset_name}/scenario{training_args.scenario}/{model_args.model_name}")
    os.makedirs(out_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(out_dir, f"round_{training_args.round_to_eval}.log"), mode="w")
    fh.setFormatter(logging.Formatter("[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"))
    logger.addHandler(fh)
    logger.info(training_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    name = model_args.model_name_or_path
    processor, tokenizer = None, None
    if "Qwen" in name:
        model, processor, data_args = get_VLMmodel(model_args, training_args, bnb, data_args)
        tokenizer = processor.tokenizer
    elif "Intern" in name:
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb, data_args)
    elif "onevision" in name.lower() or "vl" in name or "idefics" in name.lower():
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb, data_args)
    else:
        raise RuntimeError(f"VICL: unsupported model {name}")
    if getattr(model_args, "use_mtv", False):
        raise RuntimeError("VICL variant does not implement MTV")

    if not training_args.zeroshot and training_args.eval_server:
        ck = os.path.join(os.environ.get("CKPT_ROOT", "."), f"client_states_{training_args.note}",
                          f"server_model_round{training_args.round_to_eval - 1}.pth")
        logger.info(f"load {ck}")
        sd = torch.load(ck, map_location="cpu")
        res = model.load_state_dict(sd, strict=False)
        matched = len(sd) - len(res.unexpected_keys)
        if res.unexpected_keys or matched == 0:
            raise RuntimeError(f"checkpoint {ck} does not match the model: {matched} keys loaded")
        del sd
    model.eval()
    if "intern" in name.lower() or "idefics" in name.lower():
        model = model.to(device)

    _, test_datalists = get_datalists(training_args, training_args.scenario, shot=training_args.eval_shot,
                                      dataset_name=training_args.dataset_name, ordering=training_args.ordering)
    lim = int(os.environ.get("EVAL_LIMIT", "0") or 0)
    start = time.time()
    for round_i, info in enumerate(test_datalists[0]):
        if round_i + 1 > training_args.round_to_eval:
            continue
        data = info["data"][:lim] if lim else info["data"]
        records = []
        for d in tqdm(data, desc=info["data_name"]):
            pred, summaries, composed = vicl_predict(d, model, tokenizer, processor, model_args, training_args, device)
            records.append((d, pred, summaries, composed))
        score_and_save(records, None, info.get("type", "open-ended"), info["data_name"], tokenizer,
                       training_args, model_args, logger)
    logger.info(f"elapsed time {datetime.timedelta(seconds=int(time.time() - start))}")


if __name__ == "__main__":
    main()
