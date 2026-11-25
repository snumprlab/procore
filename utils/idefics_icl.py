"""[fix_icl] Idefics3 ICL training examples built with EXACTLY the eval prompt.

eval_VLM.py builds Idefics prompts with parse_image_qa_pairs() + build_idefics_prompts() +
processor.apply_chat_template(..., add_generation_prompt=True). Training used a different
hand-written "User:<image>Q<end_of_utterance>Assistant:A..." template and put loss on the
demonstration turns as well. This module reproduces the eval prompt (functions copied
verbatim from eval_VLM.py so this file does not import the eval script) and appends the
target answer, with loss ONLY on the final answer (+ <end_of_utterance>).
"""
import re

import torch

from utils.mimic_shift import strip_demos

IGNORE_INDEX = -100


def parse_image_qa_pairs(qs):
    # verbatim copy of eval_VLM.parse_image_qa_pairs
    qs = qs.replace("\n", " ").strip()

    if "<image> <image>" in qs:
        double_image = True
        chunks = [c.strip() for c in qs.split("<image> <image") if c.strip()]
        if not qs.startswith("<image> <image"):
            chunks = [" ".join(chunks[:2])] + chunks[2:]
    else:
        double_image = False
        chunks = [c.strip() for c in qs.split("<image>") if c.strip()]
        if not qs.startswith("<image>"):
            chunks = [" ".join(chunks[:2])] + chunks[2:]

    pairs = []
    for chunk in chunks:
        match = re.search(r"Question\s*[:：]\s*(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
        if not match:
            match = re.search(r"(.*?)\s*Answer\s*[:：]\s*(.*)", chunk, re.IGNORECASE)
        if match:
            q, a = match.groups()
        else:
            q, a = chunk, ""
        pairs.append((q.strip(), a.strip()))
    return pairs, double_image


def build_idefics_prompts(pairs, image_file, double_image):
    # verbatim copy of eval_VLM.build_idefics_prompts
    prompts = []
    if isinstance(image_file, list):
        if double_image:
            for pair_i, pair in enumerate(pairs):
                if pair_i == len(pairs) - 1:
                    user_dict = {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": f"User: {pair[0]} \nAssistant: "}]}
                else:
                    user_dict = {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": f"User: {pair[0]} \nAssistant: {pair[1]}"}]}
                prompts.append(user_dict)
        else:
            for image_file_i, image_path in enumerate(image_file):
                pair = pairs[image_file_i]
                if image_file_i == len(image_file) - 1:
                    user_dict = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"User: {pair[0]} \nAssistant: "}]}
                else:
                    user_dict = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"User: {pair[0]} \nAssistant: {pair[1]}"}]}
                prompts.append(user_dict)
    return prompts


def build_idefics_train_example(processor, tokenizer, source, load_image, mimic=False):
    """Return dict(input_ids, labels, attention_mask[, pixel_values, pixel_attention_mask]).

    mimic=True builds the query-only (zero-demo) version: instruction + query image(s) + query.
    """
    image_files = source.get("image", [])
    if isinstance(image_files, str):
        image_files = [image_files]
    qs = source["conversations"][0]["value"]
    answer = source["conversations"][1]["value"].strip()
    if mimic and "<image>" in qs:
        qs, n_q = strip_demos(qs, "<image>")
        image_files = image_files[-n_q:]

    if source.get("plain_prompt"):
        # [fix_halluc] single-image non-ICL instruction data (LRV / VOLCANO / LURE datalists):
        # the instruction itself may contain "answer:" (e.g. VOLCANO "Initial answer: ..."), so it
        # must not be split into ICL (question, answer) pairs.
        pairs, double_image = [(qs.replace("\n", " ").replace("<image>", "").strip(), "")], False
        image_files = list(image_files)[:1]
    else:
        pairs, double_image = parse_image_qa_pairs(qs)
    prompts = build_idefics_prompts(pairs, list(image_files), double_image)
    prompt = processor.apply_chat_template(prompts, add_generation_prompt=True)
    images = [load_image(f) for f in image_files]

    # the model continues "...\nAssistant:" with " <answer>" (leading space), as in eval generation
    target = f" {answer}<end_of_utterance>"
    inputs = processor(text=prompt + target, images=images if images else None, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]

    tgt_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    n = len(tgt_ids)
    if n == 0 or input_ids[-n:].tolist() != tgt_ids:
        # tokenization merged across the prompt/answer boundary: fall back to the last "Assistant:" token run
        eou = tokenizer.convert_tokens_to_ids("<end_of_utterance>")
        eou_pos = (input_ids == eou).nonzero(as_tuple=True)[0]
        if len(eou_pos) < 2:
            raise ValueError("cannot locate the final answer span")
        ast = tokenizer("\nAssistant:", add_special_tokens=False)["input_ids"]
        start = int(eou_pos[-2]) + 1 + len(ast)
        n = len(input_ids) - start
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[-n:] = input_ids[-n:]

    out = dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    if inputs.get("pixel_values") is not None:
        out["pixel_values"] = inputs["pixel_values"][0]
        if inputs.get("pixel_attention_mask") is not None:
            out["pixel_attention_mask"] = inputs["pixel_attention_mask"][0]
    return out
