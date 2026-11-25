"""STV training (Ma et al., AAAI 2026): builds the self-contained steering artifact.

Replaces the old methods/stv.py trainer (which trained LoRA with CE and never used its
vectors at eval).  Only the selection logits are learned; the model is frozen, no LoRA.

Pipeline (ICL training data only: datalists/<ds>/train/dataset-mixed.json)
  1. For N items: forward the ICL prompt (k demos + query) and the zero-shot prompt (same
     instruction header + query only), both ending at the generation prompt.  Record the
     per-head o_proj input at the GENERATION POSITION (last prompt token) -- no gold answer
     is in the prompt when activations are read.
  2. Sensitivity of head (l,h) = mean_i ||a_icl_i(l,h) - a_zero_i(l,h)||  (STV_SENS=mean_norm,
     default) or ||mean_i(a_icl - a_zero)|| (STV_SENS=norm_mean).  Keep the top-k heads.
  3. Per selected head, KMeans over its ICL activations -> bank of C candidate vectors.
  4. REINFORCE over a categorical logit per head (K x C): sample one vector per head, steer
     the zero-shot query, reward = -KL(p_ICL || p_steer) over the first T answer positions
     (teacher-forced with the gold answer; p_ICL from the ICL prompt), i.e. the steered
     zero-shot output is pushed to match the output with ICL demos.
  5. Pick argmax / sampled candidates by held-out validation KL, save artifact to
     $CKPT_ROOT/client_states_<note>/server_model_round0.pth (format stv_v1, see utils/stv_steer.py).
"""
import copy
import json
import math
import os
import random
import re
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image

from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
from utils.train_utils import get_VLMmodel
from utils.stv_steer import STV_FORMAT, STVSteerer, HeadRecorder, lm_geometry


def log(*a):
    print("[STV]", *a, flush=True)


# ----------------------------------------------------------------------------- data
_IMG_GROUP = re.compile(r"<image>(?:\s*<image>)*")


def make_zero_shot(item):
    """Same instruction header + the query (its image group + text); demos removed."""
    q = item["conversations"][0]["value"]
    imgs = item["image"] if isinstance(item["image"], list) else [item["image"]]
    groups = list(_IMG_GROUP.finditer(q))
    if len(groups) < 2:
        return None
    last = groups[-1]
    n_q = last.group(0).count("<image>")
    header = q[: groups[0].start()]
    query_text = q[last.end():].replace("Learn from the demos and give only the answer.", "Give only the answer")
    z = copy.deepcopy(item)
    z["conversations"][0]["value"] = header + last.group(0) + query_text
    z["image"] = imgs[-n_q:]
    return z


class InputBuilder:
    """Builds eval-identical prompt tensors (generation prompt, optional answer appended)."""

    def __init__(self, arch, model, tokenizer, processor, data_args, device):
        self.arch, self.model, self.tok, self.proc, self.data_args, self.device = arch, model, tokenizer, processor, data_args, device
        if arch == "internvl":
            mod = sys.modules[type(model).__module__]
            self.get_conv_template = mod.get_conv_template

    def answer_ids(self, gold, T):
        text = (" " + gold) if self.arch == "idefics" else gold
        ids = self.tok(text, add_special_tokens=False)["input_ids"][:T]
        return torch.tensor(ids, dtype=torch.long)

    def build(self, item):
        """returns dict(kind=..., tensors...) with input_ids [1,P]"""
        import eval_VLM as ev
        if self.arch in ("qwen3", "ov15"):
            if self.arch == "qwen3":
                ds = ev.QwenVL3_GenerationDataset([item], self.proc, self.tok, self.data_args)
            else:
                ds = ev.OneVision1_5_GenerationDataset([item], self.tok, self.proc, self.data_args)
            d = ds[0]
            return dict(input_ids=d["input_ids"].to(self.device),
                        pixel_values=d["pixel_values"].to(self.device, torch.bfloat16),
                        image_grid_thw=d["image_grid_thw"].to(self.device))
        if self.arch == "idefics":
            pairs, dbl = ev.parse_image_qa_pairs(item["conversations"][0]["value"])
            prompts = ev.build_idefics_prompts(pairs, item["image"], dbl)
            imgs = [Image.open(p).convert("RGB") for p in item["image"]]
            prompt = self.proc.apply_chat_template(prompts, add_generation_prompt=True)
            inp = self.proc(text=prompt, images=imgs, return_tensors="pt")
            out = {k: v.to(self.device) for k, v in inp.items() if k != "attention_mask"}
            out["pixel_values"] = out["pixel_values"].to(torch.bfloat16)
            return out
        if self.arch == "internvl":
            m = self.model
            pixel_values, num_patches_list, _, _ = ev.load_intern_inputs(item, self.device)
            question = item["conversations"][0]["value"]
            if "<image>" not in question:
                question = "<image>\n" + question
            t = self.get_conv_template(m.template)
            t.system_message = m.system_message
            t.append_message(t.roles[0], question)
            t.append_message(t.roles[1], None)
            query = t.get_prompt()
            for n in num_patches_list:
                query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * m.num_image_token * n + "</img>", 1)
            ids = self.tok(query, return_tensors="pt")["input_ids"].to(self.device)
            return dict(input_ids=ids, pixel_values=pixel_values)
        raise ValueError(self.arch)

    @torch.no_grad()
    def forward(self, inp, answer=None):
        """Forward prompt (+ T answer tokens). Returns logits [T, V] (float) at positions P-1..P+T-2,
        i.e. the next-token distributions for the T answer tokens (T=0 -> 1 row at P-1)."""
        ids = inp["input_ids"]
        if answer is not None and len(answer):
            ids = torch.cat([ids, answer.view(1, -1).to(ids.device)], 1)
        attn = torch.ones_like(ids)
        if self.arch == "internvl":
            m = self.model
            vit = m.extract_feature(inp["pixel_values"])
            emb = m.language_model.get_input_embeddings()(ids)
            B, N, C = emb.shape
            emb = emb.reshape(B * N, C)
            sel = ids.reshape(-1) == m.img_context_token_id
            emb[sel] = vit.reshape(-1, C).to(emb.dtype)
            fn, kw = m.language_model, dict(inputs_embeds=emb.reshape(B, N, C))
        else:
            fn, kw = self.model, {k: v for k, v in inp.items() if k != "input_ids"}
            kw["input_ids"] = ids
        n = 0 if answer is None else len(answer)
        keep = n + 1  # only the last rows are needed (full-vocab logits of 20k+ ICL tokens would not fit)
        try:
            out = fn(attention_mask=attn, use_cache=False, logits_to_keep=keep, **kw)
        except TypeError:
            out = fn(attention_mask=attn, use_cache=False, **kw)
        lg = out.logits[0, -keep:]
        return lg[: max(n, 1)].float()


def arch_of(name):
    n = name.lower()
    if "qwen3" in n:
        return "qwen3"
    if "onevision-1.5" in n:
        return "ov15"
    if "idefics" in n:
        return "idefics"
    if "internvl3_5" in n:
        return "internvl"
    raise ValueError(f"STV: unsupported model {name}")


# ----------------------------------------------------------------------------- main
def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, rest = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stv_datalist", default=None)
    ap.add_argument("--stv_rl_items", type=int, default=int(os.environ.get("STV_RL_ITEMS", 64)))
    ap.add_argument("--stv_val_items", type=int, default=int(os.environ.get("STV_VAL_ITEMS", 16)))
    ap.add_argument("--stv_max_answer_tokens", type=int, default=int(os.environ.get("STV_MAX_ANSWER_TOKENS", 16)))
    ap.add_argument("--stv_sens", default=os.environ.get("STV_SENS", "mean_norm"), choices=["mean_norm", "norm_mean"])
    ap.add_argument("--stv_max_seq", type=int, default=int(os.environ.get("STV_MAX_SEQ", 40000)))
    stv = ap.parse_args(rest)

    torch.manual_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    device = torch.device("cuda")

    arch = arch_of(model_args.model_name_or_path)
    ds_name = training_args.dataset_name
    datalist_path = stv.stv_datalist
    if not datalist_path:
        # follow scenarios/mixed/<ds>/scenario-0.json (same as mt-icl/mimic), so leak-free *_clean sets are picked up
        _scen = f"./scenarios/mixed/{ds_name}/scenario-0.json"
        if os.path.exists(_scen):
            _d = json.load(open(_scen))[0]["datasets"][0]
            datalist_path = f"./datalists/{_d['dataset']}/train/dataset-{_d['subset_id']}.json"
        else:
            datalist_path = f"./datalists/{ds_name}/train/dataset-mixed.json"
    if "/train/" not in datalist_path:
        raise RuntimeError(f"STV must be built from ICL TRAINING data, got {datalist_path}")
    ckpt_dir = os.path.join(os.environ.get("CKPT_ROOT", "."), f"client_states_{training_args.note}")
    ckpt_path = os.path.join(ckpt_dir, "server_model_round0.pth")
    log(f"arch={arch} datalist={datalist_path} -> {ckpt_path}")

    # model (frozen, no LoRA)
    training_args.lora_enable = False
    processor = None
    if arch == "qwen3":
        model, processor, data_args = get_VLMmodel(model_args, training_args, {}, data_args)
        tokenizer = processor.tokenizer
    elif arch == "internvl":
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, {}, data_args)
    else:
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, {}, data_args)
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    builder = InputBuilder(arch, model, tokenizer, processor, data_args, device)
    layers, n_heads, head_dim = lm_geometry(model)
    L = len(layers)
    log(f"LM geometry: layers={L} heads={n_heads} head_dim={head_dim} first={layers[0][1]}")

    # data: ICL items with >=1 demo
    data = json.load(open(datalist_path))
    random.Random(training_args.seed).shuffle(data)
    pairs = []
    for it in data:
        if "image" not in it:
            continue
        z = make_zero_shot(it)
        if z is not None:
            pairs.append((it, z))
    N_bank = min(training_args.intervention_n_trials, len(pairs))
    n_val = min(stv.stv_val_items, max(0, len(pairs) - N_bank))
    bank_pairs = pairs[:N_bank]
    rl_pairs = bank_pairs[: min(stv.stv_rl_items, N_bank)]
    val_pairs = pairs[N_bank:N_bank + n_val]
    log(f"items: usable={len(pairs)} bank/sensitivity={len(bank_pairs)} rl={len(rl_pairs)} val={len(val_pairs)}")
    T = stv.stv_max_answer_tokens

    # ------------------------------------------------------------------ phase 1
    rec = HeadRecorder(layers, n_heads, head_dim)
    icl_acts, sens_sum, delta_sum = [], torch.zeros(L, n_heads), torch.zeros(L, n_heads, head_dim)
    targets = {}  # id(pair) -> dict(zero_inp, ans, target_logp [T,V] fp16 cpu, P0)
    t0 = time.time()

    def prep(pair, need_target):
        icl, zero = pair
        ans = builder.answer_ids(str(icl["conversations"][1]["value"]), T)
        icl_inp = builder.build(icl)
        if icl_inp["input_ids"].shape[1] > stv.stv_max_seq:
            return None
        P1 = icl_inp["input_ids"].shape[1]
        rec.pos = P1 - 1  # generation position of the ICL prompt
        lg = builder.forward(icl_inp, ans if need_target else None)
        a_icl = rec.stacked()
        tgt = F.log_softmax(lg, -1).half().cpu() if need_target else None
        del icl_inp, lg
        zero_inp = builder.build(zero)
        P0 = zero_inp["input_ids"].shape[1]
        rec.pos = P0 - 1  # generation position of the zero-shot prompt
        builder.forward(zero_inp, None)
        a_zero = rec.stacked()
        return a_icl, a_zero, dict(zero_inp=zero_inp, ans=ans, tgt=tgt, P0=P0)

    rl_ids = {id(p) for p in rl_pairs}
    kept = 0
    for i, pair in enumerate(bank_pairs):
        r = prep(pair, id(pair) in rl_ids)
        if r is None:
            continue
        a_icl, a_zero, info = r
        d = a_icl - a_zero
        sens_sum += d.norm(dim=-1)
        delta_sum += d
        icl_acts.append(a_icl.half())
        if id(pair) in rl_ids:
            targets[id(pair)] = info
        kept += 1
        if i % 10 == 0:
            log(f"phase1 {i+1}/{len(bank_pairs)} P0={info['P0']} {time.time()-t0:.0f}s")
    val_info = []
    for pair in val_pairs:
        r = prep(pair, True)
        if r is not None:
            val_info.append(r[2])
    rec.remove()
    if kept == 0:
        raise RuntimeError("no usable items")

    sens = (sens_sum / kept) if stv.stv_sens == "mean_norm" else (delta_sum / kept).norm(dim=-1)
    K = min(training_args.intervention_topk, L * n_heads)
    top = torch.topk(sens.flatten(), K).indices
    locations = torch.stack([top // n_heads, top % n_heads], 1).long()  # [K,2]
    log(f"top-{K} sensitive heads: first 10 {locations[:10].tolist()} (sens {sens.flatten()[top[:3]].tolist()})")

    # ------------------------------------------------------------------ phase 2: per-head KMeans bank
    from sklearn.cluster import KMeans
    acts = torch.stack(icl_acts, 0).float()  # [N,L,H,hd]
    C = min(training_args.intervention_num_clusters, acts.shape[0])
    bank = torch.zeros(K, C, head_dim)
    for k, (l, h) in enumerate(locations.tolist()):
        X = acts[:, l, h, :].numpy()
        km = KMeans(n_clusters=C, random_state=training_args.seed, n_init=10).fit(X)
        bank[k] = torch.from_numpy(km.cluster_centers_).float()
    del acts, icl_acts
    log(f"bank built: K={K} C={C} ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------------ phase 3: REINFORCE selection
    steerer = STVSteerer(model)
    bank_d = bank.to(device)

    def kl_for(info, values):
        if values is None:
            steerer.enabled = False
        else:
            steerer.enabled = True
            steerer.mode = info["P0"] - 1
            steerer.set_values(locations, values)
        lg = builder.forward(info["zero_inp"], info["ans"])
        P0, n = info["P0"], len(info["ans"])
        logp = F.log_softmax(lg, -1)
        tgt = info["tgt"].to(device).float()
        steerer.enabled = True
        return (tgt.exp() * (tgt - logp)).sum(-1).mean().item()

    def values_of(idx):
        return bank_d[torch.arange(K, device=device), idx]

    def mean_kl(infos, idx):
        return float(np.mean([kl_for(i, None if idx is None else values_of(idx)) for i in infos])) if infos else float("nan")

    rl_infos = [targets[id(p)] for p in rl_pairs if id(p) in targets]
    base_rl = mean_kl(rl_infos, None)
    log(f"unsteered zero-shot KL to ICL: rl={base_rl:.4f} val={mean_kl(val_info, None):.4f}")

    logits = torch.zeros(K, C, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=training_args.reinforce_lr)
    S = max(2, training_args.reinforce_samples_per_epoch)
    E = training_args.reinforce_n_epochs
    hist = []
    for ep in range(E):
        order = list(range(len(rl_infos)))
        random.shuffle(order)
        ep_kl = []
        for j in order:
            info = rl_infos[j]
            dist = torch.distributions.Categorical(logits=logits)
            samples = dist.sample((S,))  # [S,K]
            kls = torch.tensor([kl_for(info, values_of(s)) for s in samples], device=device)
            adv = -(kls - kls.mean()) / (kls.std() + 1e-6)
            logp = dist.log_prob(samples).sum(-1)  # [S]
            loss = -(adv * logp).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_kl.append(kls.mean().item())
        ent = torch.distributions.Categorical(logits=logits).entropy().mean().item()
        hist.append(dict(epoch=ep, sampled_kl=float(np.mean(ep_kl)), entropy=ent))
        log(f"REINFORCE epoch {ep+1}/{E}: mean sampled KL={np.mean(ep_kl):.4f} (unsteered {base_rl:.4f}) "
            f"policy entropy={ent:.3f} {time.time()-t0:.0f}s")

    # ------------------------------------------------------------------ phase 4: final choice on validation
    with torch.no_grad():
        dist = torch.distributions.Categorical(logits=logits)
        cands = [logits.argmax(-1)] + [dist.sample() for _ in range(training_args.reinforce_final_num_samples)]
    sel_infos = val_info if val_info else rl_infos
    scores = [mean_kl(sel_infos, c) for c in cands]
    best = int(np.argmin(scores))
    selected = cands[best].cpu()
    val_unsteered = mean_kl(sel_infos, None)
    log(f"final selection: candidate {best} (0=argmax) KL={scores[best]:.4f}; argmax KL={scores[0]:.4f}; unsteered KL={val_unsteered:.4f} "
        f"[{'val' if val_info else 'rl (no held-out items)'}]")

    art = dict(
        format=STV_FORMAT,
        model_name_or_path=model_args.model_name_or_path,
        n_layers=L, n_heads=n_heads, head_dim=head_dim,
        hook_names=[n for _, n, _ in layers],
        locations=locations.cpu(),
        sensitivity=sens.cpu(),
        bank=bank.cpu(),
        logits=logits.detach().cpu(),
        selected=selected,
        values=bank[torch.arange(K), selected].clone(),
        config=dict(datalist=datalist_path, topk=K, num_clusters=C, n_bank_items=kept, rl_items=len(rl_infos),
                    val_items=len(val_info), epochs=E, samples=S, lr=training_args.reinforce_lr,
                    max_answer_tokens=T, sensitivity=stv.stv_sens, reward="-KL(p_icl||p_steer) teacher-forced",
                    eval_positions="last position of every forward (prefill last prompt token + each generated token)"),
        stats=dict(unsteered_kl=val_unsteered, selected_kl=scores[best], argmax_kl=scores[0],
                   rl_unsteered_kl=base_rl, history=hist),
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(art, ckpt_path)
    log(f"saved STV artifact -> {ckpt_path} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
