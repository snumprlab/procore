"""Extract faithful MTV / FV task vectors from ICL TRAINING data only (see utils/taskvec_v2.py).

Launched by extract_taskvec_v2.sh with the same HF arguments as eval.sh, plus:
  --tv_bench mic|vliclv2|truemicl|isekai   --tv_scenarios "1 2 3" (default: all of the benchmark)
  --tv_method both|mtv|fv   --tv_arch <arch key>   --tv_smoke 0|1
Outputs:
  MTV/storage_v2/<arch>/<bench>/mtv_<scen>.pt             selected heads [(layer, head, -1), ...]
  MTV/storage_v2/<arch>/<bench>/mtv_activation_<scen>.pt  mean activations (L, H, 1, head_dim)
  MTV/storage_v2/<arch>/<bench>/meta_<scen>.json
  fv_v2/<arch>/<bench>/fv_<scen>.pt   {vector, edit_layer, heads, aie, mean_activations, ...}
  fv_v2/<arch>/<bench>/meta_<scen>.json
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments  # noqa: E402
from utils.train_utils import get_VLMmodel  # noqa: E402
import utils.taskvec_v2 as TV  # noqa: E402


def load_model(model_args, data_args, training_args):
    """Same loading branches as eval_VLM.main()."""
    bnb = {}
    p = model_args.model_name_or_path
    processor = None
    if "Qwen" in p:
        model, processor, data_args = get_VLMmodel(model_args, training_args, bnb, data_args)
        tokenizer = processor.tokenizer
    elif "Intern" in p:
        model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb, data_args)
    elif "onevision" in p.lower() or "idefics" in p.lower():
        model, tokenizer, processor, data_args = get_VLMmodel(model_args, training_args, bnb, data_args)
    else:
        raise ValueError(p)
    if len(set(getattr(model, "hf_device_map", {}).values())) <= 1:
        model = model.to("cuda")
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model, tokenizer, processor, data_args


def ce_of(logits, tgt):
    return torch.nn.functional.cross_entropy(logits, torch.full((logits.shape[0],), tgt, device=logits.device), reduction="none")


def mean_activations(io, patcher, data, n, L, H, log):
    acc, cnt, top1 = None, 0, 0
    for item in data.icl_prompts(n):
        st, gold = io.build(item)
        patcher.capture = True
        logits = io.full_logits(st)
        patcher.capture = False
        cur = torch.stack([c[0] for c in patcher.captured])  # (L, D_in)
        cur = cur.view(L, H, -1)
        acc = cur if acc is None else acc + cur
        cnt += 1
        top1 += int(logits[0].argmax().item() == io.target_id(gold))
    log(f"  mean activations over {cnt} {TV.ICL_SHOT}-shot train prompts; clean first-token acc {top1}/{cnt}")
    return (acc / cnt), {"n_mean": cnt, "clean_icl_first_token_acc": top1 / max(cnt, 1)}


def reinforce(io, patcher, mean_act, data, L, H, args, log):
    """MTV head selection (Huang et al.; hyper-parameters of the official code)."""
    eps, lr = 1e-3, 0.1
    params = [torch.neg(torch.ones(H, device="cuda")).requires_grad_() for _ in range(L)]
    opt = torch.optim.Adam(params, lr=lr)
    rf_items = data.zero_shot(args.tv_n_reinforce_items)
    rng = random.Random(args.tv_seed)
    patcher.values = mean_act
    hist = []
    for ep in range(args.tv_epochs):
        item = rng.choice(rf_items)
        st, gold = io.build(item)
        tgt = io.target_id(gold)
        with torch.enable_grad():
            probs = torch.stack([torch.sigmoid(p).clamp(min=eps, max=1 - eps) for p in params])
            dist = torch.distributions.Bernoulli(probs)
            samples = dist.sample((args.tv_samples,))  # (S, L, H)
            logp = dist.log_prob(samples).sum(dim=(1, 2))  # (S,)
            losses = ce_of(TV.batched_patch_logits(io, patcher, st, samples.permute(1, 0, 2).bool(), args.tv_cache_budget_gb), tgt).detach()
            R = (losses - losses.mean()) / (losses.std() + eps)
            policy_loss = (logp * R.to(logp.device)).sum()
            opt.zero_grad()
            policy_loss.backward()
            opt.step()
        hist.append(losses.mean().item())
        if ep % 50 == 0 or ep == args.tv_epochs - 1:
            pm = torch.stack([torch.sigmoid(p) for p in params])
            log(f"  [reinforce {ep}/{args.tv_epochs}] mean CE {np.mean(hist[-50:]):.4f}  E[#heads] {pm.sum().item():.1f}")
    return torch.stack([torch.sigmoid(p).detach() for p in params]).clamp(eps, 1 - eps), hist


def val_ce(io, patcher, val_items, masks, budget):
    """Mean first-token CE over validation queries for each candidate head set (masks (L, R, H))."""
    tot = torch.zeros(masks.shape[1], device="cuda")
    for item in val_items:
        st, gold = io.build(item)
        tot += ce_of(TV.batched_patch_logits(io, patcher, st, masks, budget), io.target_id(gold))
    return tot / max(len(val_items), 1)


def run_mtv(io, patcher, mean_act, data, L, H, args, log):
    probs, hist = reinforce(io, patcher, mean_act, data, L, H, args, log)
    gen = torch.Generator(device="cuda").manual_seed(args.tv_seed)
    cands = torch.bernoulli(probs.unsqueeze(1).expand(L, 10, H), generator=gen).bool()  # 10 sampled head sets
    extra = torch.stack([torch.zeros(L, H, dtype=torch.bool, device="cuda"), probs > 0.5], 1)
    ce = val_ce(io, patcher, data.val_zs, torch.cat([cands, extra], 1), args.tv_cache_budget_gb)
    best = int(ce[:10].argmin())
    sel = cands[:, best]
    locs = [(int(l), int(h), -1) for l, h in torch.nonzero(sel).tolist()]
    info = {"val_ce_candidates": ce[:10].tolist(), "val_ce_best": ce[best].item(), "val_ce_zero_shot_no_mtv": ce[10].item(),
            "val_ce_threshold_0.5": ce[11].item(), "n_heads": len(locs), "reinforce_ce_first50": float(np.mean(hist[:50])),
            "reinforce_ce_last50": float(np.mean(hist[-50:])), "bernoulli_probs": probs.cpu().tolist()}
    log(f"  MTV: {len(locs)} heads; val CE zero-shot {ce[10]:.4f} -> MTV {ce[best]:.4f} (p>0.5 set {ce[11]:.4f})")
    return locs, info


def run_fv(io, patcher, mean_act, data, L, H, args, log, model, attn_names):
    """Todd et al.: AIE on label-shuffled ICL prompts; FV = sum of top-K heads' out-projected means at layer L//3."""
    patcher.values = mean_act
    aie = torch.zeros(L, H, device="cuda")
    n = 0
    eye = torch.eye(H, dtype=torch.bool, device="cuda")
    for item in data.icl_prompts(args.tv_n_ie, shuffled=True):
        st, gold = io.build(item)
        tgt = io.target_id(gold)
        masks = torch.zeros(L, 1 + L * H, H, dtype=torch.bool, device="cuda")
        for l in range(L):
            masks[l, 1 + l * H: 1 + (l + 1) * H] = eye
        probs = torch.softmax(TV.batched_patch_logits(io, patcher, st, masks, args.tv_cache_budget_gb), -1)[:, tgt]
        aie += (probs[1:] - probs[0]).view(L, H)
        n += 1
    aie /= max(n, 1)
    flat = aie.flatten()
    top = torch.topk(flat, args.tv_fv_topk).indices.tolist()
    heads = [(i // H, i % H, float(flat[i])) for i in top]
    vec = None
    with torch.no_grad():
        for l, h, _ in heads:
            mod = TV.get_module(model, attn_names[l])
            d_in = mod.weight.shape[1]
            hd = d_in // H
            x = torch.zeros(1, 1, d_in, device=mod.weight.device, dtype=mod.weight.dtype)
            x[..., h * hd:(h + 1) * hd] = mean_act[l, h].to(x.dtype)
            out = mod(x).float().view(-1)
            vec = out if vec is None else vec + out
    edit_layer = L // 3
    # sanity check: does the FV raise P(gold first token) on held-out zero-shot train queries?
    p0, p1 = [], []
    layer_name = TV.decoder_layer_name(attn_names[edit_layer])
    for item in data.val_zs[: args.tv_n_fv_check]:
        st, gold = io.build(item)
        tgt = io.target_id(gold)
        p0.append(torch.softmax(io.full_logits(st), -1)[0, tgt].item())
        with TV.residual_add(model, layer_name, vec):
            p1.append(torch.softmax(io.full_logits(st), -1)[0, tgt].item())
    info = {"n_ie_prompts": n, "edit_layer": edit_layer, "topk": args.tv_fv_topk, "top_heads": heads,
            "aie_max": float(flat.max()), "aie_mean": float(flat.mean()), "fv_norm": float(vec.norm()),
            "val_zero_shot_p_gold_no_fv": float(np.mean(p0)) if p0 else None,
            "val_zero_shot_p_gold_with_fv": float(np.mean(p1)) if p1 else None}
    log(f"  FV: AIE max {info['aie_max']:.4g} mean {info['aie_mean']:.3g}; top heads {[(a, b) for a, b, _ in heads]};"
        f" layer {edit_layer}; |FV| {info['fv_norm']:.2f}; val zero-shot P(gold) {info['val_zero_shot_p_gold_no_fv']} -> {info['val_zero_shot_p_gold_with_fv']}")
    return {"vector": vec.cpu(), "edit_layer": edit_layer, "heads": heads, "aie": aie.cpu(), "n_layers": L,
            "n_heads": H, "mean_activations": mean_act.cpu()}, info


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, rest = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tv_bench", required=True)
    ap.add_argument("--tv_arch", required=True)
    ap.add_argument("--tv_scenarios", default="")
    ap.add_argument("--tv_method", default="both", choices=["both", "mtv", "fv"])
    ap.add_argument("--tv_n_mean", type=int, default=100)
    ap.add_argument("--tv_n_reinforce_items", type=int, default=100)
    ap.add_argument("--tv_epochs", type=int, default=600)
    ap.add_argument("--tv_samples", type=int, default=32)
    ap.add_argument("--tv_n_val", type=int, default=50)
    ap.add_argument("--tv_n_ie", type=int, default=25)
    ap.add_argument("--tv_fv_topk", type=int, default=10)
    ap.add_argument("--tv_n_fv_check", type=int, default=20)
    ap.add_argument("--tv_cache_budget_gb", type=float, default=24.0)
    ap.add_argument("--tv_seed", type=int, default=0)
    ap.add_argument("--tv_out_tag", default="")  # e.g. _smoke -> MTV/storage_v2_smoke
    tv = ap.parse_args(rest)

    torch.manual_seed(tv.tv_seed)
    random.seed(tv.tv_seed)
    np.random.seed(tv.tv_seed)
    torch.set_grad_enabled(False)

    model, tokenizer, processor, data_args = load_model(model_args, data_args, training_args)
    import eval_VLM as EV
    cfg = EV.get_mtv_config(model, model_args.model_name_or_path)
    names = cfg["attn_hook_names"]
    L, H = len(names), cfg["n_heads"]
    io = TV.ArchIO(model, tokenizer, processor, data_args, model_args.model_name_or_path)
    patcher = TV.HeadPatcher(model, names, H)

    mtv_dir = os.path.join(TV.REPO, f"MTV/storage_v2{tv.tv_out_tag}", tv.tv_arch, tv.tv_bench)
    fv_dir = os.path.join(TV.REPO, f"fv_v2{tv.tv_out_tag}", tv.tv_arch, tv.tv_bench)
    os.makedirs(mtv_dir, exist_ok=True)
    os.makedirs(fv_dir, exist_ok=True)

    scen2task = TV.BENCH_TASKS[tv.tv_bench]
    # scenario ids, plus "pooled" (all sub-tasks; loaded by cross-benchmark eval via TASKVEC_SRC_BENCH)
    req = tv.tv_scenarios.split() if tv.tv_scenarios.strip() else [str(s) for s in sorted(scen2task)] + [TV.POOLED]
    task2scens = {}
    single = len(set(scen2task.values())) == 1
    for s in req:
        if s == TV.POOLED:
            task = next(iter(scen2task.values())) if single else TV.POOLED
            task2scens.setdefault(task, []).append(TV.POOLED)
        else:
            task2scens.setdefault(scen2task[int(s)], []).append(int(s))

    def log(msg):
        print(msg, flush=True)

    for task, ss in task2scens.items():
        t0 = time.time()
        log(f"=== {tv.tv_arch} {tv.tv_bench} task={task} scenarios={ss}")
        data = TV.make_task_data(tv.tv_bench, task, seed=tv.tv_seed, n_val=tv.tv_n_val, log=log)
        probe, _ = io.build(data.icl_prompts(1)[0])
        chk = TV.check_cache_equivalence(io, patcher, probe, L, H)
        log(f"  cache check: {chk}")
        if chk["argmax_agree_of_2"] < 2 or max(chk["max_abs_prob_diff_clean"], chk["max_abs_prob_diff_patched"]) > 0.05:
            log("  WARNING: KV-cache last-token step disagrees with full forward -> using full forwards (slow)")
            io.use_cache = False
        t1 = time.time()
        mean_act, minfo = mean_activations(io, patcher, data, tv.tv_n_mean, L, H, log)
        minfo["sec_mean"] = time.time() - t1
        common = {"arch": tv.tv_arch, "model": model_args.model_name_or_path, "bench": tv.tv_bench, "task": task,
                  "scenarios": ss, "data": data.report, "cache_check": chk, **minfo, "args": vars(tv),
                  "attn_hook_names": names[:2] + ["..."]}
        if tv.tv_method in ("both", "mtv"):
            t1 = time.time()
            locs, info = run_mtv(io, patcher, mean_act, data, L, H, tv, log)
            info["sec_mtv"] = time.time() - t1
            log(f"  MTV time {info['sec_mtv']:.0f}s")
            for s in ss:
                torch.save(locs, os.path.join(mtv_dir, f"mtv_{s}.pt"))
                torch.save(mean_act.unsqueeze(2).cpu(), os.path.join(mtv_dir, f"mtv_activation_{s}.pt"))
                json.dump({**common, "method": "mtv_v2", **info}, open(os.path.join(mtv_dir, f"meta_{s}.json"), "w"), indent=1)
        if tv.tv_method in ("both", "fv"):
            t1 = time.time()
            blob, info = run_fv(io, patcher, mean_act, data, L, H, tv, log, model, names)
            info["sec_fv"] = time.time() - t1
            log(f"  FV time {info['sec_fv']:.0f}s")
            for s in ss:
                torch.save(blob, os.path.join(fv_dir, f"fv_{s}.pt"))
                json.dump({**common, "method": "fv_v2", **info}, open(os.path.join(fv_dir, f"meta_{s}.json"), "w"), indent=1)
        io.use_cache = True
        torch.cuda.empty_cache()
        log(f"  done {task} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
