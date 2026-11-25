"""Evaluation entry point for the inference-time parts of the hallucination-mitigation
baselines: VOLCANO (critique -> revise -> decide), LURE (mask + revisor rewrite),
IBD (image-biased contrastive decoding) and SPARC (selective progressive attention
recalibration).

It runs eval_VLM.main() unchanged (same datalists, prompts, decoding settings, parsing,
scoring and result layout) after replacing the model's generation entry point with the
method wrapper (see utils/halluc_common.py). Takes every eval_VLM.py argument plus:

  --halluc_method volcano|lure|ibd|sparc
  --halluc_boost_images query|all        (IBD / SPARC: which images get boosted; default query)
  --volcano_max_iters 3
  --lure_gamma 0.9 --lure_tail_ratio 0.2 --lure_mask_token [IDN]
  --ibd_eps 2.0 --ibd_alpha 15000 --ibd_beta 0.1 --ibd_early_exit_layer -1 (auto: 24/32 scaled)
  --sparc_a 1.1 --sparc_b 0.1 --sparc_t 1.5 --sparc_sel_layer -1 (auto: 20/32 scaled)
  --sparc_lo -1 --sparc_hi -1 (auto: all layers)

Env HALLUC_MAX_ITEMS=N truncates every eval datalist to N items (smoke tests only).
IBD / SPARC need attn_impl=sdpa (forced here).
Per-sample traces (feedback/revisions, masked answers, IBD/SPARC stats) are written to
halluc_trace.json next to results.json.
"""
import json
import os
import sys


def _pop_args(argv):
    spec = {
        "--halluc_method": None, "--halluc_boost_images": "query",
        "--volcano_max_iters": "3",
        "--lure_gamma": "0.9", "--lure_tail_ratio": "0.2", "--lure_mask_token": "[IDN]",
        "--ibd_eps": "2.0", "--ibd_alpha": "15000", "--ibd_beta": "0.1", "--ibd_early_exit_layer": "-1",
        "--sparc_a": "1.1", "--sparc_b": "0.1", "--sparc_t": "1.5", "--sparc_sel_layer": "-1",
        "--sparc_lo": "-1", "--sparc_hi": "-1",
    }
    out = dict(spec)
    rest = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in spec:
            out[a] = argv[i + 1]
            i += 2
            continue
        rest.append(a)
        i += 1
    return {k[2:]: v for k, v in out.items()}, rest


HA, sys.argv = _pop_args(sys.argv)
METHOD = HA["halluc_method"]
assert METHOD in ("volcano", "lure", "ibd", "sparc"), f"--halluc_method must be volcano|lure|ibd|sparc, got {METHOD}"


def _get_arg(name):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return None


_IS_INTERN = "intern" in (_get_arg("--model_name_or_path") or "").lower()
if METHOD in ("ibd", "sparc") and _IS_INTERN:
    # InternVLChatModel.from_pretrained refuses attn_implementation="sdpa"; it is loaded as
    # usual (flash) and only its Qwen3 LM is switched to SDPA after loading (see below).
    if _get_arg("--attn_impl") != "flash_attention_2":
        if "--attn_impl" in sys.argv:
            sys.argv[sys.argv.index("--attn_impl") + 1] = "flash_attention_2"
        else:
            sys.argv += ["--attn_impl", "flash_attention_2"]
elif METHOD in ("ibd", "sparc"):
    if _get_arg("--attn_impl") is None:
        sys.argv += ["--attn_impl", "sdpa"]
    elif _get_arg("--attn_impl") != "sdpa":
        print(f"[halluc] {METHOD}: forcing --attn_impl sdpa (was {_get_arg('--attn_impl')})")
        sys.argv[sys.argv.index("--attn_impl") + 1] = "sdpa"

import eval_VLM  # noqa: E402
from utils import halluc_common as hc  # noqa: E402

TRACE = []
_orig_get_model = eval_VLM.get_VLMmodel


def _patched_get_model(model_args, training_args, bnb_args, data_args):
    ret = _orig_get_model(model_args, training_args, bnb_args, data_args)
    model, tok = ret[0], ret[1]
    arch = hc.arch_of(model_args.model_name_or_path)
    import logging
    log = logging.getLogger()
    inner = hc.innermost_model(model)
    if METHOD in ("ibd", "sparc"):
        hc.route_lm_through_sdpa(inner, arch, log)
    boost = HA["halluc_boost_images"]
    if METHOD == "volcano":
        from utils.volcano import install_volcano
        install_volcano(model, arch, tok, max_iters=int(HA["volcano_max_iters"]), seed=training_args.seed,
                        logger=log, trace=TRACE)
    elif METHOD == "lure":
        from utils.lure import install_lure
        install_lure(model, arch, tok, gamma=float(HA["lure_gamma"]), tail_ratio=float(HA["lure_tail_ratio"]),
                     mask_token=HA["lure_mask_token"], logger=log, trace=TRACE)
    elif METHOD == "ibd":
        from utils.ibd import install_ibd
        st = install_ibd(model, arch, tok, eps=float(HA["ibd_eps"]), alpha=float(HA["ibd_alpha"]),
                    beta=float(HA["ibd_beta"]), early_exit_layer=int(HA["ibd_early_exit_layer"]),
                    boost=boost, logger=log)
        TRACE.append({"ibd_state": st})
    elif METHOD == "sparc":
        from utils.sparc_icl import install_sparc
        st = install_sparc(model, arch, tok, alpha=float(HA["sparc_a"]), beta=float(HA["sparc_b"]),
                           tau=float(HA["sparc_t"]), selected_layer=int(HA["sparc_sel_layer"]),
                           start_layer=int(HA["sparc_lo"]), end_layer=int(HA["sparc_hi"]),
                           boost=boost, logger=log)
        TRACE.append({"sparc_state": st})
    log.info(f"[halluc] installed {METHOD} on {arch} ({type(inner).__name__}); args={HA}")
    return ret


eval_VLM.get_VLMmodel = _patched_get_model

# per-sample hints (number of query images) for the wrappers
_orig_tqdm = eval_VLM.tqdm


def _hint_tqdm(it, *a, **k):
    for item in _orig_tqdm(it, *a, **k):
        hc.set_hints_from_item(item)
        yield item


eval_VLM.tqdm = _hint_tqdm

_MAX = int(os.environ.get("HALLUC_MAX_ITEMS", "0"))
if _MAX > 0:
    _orig_get_datalists = eval_VLM.get_datalists

    def _short_datalists(*a, **k):
        tr, te = _orig_get_datalists(*a, **k)
        for v in te.values():
            for d in v:
                d["data"] = d["data"][:_MAX]
        return tr, te
    eval_VLM.get_datalists = _short_datalists


def _write_trace():
    a = sys.argv
    g = lambda n: a[a.index(n) + 1] if n in a else None
    path = (f"./final2_eval_results/{g('--ordering')}/{g('--eval_shot')}/{g('--dataset_name')}/"
            f"scenario{g('--scenario')}/{g('--model_name')}/halluc_trace.json")
    rec = []
    for t in TRACE:
        if "sparc_state" in t:
            st = t["sparc_state"]
            rec.append({"sparc_steps": st.steps, "sparc_selected_total": st.n_selected})
        elif "ibd_state" in t:
            st = t["ibd_state"]
            n = max(st.steps, 1)
            rec.append({"ibd_steps": st.steps, "ibd_changed_tokens": st.n_changed,
                        "ibd_biased_attention_calls": st.n_bias_calls,
                        "ibd_mean_I_sim": st.sum_i_sim / n, "ibd_mean_I_con": st.sum_i_con / n})
        else:
            rec.append(t)
    try:
        with open(path, "w") as fp:
            json.dump({"method": METHOD, "args": HA, "trace": rec}, fp, indent=1, default=str)
        print("halluc trace saved at", path)
    except Exception as e:  # never fail the eval because of the trace
        print("could not write halluc trace:", e)


if __name__ == "__main__":
    eval_VLM.main()
    _write_trace()
