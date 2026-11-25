"""SPARC: Selective Progressive Attention ReCalibration (Jung et al., ICML 2025) adapted to
the eval_VLM stack for Qwen3-VL (36 layers), InternVL3.5-8B (Qwen3 LM, 36 layers),
OneVision-1.5-8B (36 layers) and Idefics3-8B (Llama-3, 32 layers).

The algorithm is the one of the original SPARC code (attn_util.py, ported earlier in
utils/sparc.py for eager attention only):
  * at the selection layer, track the attention a_t of the current (last) token to every
    visual token, averaged over heads; keep an EMA  ema <- (1-beta)*a_t + beta*ema
    (initialised with the last prompt token's attention at prefill);
  * at each decoding step select visual tokens whose relative increase
    (a_t - ema) / ema >= tau;
  * at the next step, multiply the cached *value* vectors of the selected tokens by alpha in
    every layer of [start, end] (in place, so repeatedly selected tokens are progressively
    amplified).
Defaults (LLaVA-1.5, 32 layers): alpha=1.1, beta=0.1, tau=1.5, selection layer 20, layers 0-31.
For other depths the selection layer is scaled proportionally (20/32 -> 22 of 36) and the
recalibrated range covers all layers (0..depth-1), i.e. the same relative placement.

Differences from utils/sparc.py: it is not used (it replaced the attention forward with an
eager re-implementation, crashed on Idefics3 because DynamicCache.get_usable_length no longer
exists in transformers 4.57, and hard-coded layers 0-31). Here the attention modules are left
untouched: the attention row of the selection layer is recomputed from the q/k tensors that
reach torch SDPA (one query row, cheap), and value recalibration is done on the DynamicCache
from a decoder-layer pre-hook. This works for every arch with attn_impl=sdpa.

ICL design choice: by default only the *query* image's visual tokens are candidates for
selection/recalibration (--halluc_boost_images query). SPARC's intent is to counter the decay
of attention to the image being described during long generation; demonstration images are
context, and amplifying them would steer the answer toward the demonstrations' content.
"all" is available for ablation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

from utils.halluc_common import (
    GenCall, SDPA_STATE, find_lm_parts, innermost_model, install_generate_wrapper,
    install_sdpa_hook, scaled_layer, target_image_positions,
)


class SPARCState:
    def __init__(self):
        self.active = False
        self.positions = None
        self.ema = None
        self.indices1 = None
        self.indices2 = None
        self.cache = None
        self.n_selected = 0
        self.steps = 0


def install_sparc(model, arch, tok, alpha=1.1, beta=0.1, tau=1.5, selected_layer=-1,
                  start_layer=-1, end_layer=-1, boost="query", logger=None):
    st = SPARCState()
    inner = innermost_model(model)
    layers = install_sdpa_hook(inner)
    depth = len(layers)
    sel = selected_layer if selected_layer >= 0 else scaled_layer(20, 32, depth)
    lo = start_layer if start_layer >= 0 else 0
    hi = end_layer if end_layer >= 0 else depth - 1
    if logger:
        logger.info(f"[SPARC] alpha={alpha} beta={beta} tau={tau} selection layer={sel} "
                    f"recalibrated layers={lo}..{hi} of {depth} boost={boost}")

    def read_handler(L, q, k, mask, is_causal, scale, kw):
        if not st.active or L != sel or st.positions is None:
            return mask, is_causal
        with torch.no_grad():
            ql = q[:, :, -1:, :].float()
            kk = k.float()
            if kk.shape[1] != ql.shape[1]:
                kk = kk.repeat_interleave(ql.shape[1] // kk.shape[1], dim=1)
            sc = scale if scale is not None else ql.shape[-1] ** -0.5
            s = (ql @ kk.transpose(-1, -2)) * sc  # B,H,1,K
            if mask is not None:
                m = mask[..., -1:, :]
                if m.dtype == torch.bool:
                    s = s.masked_fill(~m, float("-inf"))
                else:
                    s = s + m.float()
            p = F.softmax(s, -1)[0, :, 0, :].mean(0)  # K
            pos = st.positions[st.positions < p.shape[-1]]
            a = p[pos]
            if q.shape[-2] > 1 or st.ema is None:  # prefill
                st.ema = a
            else:
                ratio = (a - st.ema) / st.ema.clamp_min(1e-12)
                st.indices1 = pos[ratio >= tau]
                st.n_selected += int(st.indices1.numel())
                st.steps += 1
                st.ema = (1 - beta) * a + beta * st.ema
        return mask, is_causal
    SDPA_STATE.handlers.append(read_handler)

    def pre_hook(mod, args, kwargs, i):
        if not st.active:
            return
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is None or hs.shape[-2] != 1:
            return  # prefill: nothing to recalibrate yet
        cache = st.cache
        if cache is None:
            for v in list(kwargs.values()) + list(args[1:]):
                if isinstance(v, Cache):
                    cache = st.cache = v
                    break
        if cache is None:
            return
        if i == 0:
            st.indices2 = st.indices1
        if lo <= i <= hi and st.indices2 is not None and st.indices2.numel() > 0:
            layer_cache = cache.layers[i] if hasattr(cache, "layers") else None
            vals = getattr(layer_cache, "values", None) if layer_cache is not None else cache.value_cache[i]
            if vals is not None:
                vals[:, :, st.indices2] *= alpha

    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(lambda m, a, k, i=i: pre_hook(m, a, k, i), with_kwargs=True)

    def make(orig):
        def sparc_generate(*args, **kwargs):
            call = GenCall(arch, orig, args, kwargs)
            pos = target_image_positions(arch, call.input_ids, tok, inner, boost)
            st.positions, st.ema, st.indices1, st.indices2, st.cache = pos, None, None, None, None
            st.active = pos is not None
            try:
                return orig(*args, **kwargs)
            finally:
                st.active = False
                st.cache = None
        return sparc_generate

    install_generate_wrapper(model, arch, make)
    return st
