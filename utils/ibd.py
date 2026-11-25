"""IBD: Image-Biased Decoding (Zhu et al., arXiv 2402.18476, CVPR 2025 workshops) for the
eval_VLM stack (Qwen3-VL, InternVL3.5, Idefics3, OneVision-1.5).

What is implemented (paper Sec. 3, Eqs. 3-8):
  * image-biased model theta_hat: the same weights, with every LM self-attention logit to an
    image key token shifted by +eps before the softmax (Eq. 3: c_{m,n} = eps for image keys),
    in all layers / heads. eps = 2 (paper Sec. 4.1).
  * contrastive score L_CD = logit_hat - logit (Eq. 2/8).
  * adaptive weight I = min(I_sim, I_con), I_sim = JSD(p_theta || p_theta_hat) (Eq. 5),
    I_con = JSD(p_theta || p_tilde) with p_tilde the early-exit distribution of layer 24 of a
    32-layer LM (Eq. 6) -> layer scaled to the target depth (27 for 36-layer LMs).
  * y_i ~ Softmax(logit + alpha * I * L_CD) restricted to V_head = {y: p(y) >= beta * max p}
    (Eqs. 7-8), alpha = 1.5e4, beta = 0.1 (paper defaults).

NOT implemented (stated plainly): the paper additionally prompt-tunes a small learned prompt P
(~70K params, on COCO captions) that is prepended for theta_hat. No IBD code/checkpoint is
public, and the prompt is model-specific, so this is the training-free attention-bias variant
of IBD ("IBD-attn"). Result directories are tagged accordingly (_ibd).

Both models are run in one batch of 2 rows (row 0 = original, row 1 = image-biased, the bias is
applied to row 1 only through an SDPA hook). The logits processor picks the token from the IBD
distribution and forces it on both rows so the two KV caches stay in sync.

ICL design choice: by default only the *query* image tokens get the bias (--halluc_boost_images
query). IBD is a single-image method whose intent is to strengthen reliance on the image the
answer is about; biasing demonstration images would instead push attention toward the
demonstrations' visual content, which is exactly the kind of cross-image confusion we want to
avoid. "all" is available for ablation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from utils.halluc_common import (
    GenCall, SDPA_STATE, find_lm_parts, install_generate_wrapper, install_sdpa_hook, jsd,
    materialize_mask, scaled_layer, target_image_positions,
)


class IBDState:
    def __init__(self):
        self.active = False
        self.positions = None
        self.eps = 2.0
        self.early_h = None
        # diagnostics
        self.n_bias_calls = 0
        self.steps = 0
        self.n_changed = 0      # steps where the IBD token differs from the original argmax
        self.sum_i_sim = 0.0
        self.sum_i_con = 0.0


def _dup(arch, kw):
    kw = dict(kw)
    kw["input_ids"] = kw["input_ids"].repeat(2, 1)
    if kw.get("attention_mask") is not None:
        kw["attention_mask"] = kw["attention_mask"].repeat(2, 1)
    pv = kw.get("pixel_values")
    if pv is not None:
        kw["pixel_values"] = torch.cat([pv, pv], 0)
    for k in ("image_grid_thw", "pixel_attention_mask"):
        if kw.get(k) is not None:
            kw[k] = torch.cat([kw[k], kw[k]], 0)
    for k in ("stopping_criteria",):
        kw.pop(k, None)
    return kw


class IBDLogitsProcessor(LogitsProcessor):
    def __init__(self, state, norm, lm_head, alpha, beta, do_sample, temperature, log):
        self.s, self.norm, self.lm_head = state, norm, lm_head
        self.alpha, self.beta = alpha, beta
        self.do_sample, self.T = do_sample, (temperature or 1.0)
        self.log = log

    def __call__(self, input_ids, scores):
        lp, lb = scores[0].float(), scores[1].float()
        h = self.s.early_h
        if h is not None:
            with torch.no_grad():
                le = self.lm_head(self.norm(h.to(self.lm_head.weight.dtype))).float().reshape(-1)[: lp.shape[-1]]
            i_con = jsd(lp, le)
        else:
            i_con = torch.tensor(float("inf"), device=lp.device)
        i_sim = jsd(lp, lb)
        I = torch.minimum(i_sim, i_con)
        final = lp + self.alpha * I * (lb - lp)
        p = F.softmax(lp, -1)
        final = final.masked_fill(p < self.beta * p.max(), float("-inf"))
        if self.do_sample:
            tok = torch.multinomial(F.softmax(final / self.T, -1), 1).item()
        else:
            tok = int(final.argmax())
        self.s.steps += 1
        self.s.n_changed += int(tok != int(lp.argmax()))
        self.s.sum_i_sim += float(i_sim)
        self.s.sum_i_con += float(i_con) if torch.isfinite(i_con) else 0.0
        out = torch.full_like(scores, float("-inf"))
        out[:, tok] = 0.0
        return out


def install_ibd(model, arch, tok, eps=2.0, alpha=1.5e4, beta=0.1, early_exit_layer=-1,
                boost="query", logger=None):
    state = IBDState()
    state.eps = eps
    inner = __import__("utils.halluc_common", fromlist=["innermost_model"]).innermost_model(model)
    layers = install_sdpa_hook(inner)
    _, norm, lm_head = find_lm_parts(inner)
    depth = len(layers)
    E = early_exit_layer if early_exit_layer >= 0 else scaled_layer(24, 32, depth)
    if logger:
        logger.info(f"[IBD] eps={eps} alpha={alpha} beta={beta} early-exit layer={E}/{depth} boost={boost}")

    def early_hook(mod, args, out):
        if state.active:
            h = out[0] if isinstance(out, (tuple, list)) else out
            state.early_h = h[0, -1].detach()
    layers[E - 1].register_forward_hook(early_hook)

    def bias_handler(L, q, k, mask, is_causal, scale, kw):
        if not state.active or state.positions is None or q.shape[0] != 2:
            return mask, is_causal
        m = materialize_mask(q, k, mask, is_causal)
        if m.shape[0] == 1:
            m = m.expand(2, *m.shape[1:]).clone()
        else:
            m = m.clone()
        pos = state.positions[state.positions < k.shape[-2]]
        m[1, :, :, pos] = m[1, :, :, pos] + state.eps
        state.n_bias_calls += 1
        return m, False
    SDPA_STATE.handlers.append(bias_handler)

    gen_cfg = getattr(inner, "generation_config", None)
    if arch == "intern":
        gen_cfg = getattr(inner.language_model, "generation_config", gen_cfg)

    def make(orig):
        def ibd_generate(*args, **kwargs):
            call = GenCall(arch, orig, args, kwargs)
            pos = target_image_positions(arch, call.input_ids, tok, inner, boost)
            if pos is None:
                return orig(*args, **kwargs)
            kw = _dup(arch, call.kwargs)
            do_sample = kw.get("do_sample", getattr(gen_cfg, "do_sample", False))
            T = kw.get("temperature", None) or getattr(gen_cfg, "temperature", 1.0)
            proc = IBDLogitsProcessor(state, norm, lm_head, alpha, beta, do_sample, T, None)
            # IBD picks the token itself (greedy or sampled with the harness temperature);
            # HF then just takes the argmax of the one-hot scores, so no HF warper (temperature,
            # top-k/p from generation_config) touches the raw logits IBD compares.
            kw["do_sample"] = False
            for k in ("temperature", "top_p", "top_k"):
                kw.pop(k, None)
            lp = kw.pop("logits_processor", None) or LogitsProcessorList()
            lp.append(proc)
            kw["logits_processor"] = lp
            state.positions, state.active, state.early_h = pos, True, None
            try:
                out = orig(**kw)
            finally:
                state.active = False
            if arch == "intern":
                return out[0][0:1], out[1]
            return out[0:1]
        return ibd_generate

    install_generate_wrapper(model, arch, make)
    return state
