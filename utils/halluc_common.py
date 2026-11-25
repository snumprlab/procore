"""Shared plumbing for the inference-time hallucination-mitigation baselines
(VOLCANO, LURE, IBD, SPARC) used by eval_VLM_halluc.py.

Design: eval_VLM.py's evaluate_* routines call one generation entry point per arch
(``model.generate`` for Qwen3-VL / OneVision-1.5 / Idefics3, ``InternVLChatModel.generate``
via ``model.chat`` for InternVL3.5). We replace that bound method on the innermost HF model
with a method-specific wrapper, so the prompt construction, decoding settings, answer
parsing and scoring of eval_VLM.py stay identical to the plain baseline.

This module provides:
  * arch detection and LM part lookup (decoder layers / final norm / lm_head),
  * per-sample hints (number of query images) set by eval_VLM_halluc.py,
  * image-token block localisation (query image vs. demonstration images),
  * prompt splicing at token level (wrap the full ICL user turn in a new instruction while
    keeping every image placeholder where it was),
  * a re-runnable view of the original generate call,
  * a patched ``torch.nn.functional.scaled_dot_product_attention`` that knows which LM
    decoder layer is running, so IBD / SPARC can bias / read attention for any arch.
"""
from __future__ import annotations

import contextlib
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# per-sample hints (filled by eval_VLM_halluc.py's tqdm hook)
# ---------------------------------------------------------------------------
HINTS = {"n_query_images": 1, "sample_idx": -1}


def set_hints_from_item(item):
    """Infer the number of query images of the next sample from the dataset item.
    eval_VLM's own convention (parse_image_qa_pairs): a prompt containing "<image> <image>"
    has two images per example, otherwise one."""
    text = None
    if isinstance(item, dict):
        if "conversations" in item:
            text = item["conversations"][0]["value"]
        elif "prompt" in item:
            p = item["prompt"]
            text = p[0] if isinstance(p, (list, tuple)) and p else p
    HINTS["sample_idx"] += 1
    if isinstance(text, str):
        HINTS["n_query_images"] = 2 if "<image> <image>" in text else 1
    else:
        HINTS["n_query_images"] = 1


# ---------------------------------------------------------------------------
# arch helpers
# ---------------------------------------------------------------------------
def arch_of(model_name_or_path: str) -> str:
    n = model_name_or_path.lower()
    if "intern" in n:
        return "intern"
    if "idefics" in n:
        return "idefics"
    if "onevision" in n:
        return "onevision"
    if "qwen" in n:
        return "qwen"
    raise ValueError(f"unsupported arch for halluc baselines: {model_name_or_path}")


def innermost_model(model):
    """Strip PEFT wrappers -> the HF model whose bound ``generate`` eval_VLM ends up calling."""
    m = model
    for _ in range(4):
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        elif m.__class__.__name__ == "LoraModel" and hasattr(m, "model"):
            m = m.model
        else:
            break
    return m


def find_lm_parts(model):
    """Return (decoder_layers ModuleList, final_norm, lm_head) of the language model."""
    best, best_name = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith("layers"):
            low = name.lower()
            if any(k in low for k in ("visual", "vision", "vit", "encoder.")):
                continue
            if best is None or len(mod) > len(best):
                best, best_name = mod, name
    if best is None:
        raise RuntimeError("could not find LM decoder layers")
    parent_name = best_name.rsplit(".", 1)[0]
    parent = model.get_submodule(parent_name)
    norm = getattr(parent, "norm", None)
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if lm_head is None:
        for name, mod in model.named_modules():
            if name.endswith("lm_head"):
                lm_head = mod
    return best, norm, lm_head


def scaled_layer(paper_layer: int, paper_depth: int, depth: int) -> int:
    """Map a layer index chosen for a `paper_depth`-layer LM to a `depth`-layer LM."""
    if depth == paper_depth:
        return paper_layer
    return int(round(paper_layer * depth / paper_depth))


# ---------------------------------------------------------------------------
# image token blocks
# ---------------------------------------------------------------------------
def _tok_id(tok, s):
    t = getattr(tok, "tokenizer", tok)
    i = t.convert_tokens_to_ids(s)
    if i is None or (hasattr(t, "unk_token_id") and i == t.unk_token_id):
        return None
    return i


def image_blocks(arch: str, ids: torch.Tensor, tok, model) -> List[torch.Tensor]:
    """List of 1-D LongTensors with the positions of the visual tokens of each image, in
    prompt order (one entry per *image*, sub-tiles merged)."""
    ids = ids.reshape(-1)
    dev = ids.device
    blocks: List[torch.Tensor] = []
    if arch in ("qwen", "onevision"):
        cfg = getattr(model, "config", None)
        vs = getattr(cfg, "vision_start_token_id", None) or _tok_id(tok, "<|vision_start|>")
        ve = getattr(cfg, "vision_end_token_id", None) or _tok_id(tok, "<|vision_end|>")
        starts = (ids == vs).nonzero().reshape(-1).tolist()
        ends = (ids == ve).nonzero().reshape(-1).tolist()
        j = 0
        for s in starts:
            while j < len(ends) and ends[j] <= s:
                j += 1
            if j >= len(ends):
                break
            if ends[j] > s + 1:
                blocks.append(torch.arange(s + 1, ends[j], device=dev))
            j += 1
    elif arch == "intern":
        a, b = _tok_id(tok, "<img>"), _tok_id(tok, "</img>")
        ctx = _tok_id(tok, "<IMG_CONTEXT>")
        starts = (ids == a).nonzero().reshape(-1).tolist()
        ends = (ids == b).nonzero().reshape(-1).tolist()
        for s, e in zip(starts, ends):
            pos = torch.arange(s + 1, e, device=dev)
            pos = pos[ids[pos] == ctx]
            if pos.numel():
                blocks.append(pos)
    elif arch == "idefics":
        img = getattr(getattr(model, "config", None), "image_token_id", None) or _tok_id(tok, "<image>")
        glob = _tok_id(tok, "<global-img>")
        is_img = (ids == img)
        pos = is_img.nonzero().reshape(-1).tolist()
        # split into contiguous runs
        runs, cur = [], []
        for p in pos:
            if cur and p != cur[-1] + 1:
                runs.append(cur)
                cur = []
            cur.append(p)
        if cur:
            runs.append(cur)
        group: List[int] = []
        for r in runs:
            group.extend(r)
            # the global-image run (preceded by <global-img>) closes an image
            if glob is None or (r[0] > 0 and int(ids[r[0] - 1]) == glob):
                blocks.append(torch.tensor(group, device=dev))
                group = []
        if group:
            blocks.append(torch.tensor(group, device=dev))
    return blocks


def target_image_positions(arch, ids, tok, model, which="query") -> Optional[torch.Tensor]:
    blocks = image_blocks(arch, ids, tok, model)
    if not blocks:
        return None
    if which == "all":
        return torch.cat(blocks)
    n = max(1, int(HINTS.get("n_query_images", 1)))
    return torch.cat(blocks[-n:])


# ---------------------------------------------------------------------------
# prompt splicing
# ---------------------------------------------------------------------------
def _find_sub(seq: List[int], sub: List[int], start=0, last=False):
    hit = -1
    n = len(sub)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == sub:
            if not last:
                return i
            hit = i
    return hit


def user_span(arch, ids: torch.Tensor, tok):
    """(start, end) of the user content in the chat-formatted prompt: ids[start:end] is the
    whole ICL user turn (demonstrations + query, images included)."""
    t = getattr(tok, "tokenizer", tok)
    seq = ids.reshape(-1).tolist()
    if arch == "idefics":
        head = t.encode("User:", add_special_tokens=False)
        eot = t.convert_tokens_to_ids("<end_of_utterance>")
    else:
        head = t.encode("<|im_start|>user\n", add_special_tokens=False)
        eot = t.convert_tokens_to_ids("<|im_end|>")
    s = _find_sub(seq, head)
    s = (s + len(head)) if s >= 0 else 1
    e = len(seq) - 1 - seq[::-1].index(eot) if eot in seq else len(seq)
    if e <= s:
        e = len(seq)
    return s, e


_DEBUG_PRINTED = {"n": 0}


def debug_print(tok, ids, tag=""):
    """HALLUC_DEBUG=1: print the first few spliced prompts with image tokens collapsed."""
    import os
    import re
    if not os.environ.get("HALLUC_DEBUG") or _DEBUG_PRINTED["n"] >= 6:
        return
    _DEBUG_PRINTED["n"] += 1
    t = getattr(tok, "tokenizer", tok)
    txt = t.decode(ids.reshape(-1), skip_special_tokens=False)
    txt = re.sub(r"(<\|image_pad\|>|<IMG_CONTEXT>|<image>)+", "<IMG*>", txt)
    print(f"[HALLUC_DEBUG {tag}]\n{txt}\n[/HALLUC_DEBUG]", flush=True)


def splice(arch, ids: torch.Tensor, tok, before: str, after: str, replace_content: Optional[torch.Tensor] = None):
    """new_ids = prefix + before + <user content> + after + suffix (1 x L LongTensor)."""
    t = getattr(tok, "tokenizer", tok)
    s, e = user_span(arch, ids, tok)
    flat = ids.reshape(-1)
    content = flat[s:e] if replace_content is None else replace_content
    b = torch.tensor(t.encode(before, add_special_tokens=False), dtype=flat.dtype, device=flat.device)
    a = torch.tensor(t.encode(after, add_special_tokens=False), dtype=flat.dtype, device=flat.device)
    new = torch.cat([flat[:s], b, content, a, flat[e:]]).unsqueeze(0)
    debug_print(tok, new, "splice")
    return new


# ---------------------------------------------------------------------------
# re-runnable view of the original generate call
# ---------------------------------------------------------------------------
class GenCall:
    """Normalises the arch-specific generate call so that it can be re-issued with new
    input_ids (same images, same decoding settings)."""

    DROP = ("stopping_criteria",)

    def __init__(self, arch, orig_generate, args, kwargs):
        self.arch = arch
        self.orig = orig_generate
        kw = dict(kwargs)
        args = list(args)
        if self.arch == "intern":
            names = ["pixel_values", "input_ids", "attention_mask"]
            for n, v in zip(names, args):
                kw[n] = v
        else:
            if args:
                kw["input_ids"] = args[0]
        if kw.get("input_ids") is not None and kw["input_ids"].dim() == 1:
            kw["input_ids"] = kw["input_ids"].unsqueeze(0)
        self.kwargs = kw
        self.input_ids = kw["input_ids"]

    def gen_settings(self):
        k = self.kwargs
        return dict(do_sample=bool(k.get("do_sample", False)), temperature=k.get("temperature", None))

    def run_raw(self, input_ids=None, greedy=False, extra=None, keep_stopping=False):
        kw = dict(self.kwargs)
        if input_ids is not None:
            kw["input_ids"] = input_ids
            kw["attention_mask"] = torch.ones_like(input_ids)
            if not keep_stopping:
                for d in self.DROP:
                    kw.pop(d, None)
        if greedy:
            kw["do_sample"] = False
            kw.pop("temperature", None)
            kw.pop("top_p", None)
            kw.pop("top_k", None)
        if extra:
            kw.update(extra)
        out = self.orig(**kw)
        if self.arch == "intern":
            out = out[0]
        return out, kw["input_ids"]

    def new_tokens(self, out, in_ids):
        seqs = out.sequences if hasattr(out, "sequences") else out
        if self.arch == "intern":
            return seqs[0]
        return seqs[0, in_ids.shape[1]:]

    def run_text(self, tok, input_ids=None, greedy=False):
        out, in_ids = self.run_raw(input_ids, greedy=greedy)
        t = getattr(tok, "tokenizer", tok)
        return t.decode(self.new_tokens(out, in_ids), skip_special_tokens=True).strip()

    def package(self, tok, answer_ids: torch.Tensor):
        """Return value in the format eval_VLM expects from the original call."""
        ans = answer_ids.reshape(1, -1).to(self.input_ids.device)
        if self.arch == "intern":
            return ans, None
        return torch.cat([self.input_ids, ans], dim=1)

    def package_text(self, tok, text: str):
        t = getattr(tok, "tokenizer", tok)
        if self.arch == "idefics":
            # eval_VLM's idefics branch keeps the text after the last "\nassistant: " of the
            # decoded prompt+generation; a generated answer starts with a space.
            text = " " + text
        ids = torch.tensor(t.encode(text, add_special_tokens=False), dtype=torch.long)
        return self.package(tok, ids)


def install_generate_wrapper(model, arch, make_wrapper: Callable):
    """Replace the generate entry point eval_VLM uses. `make_wrapper(orig)` -> new fn(*a, **k)."""
    target = innermost_model(model)
    orig = target.generate
    target.generate = make_wrapper(orig)
    return target


# ---------------------------------------------------------------------------
# SDPA hook with LM-layer tracking
# ---------------------------------------------------------------------------
class _SDPAState:
    layer: Optional[int] = None
    handlers: list = []


SDPA_STATE = _SDPAState()
_ORIG_SDPA = F.scaled_dot_product_attention


def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    L = SDPA_STATE.layer
    if L is not None and SDPA_STATE.handlers:
        for h in SDPA_STATE.handlers:
            attn_mask, is_causal = h(L, query, key, attn_mask, is_causal, scale, kw)
    return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, scale=scale, **kw)


def install_sdpa_hook(model):
    """Patch torch SDPA once and register layer-tracking hooks on the LM decoder layers."""
    if F.scaled_dot_product_attention is not _patched_sdpa:
        F.scaled_dot_product_attention = _patched_sdpa
        torch.nn.functional.scaled_dot_product_attention = _patched_sdpa
    layers, _, _ = find_lm_parts(model)
    for i, layer in enumerate(layers):
        def pre(mod, args, i=i):
            SDPA_STATE.layer = i
        def post(mod, args, out):
            SDPA_STATE.layer = None
        layer.register_forward_pre_hook(pre)
        layer.register_forward_hook(post)
    return layers


def materialize_mask(query, key, attn_mask, is_causal):
    """Float additive mask (B|1, 1|H, q, k) equivalent to (attn_mask, is_causal)."""
    q_len, k_len = query.shape[-2], key.shape[-2]
    dt = query.dtype
    neg = torch.finfo(dt).min
    if attn_mask is None:
        m = torch.zeros(1, 1, q_len, k_len, dtype=dt, device=query.device)
        if is_causal and q_len > 1:
            qi = torch.arange(q_len, device=query.device)[:, None] + (k_len - q_len)
            ki = torch.arange(k_len, device=query.device)[None, :]
            m = m.masked_fill((ki > qi)[None, None], neg)
        return m
    if attn_mask.dtype == torch.bool:
        return torch.zeros(attn_mask.shape, dtype=dt, device=query.device).masked_fill(~attn_mask, neg)
    return attn_mask.to(dt)


def unify_onevision_sdpa(model):
    """OneVision-1.5's decoder layers instantiate the *eager* attention class unless flash
    attention is selected, but its text model drops the causal mask when the config says
    "sdpa". Rebind every LM attention module to the SDPA forward of the same class family so
    that attn_impl=sdpa is correct (causal) and goes through torch SDPA (needed by the hook)."""
    try:
        from models.llavaonevision1_5.modeling_llavaonevision1_5 import LLaVAOneVision1_5_SdpaAttention
    except Exception:
        return 0
    import types
    n = 0
    layers, _, _ = find_lm_parts(model)
    for layer in layers:
        attn = layer.self_attn
        def _fwd(self, *a, img_pos_ranges=None, **k):  # eager-class-only kwarg
            return LLaVAOneVision1_5_SdpaAttention.forward(self, *a, **k)
        attn.forward = types.MethodType(_fwd, attn)
        n += 1
    return n


def route_lm_through_sdpa(inner, arch, log=None):
    """Make the LM self-attention of every arch go through torch SDPA (required by the hook).
      qwen:      attn_impl=sdpa already uses HF sdpa_attention_forward.
      onevision: decoder layers use the eager class unless flash -> rebind to the SDPA forward.
      idefics:   My_LlamaAttention routes "sdpa" to its eager function -> register an alias
                 "halluc_sdpa" (= HF sdpa_attention_forward); the mask builder then always
                 builds the 4D causal mask (non-flash/flex/sdpa branch).
      intern:    loaded with flash (from_pretrained refuses sdpa); switch the Qwen3 LM to sdpa.
    """
    msg = ""
    if arch == "onevision":
        msg = f"OneVision-1.5: {unify_onevision_sdpa(inner)} LM attention modules bound to the SDPA forward"
    elif arch == "idefics":
        from transformers import AttentionInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        AttentionInterface.register("halluc_sdpa", sdpa_attention_forward)
        layers, _, _ = find_lm_parts(inner)
        cfg = layers[0].self_attn.config
        cfg._attn_implementation = "halluc_sdpa"
        msg = f"Idefics3: LM attention routed through HF sdpa_attention_forward (config id {id(cfg)})"
    elif arch == "intern":
        lm = innermost_model(inner.language_model)
        lm.config.attn_implementation = "sdpa"      # read by My_Qwen3Attention at call time
        lm.config._attn_implementation = "sdpa"     # read by the mask builder
        if hasattr(inner.config, "llm_config"):
            inner.config.llm_config.attn_implementation = "sdpa"
        msg = "InternVL3.5: Qwen3 LM attention switched to sdpa after loading"
    if log is not None and msg:
        log.info("[halluc] " + msg)


def jsd(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence (nats) between softmax(p_logits) and softmax(q_logits), per row."""
    p = F.softmax(p_logits.float(), -1)
    q = F.softmax(q_logits.float(), -1)
    m = 0.5 * (p + q)
    lm = torch.log(m.clamp_min(1e-12))
    kl_pm = (p * (torch.log(p.clamp_min(1e-12)) - lm)).sum(-1)
    kl_qm = (q * (torch.log(q.clamp_min(1e-12)) - lm)).sum(-1)
    return 0.5 * (kl_pm + kl_qm)
