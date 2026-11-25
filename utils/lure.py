"""LURE inference (Zhou et al., ICLR 2024, arXiv 2310.00754; code YiyangZhou/LURE) for the
eval_VLM stack.

Pipeline per query (paper Alg. 2 + generate_IDK.py / output_LURE.py):
  1. the *base* MLLM (LoRA adapter disabled) produces the answer s with token probabilities;
  2. object words (nouns, NLTK POS tags NN/NNS as in LURE's code) are replaced by the mask token
     if  -log p(first sub-token) >= gamma  (uncertainty, gamma = 0.9 = LURE default) or if they
     lie in the last 20% of the words (position, LURE code: num_replacement = int(0.2*num_words));
  3. the *revisor* (same base MLLM + the LURE LoRA trained by train_VLM.sh METHOD=lure on
     datalists/lure/train/dataset-2.json: masked hallucinatory description -> correct
     description, COCO) rewrites it with the LURE prompt
     "According to the picture, remove the information that does not exist in the following
      description: <masked> Answer:"  (exactly the training prompt of the datalist).
The mask token is "[IDN]" because that is the token used in the training datalist (the paper
writes "[IDK]"); train and test use the same token.

ICL adaptation: the revisor input is the original ICL user turn (all demonstrations + query with
their images, verbatim) followed by the LURE revision instruction, so the revisor sees the same
visual evidence as the answering model and the demonstrations' answer format. Simplification vs.
LURE's code: only the object word itself is masked (LURE's script also drops some neighbouring
words through a buggy condition).
"""
from __future__ import annotations

import contextlib
import math
import re

import torch
import torch.nn.functional as F

from utils.halluc_common import GenCall, install_generate_wrapper, splice

REVISE_AFTER = ("\nAccording to the picture, remove the information that does not exist in the "
                "following description: {d} Answer:")

_tagger = None


def _pos_tag(words):
    global _tagger
    import nltk
    if "/tmp/nltk_data" not in nltk.data.path:
        nltk.data.path.insert(0, "/tmp/nltk_data")
    return nltk.pos_tag(words)


def mask_answer(tok, gen_ids, logits, gamma=0.9, tail_ratio=0.2, mask_token="[IDN]"):
    """Return (masked_text, original_text, n_masked)."""
    t = getattr(tok, "tokenizer", tok)
    ids = gen_ids.tolist()
    # probabilities of the generated tokens from the raw logits
    probs = []
    for k, tid in enumerate(ids):
        if k < len(logits):
            probs.append(float(F.softmax(logits[k][0].float(), -1)[tid]))
        else:
            probs.append(1.0)
    # character span of each token
    spans, prev = [], ""
    for k in range(len(ids)):
        cur = t.decode(ids[:k + 1], skip_special_tokens=True)
        spans.append((len(prev), len(cur)))
        prev = cur
    text = prev
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z][A-Za-z\-']*", text)]
    if not words:
        return text, text, 0
    tags = _pos_tag([w for w, _, _ in words])
    n = len(words)
    n_tail = int(n * tail_ratio)
    tail_start = n - n_tail
    to_mask = []
    for wi, ((w, s, e), (_, tag)) in enumerate(zip(words, tags)):
        if tag not in ("NN", "NNS"):
            continue
        p = 1.0
        for k, (a, b) in enumerate(spans):
            if b > s:
                p = probs[k]
                break
        unc = -math.log(max(p, 1e-12))
        if unc >= gamma or wi >= tail_start:
            to_mask.append((s, e))
    masked = text
    for s, e in sorted(to_mask, reverse=True):
        masked = masked[:s] + mask_token + masked[e:]
    return masked, text, len(to_mask)


def install_lure(model, arch, tok, gamma=0.9, tail_ratio=0.2, mask_token="[IDN]", logger=None, trace=None):
    peft_model = model if hasattr(model, "disable_adapter") else None
    if peft_model is None:
        raise RuntimeError("LURE needs the revisor LoRA (lora_enable True + the METHOD=lure checkpoint)")

    def make(orig):
        def lure_generate(*args, **kwargs):
            call = GenCall(arch, orig, args, kwargs)
            ids = call.input_ids
            # 1) base-model answer with raw logits
            with peft_model.disable_adapter():
                out, in_ids = call.run_raw(keep_stopping=True,
                                           extra=dict(return_dict_in_generate=True, output_logits=True))
            gen = call.new_tokens(out, in_ids)
            logits = out.logits if getattr(out, "logits", None) is not None else []
            # 2) mask uncertain / late objects
            masked, orig_text, n_masked = mask_answer(tok, gen, logits, gamma, tail_ratio, mask_token)
            # 3) revisor rewrite
            rev = call.run_text(tok, splice(arch, ids, tok, "", REVISE_AFTER.format(d=masked)))
            if trace is not None:
                trace.append({"initial": orig_text, "masked": masked, "n_masked": n_masked, "revised": rev})
                if logger is not None and len(trace) <= 3:
                    logger.info(f"[LURE] {trace[-1]}")
            return call.package_text(tok, rev)
        return lure_generate

    install_generate_wrapper(model, arch, make)
