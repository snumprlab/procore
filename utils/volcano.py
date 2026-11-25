"""VOLCANO inference (Lee et al., NAACL 2024, arXiv 2311.07362, Algorithm 1) for the eval_VLM stack.

    R_best = M(I, Q)                                  # initial answer (normal eval call)
    for it in range(3):                               # "maximum number of iterations ... is 3"
        F  = M(I, Q, R_best)                          # feedback
        Rr = M(I, Q, R_best, F)                       # revision
        Rd = M(I, Q, {R_best, Rr} in random order)    # decision (order randomised, Sec. 3.1)
        if Rd == R_best: break
        R_best = Rr
    return R_best

The model is the VOLCANO-trained LoRA checkpoint (train_VLM.sh METHOD=volcano, trained on
kaist-ai/volcano-train feedback + revision instances). Feedback / revision prompts are the
templates of the released training data; the decision prompt follows the LLaVA-1.5
multiple-choice format the paper says it is based on (App. B.1). Feedback, revision and
decision use greedy decoding (paper App. D); the initial answer uses the harness's decoding so
that it is the exact plain-eval prediction of the same checkpoint.

ICL adaptation: Q is the *whole* ICL user turn (all demonstrations + query, every image in its
original position); it is spliced verbatim into the templates at token level, so all arches
see exactly the prompt eval_VLM built, wrapped by the VOLCANO instruction.
"""
from __future__ import annotations

import random
import re

from utils.halluc_common import GenCall, install_generate_wrapper, splice

FEEDBACK_BEFORE = "Generate the feedback given initial answer referring to question and image.\nQuestion: "
FEEDBACK_AFTER = "\nInitial answer: {a}"
REVISE_BEFORE = "Adjust the initial response considering the feedback and image.\nQuestion: "
REVISE_AFTER = "\nInitial answer: {a}\nFeedback: {f}"
DECIDE_BEFORE = ""
DECIDE_AFTER = "\nA. {a}\nB. {b}\nAnswer with the option's letter from the given choices directly."


def _choice(text, a, b):
    """Map the decision output to "A" / "B" (None if undecidable -> keep R_best)."""
    t = text.strip()
    m = re.match(r"^\(?([ABab])\)?(?:[\s.:)]|$)", t)
    if m:
        return m.group(1).upper()
    low = t.lower().rstrip(".")
    if low == a.strip().lower().rstrip(".") and low != b.strip().lower().rstrip("."):
        return "A"
    if low == b.strip().lower().rstrip(".") and low != a.strip().lower().rstrip("."):
        return "B"
    return None


def install_volcano(model, arch, tok, max_iters=3, seed=1, logger=None, trace=None):
    rng = random.Random(seed)

    def make(orig):
        def volcano_generate(*args, **kwargs):
            call = GenCall(arch, orig, args, kwargs)
            ids = call.input_ids
            out, in_ids = call.run_raw(keep_stopping=True)
            t = getattr(tok, "tokenizer", tok)
            best = t.decode(call.new_tokens(out, in_ids), skip_special_tokens=True).strip()
            steps = []
            for it in range(max_iters):
                fb = call.run_text(tok, splice(arch, ids, tok, FEEDBACK_BEFORE, FEEDBACK_AFTER.format(a=best)), greedy=True)
                rev = call.run_text(tok, splice(arch, ids, tok, REVISE_BEFORE, REVISE_AFTER.format(a=best, f=fb)), greedy=True)
                if rev.strip() == best.strip():  # nothing to decide: R_revised == R_best
                    steps.append({"iter": it + 1, "feedback": fb, "revised": rev, "decision": None,
                                  "picked_revised": False})
                    break
                swap = rng.random() < 0.5
                a, b = (rev, best) if swap else (best, rev)
                dec = call.run_text(tok, splice(arch, ids, tok, DECIDE_BEFORE, DECIDE_AFTER.format(a=a, b=b)), greedy=True)
                c = _choice(dec, a, b)
                picked_revised = (c == "A" and swap) or (c == "B" and not swap)
                steps.append({"iter": it + 1, "feedback": fb, "revised": rev, "decision": dec,
                              "revised_is": "A" if swap else "B", "picked_revised": picked_revised})
                if not picked_revised or rev.strip() == best.strip():
                    break
                best = rev
            if trace is not None:
                trace.append({"final": best, "steps": steps})
            if logger is not None and len(trace or []) <= 3:
                logger.info(f"[VOLCANO] trace: {steps}")
            return call.package_text(tok, best)
        return volcano_generate

    install_generate_wrapper(model, arch, make)
