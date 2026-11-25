"""Shared MimIC helpers (Jiang et al., CVPR 2025).

* ``mimic_do_shift``: the per-head attention shift used by every *_mimic model file.
  It follows the official implementation (Kamichanw/MimIC, src/shift_encoder.py,
  AttnApproximator.do_shift):

      out = SA(q, K, V) + mu * v_shift,   mu = Z1 / (Z1 + Z2)
      log Z1 = f_l(q)  (learnable, per head),  log Z2 = logsumexp_k(q k^T / sqrt(d))

  applied after value aggregation and before o_proj. As in the official code, Z2 is
  computed over all keys passed in (no causal mask).
* ``mimic_shift_disabled``: context manager that turns the shift off (used for the
  teacher pass = frozen original model with demonstrations).
* ``strip_demos``: builds the query-only (zero-demo) version of an interleaved ICL
  prompt: keeps the instruction text before the first image + the query image(s) +
  query text, drops the demonstrations.
* ``common_suffix_masks``: token masks for the MimIC layer-wise alignment loss. The
  official code aligns the hidden states of the query part (query + answer tokens),
  which is the token suffix shared by the ICL sequence and the query-only sequence.
"""
import contextlib

import torch


def _flag(strategy, name):
    member = getattr(type(strategy), name, None)
    return member is not None and (int(strategy) & int(member)) != 0


def mimic_do_shift(approx, layer_idx, query_states, key_states, attn_output):
    """query_states [B,H,T,hd], key_states [B,H,K,hd] (already repeat_kv'ed), attn_output [B,T,H,hd]."""
    layer_idx = int(layer_idx)
    if layer_idx < 0 or layer_idx >= approx.num_layers:
        return attn_output
    if not approx.attn_shift_handles[layer_idx].active:
        return attn_output
    shift_p = getattr(approx, "attn_shift", None)
    if shift_p is None:
        return attn_output

    bsz, nh, T, hd = query_states.shape
    multi_head = _flag(approx.attn_strategy, "MULTI_HEAD")
    learnable = _flag(approx.attn_strategy, "LEARNABLE_SHIFT_SCALE") and getattr(approx, "log_Z1_lin", None) is not None

    if multi_head:
        shift = shift_p[layer_idx][None, None, :, :]
    else:
        shift = shift_p[layer_idx][None, None, :].view(1, 1, nh, hd)
    shift = shift.to(device=attn_output.device, dtype=attn_output.dtype)

    if not learnable:
        return attn_output + shift

    if key_states is None:
        raise ValueError("MimIC learnable mu needs key_states")
    # log_Z2 = logsumexp_k(q k^T / sqrt(d)) -> [B,T,H]; query-chunked to bound memory
    K = key_states.shape[-2]
    step = max(1, 4_000_000 // max(1, K))
    kT = key_states.transpose(-2, -1)
    chunks = []
    for i in range(0, T, step):
        s = torch.matmul(query_states[:, :, i:i + step], kT).float() / (hd ** 0.5)
        chunks.append(torch.logsumexp(s, dim=-1))
    log_Z2 = torch.cat(chunks, dim=-1).transpose(-2, -1)  # [B,T,H]

    q_t = query_states.transpose(1, 2)  # [B,T,H,hd]
    lin = approx.log_Z1_lin[layer_idx]
    if multi_head:
        w = lin.weight
        log_Z1 = torch.einsum("btnd,nd->btn", q_t.to(w.dtype), w.to(q_t.device)) + lin.bias.to(device=q_t.device, dtype=w.dtype)
        mu = torch.exp(log_Z1.float() - torch.logaddexp(log_Z1.float(), log_Z2)).unsqueeze(-1)  # [B,T,H,1]
    else:
        q_flat = q_t.reshape(bsz, T, -1)
        log_Z1 = torch.nn.functional.linear(q_flat.to(lin.weight.dtype), lin.weight.to(q_flat.device),
                                            None if lin.bias is None else lin.bias.to(q_flat.device))
        log_Z2m = log_Z2.mean(-1, keepdim=True)
        mu = torch.exp(log_Z1.float() - torch.logaddexp(log_Z1.float(), log_Z2m)).unsqueeze(-1).expand(bsz, T, nh, 1)
    return attn_output + mu.to(attn_output.dtype) * shift


def find_attn_approx(model):
    for m in model.modules():
        a = getattr(m, "attn_approx", None)
        if a is not None and hasattr(a, "attn_shift_handles"):
            return a
    return None


@contextlib.contextmanager
def mimic_shift_disabled(model):
    approx = find_attn_approx(model)
    if approx is None:
        yield None
        return
    prev = [h.active for h in approx.attn_shift_handles]
    approx.disable_all()
    try:
        yield approx
    finally:
        for h, a in zip(approx.attn_shift_handles, prev):
            h.active = a


def strip_demos(content, image_token):
    """Return (query_only_content, n_query_images).

    The query images are the trailing run of image tokens separated only by whitespace
    (e.g. 2 for operator_induction_interleaved). The text before the first image
    (task instruction) is kept.
    """
    parts = content.split(image_token)
    n = len(parts) - 1
    if n <= 0:
        return content, 0
    s = n - 1
    while s >= 1 and parts[s].strip() == "":
        s -= 1
    query = parts[0] + image_token + image_token.join(parts[s + 1:])
    return query, n - s


def common_suffix_masks(full_ids, full_attn, q_ids, q_attn):
    """Right-aligned masks over the longest common token suffix of each (full, query) pair.

    Returns (full_mask, query_mask) bool tensors with equal numbers of True per row.
    """
    B = full_ids.shape[0]
    fm = torch.zeros_like(full_ids, dtype=torch.bool)
    qm = torch.zeros_like(q_ids, dtype=torch.bool)
    for b in range(B):
        fidx = torch.nonzero(full_attn[b].bool(), as_tuple=False).squeeze(-1) if full_attn is not None else torch.arange(full_ids.shape[1], device=full_ids.device)
        qidx = torch.nonzero(q_attn[b].bool(), as_tuple=False).squeeze(-1) if q_attn is not None else torch.arange(q_ids.shape[1], device=q_ids.device)
        f = full_ids[b, fidx]
        q = q_ids[b, qidx]
        n = min(len(f), len(q))
        eq = (f[len(f) - n:].flip(0) == q[len(q) - n:].flip(0)).long()
        L = int(torch.cumprod(eq, 0).sum().item()) if n > 0 else 0
        if L > 0:
            fm[b, fidx[len(f) - L:]] = True
            qm[b, qidx[len(q) - L:]] = True
    return fm, qm
