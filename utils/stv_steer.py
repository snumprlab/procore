"""STV (Ma et al., AAAI 2026, "Where and what matters: sensitivity-aware task vectors
for many-shot multimodal ICL") -- steering artifact + activation-replacement hook.

Shared by ``stv_train.py`` (builds the artifact) and ``eval_VLM.py`` (applies it).

What is steered
    The per-head input of every language-model ``self_attn.o_proj`` (= the attention-head
    outputs, as in MTV/STV).  For each selected (layer, head) the activation at the
    *generation position* is REPLACED by the vector chosen from that head's cluster bank.

Positions
    eval  : the last position of every forward call.  With KV-cache generation that is the
            last prompt token at prefill and every newly generated token afterwards
            (same convention as MTV/STV reference code: ``inputs[:, -1, head] = value``).
    train : teacher-forced equivalent -- every position >= P-1 (P = prompt length), i.e.
            exactly the positions that eval would steer while producing the answer tokens.

Artifact (torch.save'd dict, ``format == "stv_v1"``)
    locations [K,2] long (layer, head) | bank [K,C,head_dim] float | logits [K,C] float
    selected [K] long | values [K,head_dim] float (= bank[k, selected[k]])
    n_layers, n_heads, head_dim, model_name_or_path, config, stats
"""
import os
import re

import torch

STV_FORMAT = "stv_v1"
_SKIP = ("visual", "vision", "vit", "mm_projector", "mlp1", "connector")


def find_lm_o_proj(model):
    """Return [(layer_idx, module_name, module)] for the language model's attention o_proj."""
    found = {}
    for name, mod in model.named_modules():
        if not name.endswith("self_attn.o_proj"):
            continue
        low = name.lower()
        if any(s in low for s in _SKIP):
            continue
        m = re.search(r"layers\.(\d+)\.self_attn\.o_proj$", name)
        if m is None:
            continue
        li = int(m.group(1))
        if li in found:
            raise RuntimeError(f"[STV] duplicate LM layer {li}: {found[li][0]} vs {name}")
        found[li] = (name, mod)
    if not found:
        raise RuntimeError("[STV] no language-model self_attn.o_proj modules found")
    idx = sorted(found)
    if idx != list(range(len(idx))):
        raise RuntimeError(f"[STV] non-contiguous LM layer indices: {idx[:5]}...")
    return [(i, found[i][0], found[i][1]) for i in idx]


def get_text_num_heads(model):
    cfg = model.config
    for c in (getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None), cfg):
        if c is not None and getattr(c, "num_attention_heads", None):
            return int(c.num_attention_heads)
    raise RuntimeError("[STV] cannot determine num_attention_heads")


def lm_geometry(model):
    layers = find_lm_o_proj(model)
    n_heads = get_text_num_heads(model)
    width = layers[0][2].in_features
    if width % n_heads:
        raise RuntimeError(f"[STV] o_proj width {width} not divisible by {n_heads} heads")
    return layers, n_heads, width // n_heads


class STVSteerer:
    """Forward-pre-hooks on the selected o_proj modules that replace head activations.

    ``set_values(locations, values)`` chooses what is written; ``mode`` is "last"
    (eval) or an int start position (teacher-forced training: positions >= start).
    ``enabled=False`` turns the hooks into no-ops (used to read clean activations).
    """

    def __init__(self, model):
        self.layers, self.n_heads, self.head_dim = lm_geometry(model)
        self.mode = "last"
        self.enabled = True
        self._per_layer = {}
        self._handles = []

    def set_values(self, locations, values):
        """locations: LongTensor/list [K,2]; values: Tensor [K, head_dim]."""
        per_layer = {}
        locs = torch.as_tensor(locations).tolist()
        for k, (l, h) in enumerate(locs):
            per_layer.setdefault(int(l), ([], []))
            per_layer[int(l)][0].append(int(h))
            per_layer[int(l)][1].append(values[k])
        new = {}
        for l, (hs, vs) in per_layer.items():
            new[l] = (torch.tensor(hs, dtype=torch.long), torch.stack(vs, 0))
        self._per_layer = new
        self._cache = {}
        needed = set(new)
        hooked = {l for l, _ in self._handles}
        if needed != hooked:
            self.remove()
            for l, name, mod in self.layers:
                if l in needed:
                    self._handles.append((l, mod.register_forward_pre_hook(self._make_hook(l))))

    def _make_hook(self, layer):
        def hook(module, args):
            if not self.enabled:
                return None
            x = args[0]
            hs, vs = self._per_layer[layer]
            key = (layer, x.device, x.dtype)
            cached = self._cache.get(key)
            if cached is None:
                cached = (hs.to(x.device), vs.to(device=x.device, dtype=x.dtype))
                self._cache[key] = cached
            hs_d, vs_d = cached
            B, T, D = x.shape
            x = x.clone().view(B, T, self.n_heads, self.head_dim)
            if self.mode == "last":
                x[:, -1, hs_d, :] = vs_d.unsqueeze(0)
            else:
                start = int(self.mode)
                n = T - start
                if n > 0:
                    x[:, start:, hs_d, :] = vs_d.view(1, 1, *vs_d.shape).expand(B, n, *vs_d.shape)
            return (x.view(B, T, D),) + tuple(args[1:])
        return hook

    def remove(self):
        for _, h in self._handles:
            h.remove()
        self._handles = []


class HeadRecorder:
    """Records the o_proj input (all heads) at one position for every LM layer."""

    def __init__(self, layers, n_heads, head_dim):
        self.layers, self.n_heads, self.head_dim = layers, n_heads, head_dim
        self.pos = -1
        self.store = {}
        self._h = [mod.register_forward_pre_hook(self._mk(l)) for l, _, mod in layers]

    def _mk(self, l):
        def hook(module, args):
            x = args[0]
            self.store[l] = x[0, self.pos].detach().float().view(self.n_heads, self.head_dim).cpu()
        return hook

    def stacked(self):
        return torch.stack([self.store[l] for l, _, _ in self.layers], 0)  # [L,H,hd]

    def remove(self):
        for h in self._h:
            h.remove()


def load_stv_artifact(path):
    art = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(art, dict) or art.get("format") != STV_FORMAT:
        raise RuntimeError(f"[STV] {path} is not an STV steering artifact (format={getattr(art, 'get', lambda k: None)('format')})")
    return art


def attach_stv(model, art, logger=None):
    steerer = STVSteerer(model)
    if steerer.n_heads != art["n_heads"] or steerer.head_dim != art["head_dim"] or len(steerer.layers) != art["n_layers"]:
        raise RuntimeError(
            f"[STV] artifact geometry L={art['n_layers']} H={art['n_heads']} hd={art['head_dim']} does not match "
            f"model L={len(steerer.layers)} H={steerer.n_heads} hd={steerer.head_dim}")
    steerer.mode = "last"
    steerer.set_values(art["locations"], art["values"].float())
    if logger is not None:
        logger.info(f"[STV] steering {len(art['locations'])} heads over {len(steerer._handles)} layers "
                    f"(bank C={art['bank'].shape[1]}, trained on {art.get('config', {}).get('datalist')})")
    return steerer


def attach_stv_from_note(model, training_args, logger=None):
    """eval_VLM.py entry: load $CKPT_ROOT/client_states_<note>/server_model_round{r-1}.pth.

    The STV artifact is NOT a state_dict, so model.load_state_dict (and its 0-key guard,
    which stays in place for LoRA-type methods) is bypassed on purpose.
    """
    path = os.path.join(os.environ.get("CKPT_ROOT", "."), f"client_states_{training_args.note}",
                        f"server_model_round{training_args.round_to_eval - 1}.pth")
    if logger is not None:
        logger.info(f"[STV] load steering artifact {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[STV] steering artifact not found: {path}")
    return attach_stv(model, load_stv_artifact(path), logger)
