"""Task-vector baselines v2: faithful MTV (Huang et al., NeurIPS'24) and FV (Todd et al., ICLR'24).

Used by taskvec_v2_extract.py (extraction / head selection) and by eval_VLM.py (METHOD=mtv_v2 / fv_v2).

Design (see also MTV/storage_v2/README.md):
  * Only ICL TRAINING data is used (datalists/<bench>/train/..., datalists/vliclv2/<task>/train/...).
    Every training item whose query image(s) appear as a query in ANY eval datalist of the same
    benchmark (all shots / scenarios, resolved from scenarios/*/<bench>/*/scenario-*.json) is dropped.
  * One vector per (arch, benchmark, sub-task); the sub-task = the eval scenario's dataset
    (mic: 10 sub-tasks, vliclv2: 4, truemicl: 7, isekai: 1).
  * Prompts are built with exactly the same code path the evaluator uses for each arch
    (QwenVL3_GenerationDataset / OneVision1_5_GenerationDataset / idefics chat template /
    InternVL model.chat), so the activations live in the eval prompt distribution.
  * MTV: mean last-token o_proj-input (per-head) activations over N few-shot (4-shot) ICL prompts;
    REINFORCE over a Bernoulli per head (lr 0.1, 600 steps, 32 samples, init logit -1, reward =
    standardized CE of the first answer token) on ZERO-SHOT training queries whose last-token
    heads are REPLACED by the means; best of 10 sampled head sets on a held-out training split.
  * FV: same mean activations; average indirect effect (AIE) of each head = P(gold first token |
    head patched with its mean) - P(gold | no patch), on ICL prompts whose demo LABELS ARE SHUFFLED;
    FV = sum over the top-K heads of o_proj(mean head output), added to the residual stream
    (decoder-layer output) at layer L//3.
  * Both interventions act on the last position of every forward call (prompt last token, then
    each decoding step), as in the original code bases.

Head patching at the last position is computed exactly with a KV cache: the prompt minus its last
token is prefilled once, then the last token is re-run in a batch where each row patches a
different head set (verified against a full forward, see check_cache_equivalence).
"""
import collections
import contextlib
import copy
import glob
import json
import os
import random
import re

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = "<image>"

# ----------------------------------------------------------------------------------------------
# Benchmarks, sub-tasks and training sources
# ----------------------------------------------------------------------------------------------
PICL = ("pororov4", "storystreamv2", "flintstonesv3")
# scenario id (as used by eval.sh) -> sub-task name; "pooled" (all sub-tasks of the benchmark) is
# extracted too and is what cross-benchmark evaluation (TASKVEC_SRC_BENCH) loads.
BENCH_TASKS = {
    "mic": {1: "coco", 2: "flickr", 3: "miniimage", 4: "gqa", 5: "nlvr2", 6: "iconqa",
            7: "vqa",  # okvqa: MIC's train split stores OK-VQA and VQAv2 under data/vqa (not separable)
            8: "stvqa", 9: "textvqa", 10: "vqa"},
    "vliclv2": {1: "clevr", 2: "textocr", 3: "operator_induction", 4: "operator_induction_interleaved"},
    "truemicl": {1: "classification", 2: "clevr", 3: "operator_induction", 4: "palindrome_dataset",
                 5: "shapes_count", 6: "sudoku", 8: "clock"},
    "isekai": {1: "isekai", 2: "isekai"},
    "coco": {1: "coco"},
    **{d: {1: d} for d in PICL},
}
POOLED = "pooled"

# Leak-free training sources (train files with test-query overlap removed; PICL train is disjoint by
# construction; vliclv2 uses its per-task train files + the canonical-key filter below).
#   zs: zero-shot queries; icl: 4-shot prompts; compose: build 4-shot prompts from zero-shot items
#   (formats without a per-demo answer delimiter, so labels can be shuffled for FV).
BENCH_SOURCES = {
    "mic": {"zs": "datalists/mic/train/dataset-0_split.json", "icl": None, "compose": True},
    "coco": {"zs": None, "icl": "datalists/coco/train/dataset-14.json", "compose": True},
    "vliclv2": {"zs": "datalists/vliclv2/{task}/train/dataset-0.json",
                "icl": "datalists/vliclv2/{task}/train/dataset-4.json", "compose": False},
    "truemicl": {"zs": "datalists/truemicl/train/dataset-0.json",
                 "icl": "datalists/truemicl/train/dataset-4.json", "compose": False},
    "isekai": {"zs": "datalists/isekai/train/dataset-0_split.json",
               "icl": "datalists/isekai/train/dataset-4_split.json", "compose": False},
    **{d: {"zs": None, "icl": f"datalists/{d}/train/dataset-4.json", "compose": False} for d in PICL},
}
ICL_SHOT = 4  # demos per ICL prompt used for the mean activations / AIE (MTV & FV papers use few-shot prompts)


def _p(path):
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def norm_parts(path):
    return os.path.normpath(path.lstrip("./") if path.startswith("./") else path).split("/")


def _common():
    import importlib.util
    spec = importlib.util.spec_from_file_location("retrieval_common", _p("tools/retrieval/common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def img_key(path):
    """Canonical image identity = tools/retrieval/common.norm_key (same keys as leak_check.py)."""
    return _common().norm_key(path)


_NORM = None


def _norm(path):
    global _NORM
    if _NORM is None:
        _NORM = _common().norm_key
    return _NORM(path)


def imgs_per_example(item):
    return 2 if "operator_induction_interleaved" in item["image"][-1] else 1


def query_key(item):
    k = imgs_per_example(item)
    return tuple(_norm(p) for p in item["image"][-k:])


def task_of(item, bench):
    if bench in ("isekai", "coco") or bench in PICL:
        return bench if bench != "coco" else "coco"
    parts = norm_parts(item["image"][-1])
    if bench == "vliclv2":
        return parts[2]  # data/vlicl/<task>/...
    return parts[1]      # data/<task>/...


_TEST_CACHE = {}


def eval_query_keys(bench):
    """Test query keys of `bench` from tools/retrieval/common.test_keys (all shots, scenarios and
    orderings except *_v2/_smoke/.leaky) + query image-pair keys (operator_induction_interleaved)."""
    if bench not in _TEST_CACHE:
        C = _common()
        _, qk, _ = C.test_keys(bench)
        files = C.test_datalist_paths(bench)
        pairs = set()
        for f in files:
            for y in C.load(f):
                if y.get("image") and imgs_per_example(y) == 2:
                    pairs.add(query_key(y))
        _TEST_CACHE[bench] = ({(k,) for k in qk} | pairs, files)
    return _TEST_CACHE[bench]


def _valid(item):
    if not item.get("image") or len(item.get("conversations", [])) < 2:
        return False
    k = imgs_per_example(item)
    n = item["conversations"][0]["value"].count(IMG)
    return n == len(item["image"]) and n % k == 0


def leaks(item, ev_keys):
    """True if the item's query image(s) are a test query, or (single-image tasks) a demo image is a test query."""
    if query_key(item) in ev_keys:
        return True
    if imgs_per_example(item) == 1:
        return any((_norm(p),) in ev_keys for p in item["image"][:-1])
    return False


def load_pools(bench, task, log=print):
    """Return (zs_pool, icl_pool_or_None, report). Pools contain only leak-filtered training items."""
    src = BENCH_SOURCES[bench]
    ev_keys, ev_files = eval_query_keys(bench)
    report = {"bench": bench, "task": task, "n_test_datalists": len(ev_files)}

    def load(path_tmpl, name):
        if path_tmpl is None:
            return None
        path = _p(path_tmpl.format(task=task))
        data = [y for y in json.load(open(path)) if _valid(y) and task_of(y, bench) == task]
        kept = [y for y in data if not leaks(y, ev_keys)]
        report[name] = {"file": os.path.relpath(path, REPO), "n_task": len(data), "n_kept": len(kept)}
        return kept

    zs = load(src["zs"], "zs")
    icl = load(src["icl"], "icl")
    log(f"[taskvec_v2] pools {bench}/{task}: {json.dumps(report)}")
    return zs, icl, report


# ----------------------------------------------------------------------------------------------
# Prompt manipulation (training items are in the same llava-style format as the eval datalists)
# ----------------------------------------------------------------------------------------------
def split_icl(item):
    """-> (prefix, parts, k) with parts[j] = text after the (j+1)-th image; example i ends at parts[(i+1)*k-1]."""
    parts = item["conversations"][0]["value"].split(IMG)
    return parts[0], parts[1:], imgs_per_example(item)


def join_icl(prefix, parts):
    return prefix + "".join(IMG + p for p in parts)


def to_zero_shot(item):
    """Drop the demos of a k-shot item: keep the instruction prefix and the query example."""
    prefix, parts, k = split_icl(item)
    new = copy.deepcopy(item)
    new["conversations"][0]["value"] = join_icl(prefix, parts[-k:])
    new["image"] = list(item["image"][-k:])
    return new


def _split_answer(seg):
    idx = seg.rfind("Answer:")
    if idx < 0:
        return None
    ans = seg[idx + len("Answer:"):]
    core = ans.strip()
    lead = ans[: len(ans) - len(ans.lstrip())]
    trail = ans[len(ans.rstrip()):]
    return seg[: idx + len("Answer:")], lead or " ", core, trail


def _derangement(n, rng):
    if n < 2:
        return None
    for _ in range(100):
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
    return None


def shuffle_labels(item, rng):
    """Todd et al.: shuffle the demo labels so the prompt has no consistent input->label mapping.

    Demo answers are the text after the last 'Answer:' of each demo. Returns None if the item
    cannot be shuffled (no delimiter, <2 demos, or all labels identical).
    """
    prefix, parts, k = split_icl(item)
    n_demo = len(parts) // k - 1
    idxs = [(i + 1) * k - 1 for i in range(n_demo)]
    pieces = [_split_answer(parts[j]) for j in idxs]
    if n_demo < 2 or any(p is None for p in pieces):
        return None
    labels = [p[2] for p in pieces]
    if len(set(labels)) < 2:
        return None
    for _ in range(20):
        perm = _derangement(n_demo, rng)
        if perm is None:
            return None
        new_labels = [labels[perm[i]] for i in range(n_demo)]
        if new_labels != labels:
            break
    else:
        return None
    new_parts = list(parts)
    for j, p, lab in zip(idxs, pieces, new_labels):
        new_parts[j] = p[0] + p[1] + lab + p[3]
    new = copy.deepcopy(item)
    new["conversations"][0]["value"] = join_icl(prefix, new_parts)
    return new


# --- MIC: k-shot prompts are composed from zero-shot items (template + " " + answer per demo) ---
def mic_clean(text):
    text = re.sub(r"(image\d+):\s*image\d+图", r"\1", text)
    return re.sub(r"(image\d+)图", r"\1", text)


def mic_compose(demos, query, labels=None):
    """MIC ICL format: demos are numbered image{8-k}..image7, the query is image8, examples joined by blank lines."""
    k = len(demos)
    labels = labels if labels is not None else [str(d["conversations"][1]["value"]).strip() for d in demos]
    texts = []
    for j, (d, lab) in enumerate(zip(demos, labels)):
        t = mic_clean(d["conversations"][0]["value"]).strip()
        t = re.sub(r"image8(?!\d)", f"image{8 - k + j}", t)
        texts.append(t + " " + lab)
    texts.append(mic_clean(query["conversations"][0]["value"]).strip())
    new = copy.deepcopy(query)
    new["conversations"][0]["value"] = "\n\n".join(texts)
    new["image"] = [d["image"][-1] for d in demos] + [query["image"][-1]]
    return new


def mic_zero_shot(item):
    new = copy.deepcopy(item)
    new["conversations"][0]["value"] = mic_clean(item["conversations"][0]["value"])
    return new


class TaskData:
    """Sampling interface over the leak-filtered pools of one (bench, task)."""

    def __init__(self, bench, task, seed=0, n_val=50, log=print):
        self.bench, self.task = bench, task
        self.rng = random.Random(seed)
        zs, icl, self.report = load_pools(bench, task, log=log)
        compose = BENCH_SOURCES[bench]["compose"]
        if not zs and icl:
            zs = [to_zero_shot(y) for y in icl]
        if compose:  # MIC-format benchmarks (mic, coco)
            zs = [mic_zero_shot(y) for y in zs]
            icl = None
        zs = list(zs)
        self.rng.shuffle(zs)
        nv = min(n_val, max(1, len(zs) // 5))
        self.val_zs, self.train_zs = zs[:nv], zs[nv:]
        self.icl = None
        if icl is not None:
            # val_zs (MTV best-of-10 selection) is disjoint from the REINFORCE queries (train_zs); like the
            # official MTV code, the ICL prompts for the mean activations may reuse any training query.
            self.icl = list(icl)
            self.rng.shuffle(self.icl)
        self.report.update({"n_val_zs": len(self.val_zs), "n_train_zs": len(self.train_zs),
                            "n_icl": len(self.icl) if self.icl is not None else None})

    def icl_prompts(self, n, shuffled=False):
        out, tries = [], 0
        while len(out) < n and tries < 20 * n:
            tries += 1
            if self.icl is not None:
                item = self.icl[(tries - 1) % len(self.icl)]
                if shuffled:
                    item = shuffle_labels(item, self.rng)
            else:
                if len(self.train_zs) < ICL_SHOT + 1:
                    raise RuntimeError(f"pool too small for {self.bench}/{self.task}")
                picks = self.rng.sample(self.train_zs, ICL_SHOT + 1)
                demos, query = picks[:-1], picks[-1]
                labels = None
                if shuffled:
                    labs = [str(d["conversations"][1]["value"]).strip() for d in demos]
                    perm = _derangement(len(labs), self.rng)
                    if perm is None or len(set(labs)) < 2:
                        continue
                    labels = [labs[perm[i]] for i in range(len(labs))]
                item = mic_compose(demos, query, labels)
            if item is not None:
                out.append(item)
        return out

    def zero_shot(self, n):
        return self.rng.sample(self.train_zs, min(n, len(self.train_zs)))


class PooledTaskData:
    """All sub-tasks of a benchmark pooled (used for cross-benchmark evaluation). Each prompt is drawn
    from one sub-task chosen uniformly at random (demos and query always share a sub-task)."""

    def __init__(self, bench, seed=0, n_val=50, log=print):
        self.bench, self.task = bench, POOLED
        self.rng = random.Random(seed)
        tasks = sorted(set(BENCH_TASKS[bench].values()))
        per_val = max(1, n_val // len(tasks))
        self.children = [TaskData(bench, t, seed=seed, n_val=per_val, log=log) for t in tasks]
        self.val_zs = [y for c in self.children for y in c.val_zs]
        self.train_zs = [y for c in self.children for y in c.train_zs]
        self.report = {"bench": bench, "task": POOLED, "subtasks": {c.task: c.report for c in self.children},
                       "n_val_zs": len(self.val_zs), "n_train_zs": len(self.train_zs)}

    def icl_prompts(self, n, shuffled=False):
        out = []
        for _ in range(n):
            out += self.rng.choice(self.children).icl_prompts(1, shuffled=shuffled)
        return out

    def zero_shot(self, n):
        return self.rng.sample(self.train_zs, min(n, len(self.train_zs)))


def make_task_data(bench, task, **kw):
    return PooledTaskData(bench, **kw) if task == POOLED else TaskData(bench, task, **kw)


# ----------------------------------------------------------------------------------------------
# Arch I/O: build inputs exactly like eval_VLM.py, run full / prefix / last-token-step forwards
# ----------------------------------------------------------------------------------------------
def arch_family(name_or_path):
    n = name_or_path.lower()
    if "qwen3" in n:
        return "qwen3"
    if "internvl3_5" in n or "internvl3.5" in n:
        return "internvl"
    if "onevision-1.5" in n or "onevision1_5" in n:
        return "ov15"
    if "idefics" in n:
        return "idefics"
    raise ValueError(f"taskvec_v2: unsupported model {name_or_path}")


class ArchIO:
    def __init__(self, model, tokenizer, processor, data_args, model_name_or_path, device="cuda"):
        import eval_VLM as EV  # reuse the evaluator's prompt builders
        self.EV = EV
        self.model, self.tokenizer, self.processor, self.data_args = model, tokenizer, processor, data_args
        self.family = arch_family(model_name_or_path)
        self.device = device
        self.use_cache = True

    # ---------- inputs ----------
    def build(self, item):
        f = self.family
        gold = str(item["conversations"][1]["value"])
        if f == "qwen3":
            from utils.data_loader_VLM import QwenVL3_GenerationDataset
            d = QwenVL3_GenerationDataset([item], self.processor, self.tokenizer, self.data_args)[0]
            ids = d["input_ids"].reshape(1, -1)
            st = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=d["pixel_values"].to(torch.bfloat16),
                      image_grid_thw=d["image_grid_thw"])
        elif f == "ov15":
            from utils.data_loader_VLM import OneVision1_5_GenerationDataset
            d = OneVision1_5_GenerationDataset([item], self.tokenizer, self.processor, self.data_args)[0]
            ids = d["input_ids"].reshape(1, -1)
            st = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=d["pixel_values"].to(torch.bfloat16),
                      image_grid_thw=d["image_grid_thw"])
        elif f == "idefics":
            from PIL import Image
            pairs, double_image = self.EV.parse_image_qa_pairs(item["conversations"][0]["value"])
            prompts = self.EV.build_idefics_prompts(pairs, item["image"], double_image)
            images = [Image.open(p).convert("RGB") for p in item["image"]]
            prompt = self.processor.apply_chat_template(prompts, add_generation_prompt=True)
            st = dict(self.processor(text=prompt, images=images, return_tensors="pt"))
        elif f == "internvl":
            st = self._internvl_inputs(item)
        st = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in st.items()}
        return st, gold

    def _internvl_inputs(self, item):
        """Capture exactly what model.chat (the eval path) feeds to generate, then embed it."""
        model = self.model
        pixel_values, num_patches_list, _, _ = self.EV.load_intern_inputs(item, self.device)
        captured = {}
        eos = self.tokenizer.eos_token_id or 0

        def fake_generate(pixel_values=None, input_ids=None, attention_mask=None, **kw):
            captured.update(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            return torch.tensor([[eos]], device=input_ids.device), None

        model.generate = fake_generate
        try:
            model.chat(self.tokenizer, pixel_values, item["conversations"][0]["value"],
                       dict(max_new_tokens=1, do_sample=False), num_patches_list=num_patches_list,
                       history=None, return_history=True)
        finally:
            del model.generate
        ids = captured["input_ids"]
        with torch.no_grad():
            vit = model.extract_feature(captured["pixel_values"])
            emb = model.language_model.get_input_embeddings()(ids).clone()
            B, N, C = emb.shape
            emb = emb.reshape(B * N, C)
            sel = ids.reshape(B * N) == model.img_context_token_id
            emb[sel] = vit.reshape(-1, C).to(emb.dtype)
            emb = emb.reshape(B, N, C)
        return dict(inputs_embeds=emb, attention_mask=captured["attention_mask"], input_ids=ids)

    def target_id(self, gold):
        text = (" " + gold.strip()) if self.family == "idefics" else gold.strip()
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids[0]

    def seq_len(self, st):
        return st["input_ids"].shape[1]

    # ---------- forwards ----------
    def full_logits(self, st):
        f = self.family
        with torch.no_grad():
            if f == "internvl":
                out = self.model.language_model(inputs_embeds=st["inputs_embeds"], attention_mask=st["attention_mask"])
            else:
                out = self.model(**st)
        return out.logits[:, -1, :].float()

    def prefix(self, st):
        f = self.family
        T = self.seq_len(st)
        with torch.no_grad():
            if f == "internvl":
                out = self.model.language_model(inputs_embeds=st["inputs_embeds"][:, :-1],
                                                attention_mask=st["attention_mask"][:, :-1], use_cache=True)
            else:
                kw = dict(st)
                kw["input_ids"] = st["input_ids"][:, :-1]
                kw["attention_mask"] = st["attention_mask"][:, :-1]
                if f == "ov15":
                    kw["cache_position"] = torch.arange(T - 1, device=self.device)
                out = self.model(**kw, use_cache=True)
        return out.past_key_values, T

    def step(self, st, cache, T, B):
        """Run the last prompt token for a batch of B rows on top of a (B-row) prefix cache."""
        f = self.family
        cp = torch.tensor([T - 1], device=self.device)
        am = torch.ones(B, T, dtype=torch.long, device=self.device)
        with torch.no_grad():
            if f == "internvl":
                e = st["inputs_embeds"][:, -1:].expand(B, -1, -1).contiguous()
                out = self.model.language_model(inputs_embeds=e, attention_mask=am, past_key_values=cache,
                                                cache_position=cp, position_ids=cp.view(1, 1).expand(B, 1),
                                                use_cache=True)
            else:
                ids = st["input_ids"][:, -1:].expand(B, 1).contiguous()
                kw = dict(input_ids=ids, attention_mask=am, past_key_values=cache, cache_position=cp, use_cache=True)
                if f == "ov15":
                    kw["position_ids"] = cp.view(1, 1).expand(B, 1)
                out = self.model(**kw)
        cache.crop(T - 1)
        return out.logits[:, -1, :].float()


def cache_bytes(cache):
    tot = 0
    for layer in getattr(cache, "layers", []):
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if torch.is_tensor(t):
                tot += t.numel() * t.element_size()
    if tot == 0 and hasattr(cache, "key_cache"):
        for t in list(cache.key_cache) + list(cache.value_cache):
            tot += t.numel() * t.element_size()
    return max(tot, 1)


# ----------------------------------------------------------------------------------------------
# Hooks
# ----------------------------------------------------------------------------------------------
def _layer_idx(name):
    for p in name.split("."):
        if p.isdigit():
            return int(p)
    raise ValueError(name)


def get_module(model, name):
    m = model
    for p in name.split("."):
        m = getattr(m, p)
    return m


class HeadPatcher:
    """forward_pre_hooks on every o_proj: capture the last-position per-head input and/or replace it.

    mask: bool tensor (L, B, H) - row b of the current batch gets heads mask[l, b] replaced by values[l].
    """

    def __init__(self, model, attn_hook_names, n_heads):
        self.n_heads = n_heads
        self.names = list(attn_hook_names)
        self.mods = [get_module(model, n) for n in self.names]
        self.capture = False
        self.captured = [None] * len(self.mods)
        self.mask = None
        self.values = None
        self.handles = [m.register_forward_pre_hook(self._make(l)) for l, m in enumerate(self.mods)]

    def _make(self, l):
        def pre(module, args):
            x = args[0]
            if self.capture:
                self.captured[l] = x[:, -1].detach().float()
            if self.mask is None:
                return None
            m = self.mask[l]
            if not bool(m.any()):
                return None
            B, T, D = x.shape
            H = self.n_heads
            hd = D // H
            if m.shape[0] != B:
                raise RuntimeError(f"HeadPatcher batch {m.shape[0]} != input batch {B}")
            x = x.clone()
            xh = x.view(B, T, H, hd)
            vals = self.values[l].to(x.device, x.dtype).view(1, H, hd)
            xh[:, -1] = torch.where(m.to(x.device)[:, :, None], vals, xh[:, -1])
            return (x,) + tuple(args[1:])
        return pre

    def remove(self):
        for h in self.handles:
            h.remove()


def batched_patch_logits(io, patcher, st, masks, budget_gb=24.0):
    """Last-position logits for each row of masks (L, R, H) (row r patches heads masks[:, r])."""
    R = masks.shape[1]
    if not io.use_cache:
        outs = []
        for r in range(R):
            patcher.mask = masks[:, r:r + 1]
            outs.append(io.full_logits(st))
        patcher.mask = None
        return torch.cat(outs, 0)
    patcher.mask = None
    cache, T = io.prefix(st)
    per = cache_bytes(cache)
    bs = int(max(1, min(R, (budget_gb * 2 ** 30) // per)))
    if bs > 1:
        cache.batch_repeat_interleave(bs)
    outs = []
    for s in range(0, R, bs):
        m = masks[:, s:s + bs]
        n = m.shape[1]
        if n < bs:
            m = torch.cat([m, torch.zeros(m.shape[0], bs - n, m.shape[2], dtype=m.dtype, device=m.device)], 1)
        patcher.mask = m
        outs.append(io.step(st, cache, T, bs)[:n])
    patcher.mask = None
    del cache
    return torch.cat(outs, 0)


def check_cache_equivalence(io, patcher, st, n_layers, n_heads):
    """Compare (prefix + last-token step) against a full forward, with and without a head patch."""
    full = io.full_logits(st)
    saved = patcher.values
    if saved is None:  # probe patch value: the clean last-token activations
        patcher.capture = True
        io.full_logits(st)
        patcher.capture = False
        patcher.values = torch.stack([c[0] for c in patcher.captured]).view(n_layers, n_heads, -1) * 0.5
    masks = torch.zeros(n_layers, 2, n_heads, dtype=torch.bool, device=io.device)
    masks[n_layers // 2, 1, :4] = True
    stepped = batched_patch_logits(io, patcher, st, masks)
    patcher.mask = masks[:, 1:2]
    full_p = io.full_logits(st)
    patcher.mask = None
    patcher.values = saved
    def diffs(a, b):
        pa, pb = torch.softmax(a, -1), torch.softmax(b, -1)
        top = a.topk(20).indices
        return (pa - pb).abs().max().item(), (torch.log_softmax(a, -1)[top] - torch.log_softmax(b, -1)[top]).abs().max().item()
    d0, t0 = diffs(full[0], stepped[0])
    d1, t1 = diffs(full_p[0], stepped[1])
    agree = int(full[0].argmax() == stepped[0].argmax()) + int(full_p[0].argmax() == stepped[1].argmax())
    patch_effect = (torch.softmax(full[0], -1) - torch.softmax(full_p[0], -1)).abs().max().item()
    return {"max_abs_prob_diff_clean": d0, "max_abs_prob_diff_patched": d1, "top20_logprob_diff_clean": t0,
            "top20_logprob_diff_patched": t1, "argmax_agree_of_2": agree, "probe_patch_effect": patch_effect}


@contextlib.contextmanager
def residual_add(model, layer_name, vector):
    """Add `vector` to the decoder-layer output (residual stream) at the last position of every forward."""
    mod = get_module(model, layer_name)

    def post(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1] = h[:, -1] + vector.to(h.device, h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = mod.register_forward_hook(post)
    try:
        yield
    finally:
        handle.remove()


def decoder_layer_name(attn_hook_name):
    return attn_hook_name.rsplit(".self_attn.o_proj", 1)[0]


# ----------------------------------------------------------------------------------------------
# Eval-time loaders (used by eval_VLM.py)
# ----------------------------------------------------------------------------------------------
def load_fv_v2(model, model_name_or_path, fv_path, device, get_mtv_config):
    blob = torch.load(fv_path, map_location="cpu")
    cfg = get_mtv_config(model, model_name_or_path)
    names = cfg["attn_hook_names"]
    if len(names) != blob["n_layers"]:
        raise RuntimeError(f"[FV v2] {fv_path}: n_layers {blob['n_layers']} != model {len(names)}")
    layer = int(blob["edit_layer"])
    return {"v2": True, "vector": blob["vector"].to(device), "layer_name": decoder_layer_name(names[layer]),
            "edit_layer": layer, "locations": [tuple(h[:2]) for h in blob["heads"]], "config": cfg}


def fv_v2_context(model, fv_data):
    return residual_add(model, fv_data["layer_name"], fv_data["vector"])
