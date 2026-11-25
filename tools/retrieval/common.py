"""Shared helpers for leak checks, clean-train building and retrieval baselines.

All paths are relative to the repo root /tmp/procore (scripts chdir there).
"""
import glob
import json
import os
import re

ROOT = "/tmp/procore"

# benchmark -> scenario dataset dir names whose test datalists define the benchmark
BENCH_SCEN_DIRS = {
    "isekai": ["isekai"],
    "mic": ["mic"],
    "coco": ["mic", "coco"],          # coco test files are the coco/ part of MIC
    "truemicl": ["truemicl"],
    "vliclv2": ["vliclv2"],
    "pororov4": ["pororov4"],
    "storystreamv2": ["storystreamv2"],
    "flintstonesv3": ["flintstonesv3"],
}


def norm_key(p):
    """Canonical image identity. COCO images are keyed by basename so that
    data/coco/train2014, data/coco/coco/train2014 and data/vqa/vqa/train2014
    copies of the same picture collide."""
    p = p.replace("\\", "/")
    b = os.path.basename(p)
    if b.startswith("COCO_"):
        return "COCO/" + b
    if "/gqa/images/" in p or "/VisualGenome/" in p or "/vg/" in p.lower():
        return "VG/" + b
    p = re.sub(r"^\./", "", p)
    p = re.sub(r"^data/", "", p)
    parts = p.split("/")
    if len(parts) > 1 and parts[0] == parts[1]:
        parts = parts[1:]
    return "/".join(parts)


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)


def test_datalist_paths(bench, include_orderings=True):
    """All datalists referenced by any eval scenario of a benchmark
    (scenarios/<k>-shot/<dir>/<ordering>/scenario-*.json)."""
    out = set()
    for d in BENCH_SCEN_DIRS[bench]:
        pat = f"{ROOT}/scenarios/*-shot/{d}/*/scenario-*.json"
        for sf in glob.glob(pat):
            ordering = sf.split("/")[-2]
            if not include_orderings and ordering != "random":
                continue
            if sf.endswith(".leaky.json"):
                continue
            if ordering.endswith("_v2") or ordering.endswith("_smoke"):
                continue   # v2 retrieval files: demos come from the clean train pool by design
            try:
                sc = load(sf)
            except Exception:
                continue
            for cl in sc:
                for ds in cl["datasets"]:
                    if bench == "coco" and ds["dataset"] != "coco":
                        continue
                    if bench == "mic" and ds["dataset"] == "coco" and False:
                        continue
                    p = f"{ROOT}/datalists/{ds['dataset']}/dataset-{ds['subset_id']}.json"
                    if os.path.exists(p):
                        out.add(p)
    return sorted(out)


def item_answer(it):
    return it["conversations"][-1]["value"].strip().lower()


def test_keys(bench, include_orderings=True):
    """Return (all image keys, query image keys, set of (query key, answer))."""
    allk, qk, qa = set(), set(), set()
    for p in test_datalist_paths(bench, include_orderings):
        for it in load(p):
            imgs = it.get("image") or []
            if isinstance(imgs, str):
                imgs = [imgs]
            for im in imgs:
                allk.add(norm_key(im))
            if imgs:
                k = norm_key(imgs[-1])
                qk.add(k)
                qa.add((k, item_answer(it)))
    return allk, qk, qa


def leak_stats(items, allk, qk, qa):
    n = len(items)
    q_in_test_q = q_in_test_any = qa_hit = demo_hit = any_hit = 0
    for it in items:
        imgs = it.get("image") or []
        if isinstance(imgs, str):
            imgs = [imgs]
        if not imgs:
            continue
        k = norm_key(imgs[-1])
        q_in_test_q += k in qk
        q_in_test_any += k in allk
        qa_hit += (k, item_answer(it)) in qa
        d = any(norm_key(x) in allk for x in imgs[:-1])
        demo_hit += d
        any_hit += d or (k in allk)
    return dict(n=n, query_in_test_queries=q_in_test_q, query_in_any_test_image=q_in_test_any,
                query_img_and_answer_in_test=qa_hit, demo_in_any_test_image=demo_hit,
                item_touches_test=any_hit)
