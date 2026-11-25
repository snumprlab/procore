import json
import re
import random
from pathlib import Path
from copy import deepcopy
from collections import Counter


    


INPUT_FILES = [
    "/ceph/hpc/data/b2025b08-038-users/LMM-ICL/datalists/operator_induction/dataset-16_truemicl.json",
    "/ceph/hpc/data/b2025b08-038-users/LMM-ICL/datalists/clevr/dataset-16_truemicl.json",
    "/ceph/hpc/data/b2025b08-038-users/LMM-ICL/datalists/classification/dataset-16_truemicl.json",
    "/ceph/hpc/data/b2025b08-038-users/LMM-ICL/datalists/palindrome_dataset/dataset-16.json",
    "/ceph/hpc/data/b2025b08-038-users/LMM-ICL/datalists/shapes_count/dataset-16.json",
    "/ceph/hpc/data/b2025b08-038-users/LMM-ICL/datalists/sudoku/dataset-16.json",
    "/ceph/hpc/data/b2025b08-038-users/LMM-ICL/datalists/clock/dataset-16.json",
    
    # "/path/to/file2.json",
]

INPUT_DIR = None   # e.g. "/path/to/json_folder"
OUTPUT_DIR = "/ceph/hpc/data/b2025b08-038-users/LMM-ICL/datalists/truemicl/train"

SHOTS = [0, 1, 2, 3,4, 8]
SAMPLES_PER_FILE = 115
SEED = 42
CANDIDATES_PER_OUTPUT = 300

random.seed(SEED)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)



# =========================================================
# Helpers
# =========================================================

# =========================================================
# Helpers
# =========================================================
def get_input_paths():
    paths = []
    if INPUT_DIR is not None:
        paths.extend(sorted(Path(INPUT_DIR).glob("*.json")))
    paths.extend(Path(p) for p in INPUT_FILES)
    return paths

def extract_answer_from_demo_block(block):
    """
    Example block:
    '\nWhat is the result of the following mathematical expression?\nAnswer: 8\n'

    Returns:
    '8'
    """
    m = re.search(r"Answer:\s*(.*?)\s*$", block.strip(), flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not extract answer from block:\n{block}")
    return m.group(1).strip()

def split_blocks(text):
    parts = text.split("<image>")
    return parts[0], parts[1:]


def parse_item(item):
    if "image" not in item or "conversations" not in item:
        return None

    images = item["image"]
    human = item["conversations"][0]["value"]

    prefix, blocks = split_blocks(human)

    if len(images) != len(blocks):
        return None

    return {
        "prefix": prefix,
        "demo_blocks": blocks[:-1],  # ignore original query block
        "support_images": images[:-1],
        "original": item,
    }

import re

def extract_instruction_from_demo_block(block):
    """
    Extract instruction text from a demo block.

    Example block:

    "\nWhat is the result of the following mathematical expression?\nAnswer: 8\n"

    Returns:

    "What is the result of the following mathematical expression"
    """

    text = block.strip()

    # remove Answer: value
    text = re.sub(r"Answer:\s*.*$", "", text).strip()

    # remove trailing ?
    text = text.rstrip("?").strip()

    return text

def rebuild(prefix, demo_blocks, instruction):
    text = prefix

    for block in demo_blocks:
        text += "<image>" + block

    text += f"<image>\n{instruction}?\nAnswer:"
    return text

def nested_subsets(indices8):
    shot8 = sorted(indices8)
    shot4 = sorted(random.sample(shot8, 4))
    shot3 = sorted(random.sample(shot4, 3))
    shot2 = sorted(random.sample(shot3, 2))
    shot1 = sorted(random.sample(shot2, 1))
    shot0 = []

    return {
        8: shot8,
        4: shot4,
        3: shot3,
        2: shot2,
        1: shot1,
        0: shot0,
    }


# =========================================================
# Sampling base structures
# =========================================================
def sample_base(parsed_items, target):

    usage = Counter()
    used = set()
    outputs = []

    valid = [p for p in parsed_items if len(p["support_images"]) >= 9]

    if not valid:
        return []

    fail = 0
    max_fail = target * 20

    while len(outputs) < target and fail < max_fail:

        best = None
        best_score = None

        for _ in range(CANDIDATES_PER_OUTPUT):

            parsed = random.choice(valid)

            pool = parsed["support_images"]

            indices9 = sorted(random.sample(range(len(pool)), 9))

            sig = tuple(pool[i] for i in indices9)

            if sig in used:
                continue

            imgs = [pool[i] for i in indices9]

            score = sum(usage[x] for x in imgs)

            score_tuple = (score, random.random())

            if best is None or score_tuple < best_score:
                best = (parsed, indices9)
                best_score = score_tuple

        if best is None:
            fail += 1
            continue

        parsed, indices9 = best
        used.add(tuple(parsed["support_images"][i] for i in indices9))

        for i in indices9:
            usage[parsed["support_images"][i]] += 1

        outputs.append((parsed, indices9))
        fail = 0

    return outputs
def extract_instruction_from_demo_block(block):
    text = block.strip()
    text = re.sub(r"Answer:\s*.*$", "", text).strip()
    text = text.rstrip("?").strip()
    return text
# =========================================================
# Build item
# =========================================================
def build_item(parsed, indices9, subset_indices, new_id):
    original = parsed["original"]
    pool = parsed["support_images"]
    blocks = parsed["demo_blocks"]

    # last one in subset_indices is the newly chosen query
    query_idx = subset_indices[-1]

    # demos are everything before the last one
    demo_indices = subset_indices[:-1]

    selected_supports = [pool[i] for i in demo_indices]
    selected_blocks = [blocks[i] for i in demo_indices]

    query_image = pool[query_idx]

    instruction = extract_instruction_from_demo_block(blocks[query_idx])
    query_answer = extract_answer_from_demo_block(blocks[query_idx])

    item = deepcopy(original)

    item["id"] = new_id
    item["image"] = selected_supports + [query_image]

    item["conversations"][0]["value"] = rebuild(
        parsed["prefix"],
        selected_blocks,
        instruction,
    )

    # update target answer to match the new sampled query
    item["conversations"][1]["value"] = query_answer

    return item


# =========================================================
# Main
# =========================================================
def main():

    paths = get_input_paths()

    combined = {0: [], 1: [], 2: [],3:[], 4: [], 8: []}

    for path in paths:

        print(f"Processing {path}")

        data = json.load(open(path))

        parsed = [parse_item(x) for x in data]
        parsed = [p for p in parsed if p]

        base_samples = sample_base(parsed, SAMPLES_PER_FILE)

        print(f"Generated {len(base_samples)} base samples")

        for parsed_item, indices9 in base_samples:

            subsets = nested_subsets(indices9[:8])

            query_idx = indices9[8]

            for shot in [0, 1, 2, 3,4, 8]:

                if shot == 0:
                    subset = [query_idx]
                else:
                    subset = subsets[shot] + [query_idx]

                item = build_item(
                    parsed_item,
                    indices9,
                    subset,
                    new_id=len(combined[shot]),
                )

                combined[shot].append(item)

    for shot in [0, 1, 2, 3,4, 8]:

        out = Path(OUTPUT_DIR) / f"dataset-{shot}.json"
        
        with open(f"/ceph/hpc/data/b2025b08-038-users/LMM-ICL/datalists/truemicl/train/old/dataset-{shot}.json", "r") as r_json:
            read_json = json.load(r_json)
        combined[shot].extend(read_json)
        
        real_datalist = combined[shot][:1000]
        random.shuffle(real_datalist)
        
        json.dump(real_datalist, open(out, "w"), indent=2)

        print(f"Saved {out} ({len(real_datalist)} samples)")


if __name__ == "__main__":
    main()
        
        