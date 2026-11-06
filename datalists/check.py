import json
import random
# INPUT_JSON = "/disk3/mjl/LMM-ICL/datalists/pope_pop/dataset-8.json"
# OUTPUT_JSON = "/disk3/mjl/LMM-ICL/datalists/pope_pop/dataset-8.json"

# BAD_PREFIX = "data/coco/coco/val2014/data/coco/val2014"
# GOOD_PREFIX = "data/coco/val2014"


# def fix_path(path):
#     if isinstance(path, str):
#         return path.replace(BAD_PREFIX, GOOD_PREFIX)
#     return path


# def process_sample(sample):
#     # Fix image list
#     if "image" in sample:
#         sample["image"] = [fix_path(p) for p in sample["image"]]

#     # Fix optional fields
#     if "primary_image" in sample:
#         sample["primary_image"] = fix_path(sample["primary_image"])

#     if "secondary_image" in sample:
#         sample["secondary_image"] = fix_path(sample["secondary_image"])

#     return sample


# with open(INPUT_JSON, "r") as f:
#     data = json.load(f)

# data = [process_sample(sample) for sample in data]

# with open(OUTPUT_JSON, "w") as f:
#     json.dump(data, f, indent=2)

# print(f"Saved cleaned JSON to {OUTPUT_JSON}")

with open("/disk3/mjl/LMM-ICL/datalists/lure/train/dataset-1.json", "r") as f:
    read_json = json.load(f)

sampled = random.sample(read_json, 1000)
with open("/disk3/mjl/LMM-ICL/datalists/lure/train/dataset-2.json", "w") as f:
    json.dump(sampled, f, indent=2)