import json
import os

before_scen = 2
after_scen = f"{before_scen}_0-shot"
json_file = f"/disk1/mjlee/LMM-ICL/datalists/storystream/dataset-{before_scen}.json"

with open(json_file, "r") as r_json:
    read_json = json.load(r_json)
    
datalist = []
for data in read_json:
    new_images = []
    new_images.append(data["image"][-1])
    org_text = data["conversations"][0]["value"]
    new_text = org_text.split("\n\n")[-1]
    new_text = "<image>\n"+new_text
    new_text = new_text.replace("Learn from the previous images and captions.", "")
    data["conversations"][0]["value"] = new_text
    data["image"] = new_images
    datalist.append(data)

with open(f"/disk1/mjlee/LMM-ICL/datalists/storystream/dataset-{after_scen}.json", "w") as w_json:
    json.dump(datalist, w_json)