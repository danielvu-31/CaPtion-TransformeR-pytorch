import json
import os
import argparse


def split_json_files(img_root, ann_root, out_dir):
    final_data = {
        "train": [],
        "val": [],
        "test": [],
        "restval": []
    }

    with open(ann_root, "r") as f:
        karpathy_split = json.load(f)
    
    max = 0
    for img in karpathy_split["images"]:
        info_dict = {}
        info_dict["imgid"] = img["imgid"]
        info_dict["filepath"] = os.path.join(img_root, img["filepath"], img["filename"])
        info_dict["sentences"] = img["sentences"]
        info_dict["captions_number"] = len(img["sentences"])
        final_data[img["split"]].append(info_dict)

    with open(os.path.join(out_dir, "final_dataset.json"), "w") as outfile:
        json.dump(final_data, outfile)
    
    print("Reorganize Successfully")

if __name__ == "__main__":
    img_root = "/Users/mac/Projects/CaPtion-TransformeR-pytorch/data/image"
    ann_root = "/Users/mac/Projects/CaPtion-TransformeR-pytorch/data/annotation/dataset_coco.json"
    out_dir = "/Users/mac/Projects/CaPtion-TransformeR-pytorch/data/annotation/"
    split_json_files(img_root, ann_root, out_dir)