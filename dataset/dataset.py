# -*- coding: utf-8 -*-


from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
import itertools
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from .dataset_tool import preprocess_annotation
#__all__ = ["register_pascal_voc"]


# fmt: off
CLASS_NAMES = [
         'plane', 
         'baseball-diamond', 
         'bridge', 
         'ground-track-field', 
         'small-vehicle', 
         'large-vehicle', 
         'ship', 
         'tennis-court', 
         'basketball-court', 
         'storage-tank',  
         'soccer-ball-field', 
         'roundabout', 
         'harbor', 
         'swimming-pool', 
         'helicopter', 
]
# fmt: on


def load_dota_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "labelxml_voc", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "images", fileid + ".png")

        tree = ET.parse(anno_file)
        anno = preprocess_annotation(tree)
        anno["im_info"]
        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "split": split,
            "height": anno["im_info"][0],
            "width": anno["im_info"][1],
        }
        instances = []

        for ind in range(len(anno["boxes"])):
            cls = anno["labels"][ind]
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            poly = anno["boxes"][ind]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)

            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "boxes": poly}
            )
        
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_dota_voc(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_dota_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, split=split
    )
    
def register_all_dota_voc(root="datasets"):
    SPLITS = [
#        ("dota_train_1024", "train"),
#        ("dota_val_1024", "val"),
#        ("dota_test_1024", "test"),
        ("dota_test_800", "DOTA_800_0_5_500", "test"),
        ("dota_test", "./", "test"),
        ("dota_test_800_200", "DOTA_800_200", "test"),
        # ("dota_val_800_aug", "DOTA_800_0_5_500", "val"),
        ("dota_val_mini_800", "DOTA_800_0_5_500", "val_mini"),
        # ("dota_train_800_aug", "DOTA_800_0_5_500", "train"),
        ("dota_train_800", "DOTA_800_200", "train"),
        ("dota_val_800", "DOTA_800_200", "val"),
    ]
    for name, size, split in SPLITS:
        register_dota_voc(name, os.path.join(root.format(size), split), split)
        MetadataCatalog.get(name).evaluator_type = "hw"
