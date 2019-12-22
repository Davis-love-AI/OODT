from ienet import add_ienet_config
from dataset import (
        register_all_dota_voc,
        DatasetMapper, 
        DotaVOCDetectionEvaluator,
        load_dota_instances
    )
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    RotatedBoxes,
)
import copy
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import random
import cv2
import math
import numpy as np
from projects.Avod.test.vis_tool import (
        polygonToRotRectangle,
        hrbb_anchor2hw_obb,
        rotate,
        transform_dota_instance_annotations,
        dota_annotations_to_instances,
        filter_empty_instances,
        batch_polygonToRotRectangle,
        batch_hbb_hw2poly,
        get_best_begin_point,
        batch_get_best_begin_point,
        read_image
    )
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from projects.Avod.test.test_vis_tool import (
        Visualizer,
        _create_text_labels
    )
import cv2 as cv
from detectron2.utils.colormap import random_color
from PIL import Image
import tqdm

cfg = get_cfg()
add_ienet_config(cfg)
# file = 'projects/Avod/configs/ienet_R_50_FPN_s1x_test.yaml'
file = 'projects/Avod/configs/ienet_R_101_FPN_demo_s1x.yaml'
cfg.merge_from_file(file)
model_w = "/home/omnisky/Pengming_workspace/disk_2T/DOTA/training/ienet/avod_center_demo/model_0119999.pth"
cfg.MODEL.WEIGHTS = model_w

predictor = DefaultPredictor(cfg)
size= "DOTA_800"
root = cfg.MODEL.AVOD.DATASET
register_all_dota_voc(root)
split = "val_mini"
dir_root = os.path.join(root.format(size), split) 

dataset_dicts = load_dota_instances(dir_root, split)
dota_metadata = MetadataCatalog.get("dota_val_mini_800")

image_shape = np.array([800, 800])
i=0


trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)

evaluator = DotaVOCDetectionEvaluator("dota_val_mini_800", 'rot')
val_loader = build_detection_test_loader(cfg, "dota_val_mini_800",
                                         mapper=DatasetMapper(cfg, False))
inference_on_dataset(trainer.model, val_loader, evaluator)