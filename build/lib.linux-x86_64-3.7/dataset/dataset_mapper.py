# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
import random
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from .mapper_tool import (
        read_image,
        filter_empty_instances,
        dota_annotations_to_instances, 
        transform_dota_instance_annotations,
    )

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)


        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.rota_aug_on    = cfg.MODEL.AVOD.ROTA_AUG_ON
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
#        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        rota = 0
        
        if self.rota_aug_on and dataset_dict["split"] != "val_mini" and dataset_dict["split"] != "test":
            rotaed_aug = [0, 90, 180, 270]
            rota = random.sample(rotaed_aug, 1)[0]
            
        image = read_image(dataset_dict["file_name"], format=self.img_format, rota=rota)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
 

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day


        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_dota_instance_annotations(
                    obj, image_shape, rota, transforms
                )
                for obj in dataset_dict.pop("annotations")
            ]
            
            
            instances = dota_annotations_to_instances(
                annos, image_shape
            )
         
            dataset_dict["instances"] = filter_empty_instances(instances)


        return dataset_dict
