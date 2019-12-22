# -*- coding: utf-8 -*-
import logging
import torch
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.structures import Instances
from detectron2.modeling import build_backbone

# from detectron2.modeling import detector_postprocess
import numpy as np

from .ienet_module import (
        build_avod_head,
        build_avod_loss_evaluator, 
        build_avod_postprocessor,
    )

from detectron2.structures import (
    BoxMode,
)

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

__all__ = ["IENet"]



def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())


    output_boxes = results.pred_boxes

    output_boxes.scale(scale_x, scale_y)

    results = results[output_boxes.nonempty()]



    return results

@META_ARCH_REGISTRY.register()
class IENet(nn.Module):


    def __init__(self, cfg):
        super().__init__()
        
        in_channels = cfg.MODEL.AVOD.IN_CHANNELS
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.head = build_avod_head(cfg, in_channels)
        self.box_selector_test = build_avod_postprocessor(cfg)
        self.loss_evaluator = build_avod_loss_evaluator(cfg)
        self.fpn_strides = cfg.MODEL.AVOD.FPN_STRIDES
        self.in_features = cfg.MODEL.AVOD.IN_FEATURES

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)

#        backbone_shape = self.backbone.output_shape()
#        feature_shapes = [backbone_shape[f] for f in self.in_features]

        

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = self.get_ground_truth(batched_inputs)
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        
        prediction = self.head(features)
        locations = self.compute_locations(features)
        
        
        if self.training:
            return self._forward_train(locations, prediction, gt_instances)
        else:
            results = self._forward_test(locations, prediction, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
    
    def _forward_train(self, locations, prediction, targets):
        losses = self.loss_evaluator(
            locations, prediction, targets
        )
        return losses
    @torch.no_grad()
    def get_ground_truth(self, targets):
        target_dicts = []
        for target in targets:
            anno = target["instances"]
            target_dict = {}
            pt_inbox = anno.gt_pt_inbox_boxes.to(self.device)
            target_dict['pt_inbox'] = pt_inbox.tensor
            target_dict['pt_hbb'] = anno.gt_pt_hbb_boxes.tensor.to(self.device)
            target_dict['poly'] = anno.gt_poly.to(self.device)
            target_dict['area'] = anno.gt_areas.to(self.device)
            target_dict['labels'] = anno.gt_classes.to(self.device)
            target_dicts.append(target_dict)
            
                

        return target_dicts
    def _forward_test(self, locations, prediction, image_sizes):
        boxes = self.box_selector_test(
            locations, prediction, image_sizes
        )
        return boxes  
    
    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
