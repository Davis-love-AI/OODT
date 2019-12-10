
# -*- coding: utf-8 -*-

import torch
from torch import nn

from .build import PROPOSAL_GENERATOR_REGISTRY
from .ienet_module import build_avod_head, \
                          build_avod_loss_evaluator, \
                          build_avod_postprocessor

class IENet_RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, in_channels):
        super(IENet_RPN, self).__init__()


        self.head = build_avod_head(cfg, in_channels)
        self.box_selector_test = build_avod_postprocessor(cfg)
        self.loss_evaluator = build_avod_loss_evaluator(cfg)
        self.fpn_strides = cfg.MODEL.AVOD.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        prediction = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, prediction, targets
            )
        else:
            return self._forward_test(
                locations, prediction, images.image_sizes
            )

    def _forward_train(self, locations, prediction, targets):
        losses = self.loss_evaluator(
            locations, prediction, targets
        )
        return None, losses

    def _forward_test(self, locations, prediction, image_sizes):
        boxes = self.box_selector_test(
            locations, prediction, image_sizes
        )
        return boxes, {}

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




