# -*- coding: utf-8 -*-


import math
import torch
import torch.nn.functional as F
from torch import nn

from ienet.layers import Scale, Self_Attn
from detectron2.layers import DeformConv
from detectron2.layers import Conv2d
from ienet.ienet_module.build import IENET_HEAD_REGISTRY


def tower_build(tower, conv_func, in_channels, out_channels, 
                kernel_size, stride, padding, bias):

    tower.append(
        conv_func(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
    )
    tower.append(nn.GroupNorm(32, out_channels))
    tower.append(nn.ReLU())
    return tower


@IENET_HEAD_REGISTRY.register()
class AVODWH_CENTER_HEAD(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(AVODWH_CENTER_HEAD, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.AVOD.NUM_CLASSES 
        self.fpn_strides = cfg.MODEL.AVOD.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.AVOD.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.AVOD.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.AVOD.USE_DCN_IN_TOWER
        self.pt_tower_define = cfg.MODEL.AVOD.INDEPENDENT_BRANCH

        cls_tower = []
        bbox_tower = []
        if self.pt_tower_define:
            pt_tower = []
        else:
            pt_tower = None
        bias = True
        
        for i in range(cfg.MODEL.AVOD.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.AVOD.NUM_CONVS - 1:
                conv_func = DeformConv
                bias = False
            else:
                conv_func = Conv2d
                
            if pt_tower is not None:
                tower_build(pt_tower, conv_func, in_channels, 
                        in_channels, 3, 1, 1, bias)
            
            tower_build(cls_tower, conv_func, in_channels, 
                        in_channels, 3, 1, 1, bias)
            
            tower_build(bbox_tower, conv_func, in_channels, 
                        in_channels, 3, 1, 1, bias)

            
        if pt_tower is not None:
             self.add_module('pt_tower', nn.Sequential(*pt_tower))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred_size = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        
             
        self.bbox_center = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        )
        
        self.bbox_conf = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        
        
        # initialization
        init_modules = []
        init_modules.append(self.cls_tower)
        init_modules.append(self.bbox_tower)
        init_modules.append(self.cls_logits)
        init_modules.append(self.bbox_pred_size)
        init_modules.append(self.bbox_center)
        init_modules.append(self.bbox_conf)
        
        if pt_tower is not None: 
            init_modules.append(self.pt_tower)
        
        for modules in init_modules:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.AVOD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        self.scales_center = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg_size = []
        center = []
        confs = []
        for l, feature in enumerate(x):
            
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            
            confs.append(self.bbox_conf(box_tower))

            if self.pt_tower_define:
                pt_tower = self.pt_tower(feature)
                bbox_pred_size = self.scales[l](self.bbox_pred_size(pt_tower))
            else:
                bbox_pred_size = self.scales[l](self.bbox_pred_size(box_tower))
            bbox_pre_center = self.scales_center[l](self.bbox_center(box_tower))

            if self.norm_reg_targets:
                bbox_pred_size = F.relu(bbox_pred_size)
                if self.training:
                    bbox_reg_size.append(bbox_pred_size)
                    center.append(bbox_pre_center)
                else:
                    bbox_reg_size.append(bbox_pred_size * self.fpn_strides[l])
                    center.append(bbox_pre_center * self.fpn_strides[l])
            else:
                bbox_reg_size.append(torch.exp(bbox_pred_size))
                center.append(torch.exp(bbox_pre_center))
                

        prediction = {
            "logits": logits,
            "bbox_reg_size": bbox_reg_size,
            "center": center,
            "confs": confs,
        }
        return prediction

