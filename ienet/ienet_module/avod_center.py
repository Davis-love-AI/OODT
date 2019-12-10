# -*- coding: utf-8 -*-


import math
import torch
import torch.nn.functional as F
from torch import nn

from ..layers import Scale, Self_Attn
from detectron2.layers import DeformConv
from detectron2.layers import Conv2d
from .build import IENET_HEAD_REGISTRY



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

        cls_tower = []
        bbox_tower = []
#        pt_tower = []
        bias = True
        
        for i in range(cfg.MODEL.AVOD.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.AVOD.NUM_CONVS - 1:
                conv_func = DeformConv
                bias = False
            else:
                conv_func = nn.Conv2d
                
                
#            pt_tower.append(
#                conv_func(
#                    in_channels,
#                    in_channels,
#                    kernel_size=3,
#                    stride=1,
#                    padding=1,
#                    bias=True
#                )
#            )
#            pt_tower.append(nn.GroupNorm(32, in_channels))
#            pt_tower.append(nn.ReLU())

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
            

#        self.add_module('pt_tower', nn.Sequential(*pt_tower))
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
        
#        self.bbox_pred_insize = nn.Conv2d(
#            in_channels, 2, kernel_size=3, stride=1,
#            padding=1
#        )
             
        self.bbox_center = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        )
        
        self.bbox_conf = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        
#        self.quarter_w = nn.Conv2d(
#            in_channels, 1, kernel_size=3, stride=1,
#            padding=1
#        )
#        self.quarter_h = nn.Conv2d(
#            in_channels, 1, kernel_size=3, stride=1,
#            padding=1
#        )


#        self.atten_pt = Self_Attn(in_channels, 'relu')
#        self.atten_box = Self_Attn(in_channels, 'prelu')

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred_size,
                        self.bbox_center, self.bbox_conf]:
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
#        bbox_reg_insize = []
        center = []
        confs = []
#        quarters_w = []
#        quarters_h = []
        for l, feature in enumerate(x):
#            feature, p1 = self.atten_pt(feature)
#            pt_tower = F.relu(feature)
#            pt_tower = self.pt_tower(feature)
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
#            pt_tower = F.relu(pt_tower)
#            atten_cls, p1 = self.atten_cls(cls_tower.detach())
#            atten_box, p2 = self.atten_box(box_tower)
#            atten_cls = F.hardshrink(atten_cls)
#            atten_box = F.softshrink(atten_box)

            logits.append(self.cls_logits(cls_tower))

#            center.append(self.bbox_center(box_tower))
            
            confs.append(self.bbox_conf(box_tower))
            
#            quarters_w.append(self.quarter_w(pt_tower))
#            quarters_h.append(self.quarter_h(pt_tower))
            
#            bbox_reg.append(F.relu(self.bbox_pred(box_tower)))
            
            bbox_pred_size = self.scales[l](self.bbox_pred_size(box_tower))
            bbox_pre_center = self.scales_center[l](self.bbox_center(box_tower))
#            bbox_reg_size.append(bbox_pred_size)
            
#            bbox_pred_insize = self.bbox_pred_insize(box_tower)
#            bbox_reg_insize.append(bbox_pred_insize)
            if self.norm_reg_targets:
#                bbox_pred_size = self.conv_size(bbox_pred_size)
#                bbox_pre_center = self.conv_cen(bbox_pre_center)
                if self.training:
                    bbox_reg_size.append(bbox_pred_size)
                    center.append(bbox_pre_center)
                else:
                    bbox_reg_size.append(bbox_pred_size * self.fpn_strides[l])
                    center.append(bbox_pre_center * self.fpn_strides[l])
            else:
                bbox_reg_size.append(torch.exp(bbox_pred_size))
                center.append(torch.exp(bbox_pre_center))
                
#            bbox_center = self.scales_center[l](self.bbox_center(pt_tower))
#            if self.norm_reg_targets:
#                bbox_center = F.relu(bbox_center)
#                if self.training:
#                    center.append(bbox_center)
#                else:
#                    center.append(bbox_center * self.fpn_strides[l])
#            else:
#                center.append(torch.exp(bbox_pred))
                
            
                
            
#            bbox_pt_h = self.scales_ltrb[l](self.bbox_pt_h(pt_tower))
#            if self.norm_reg_targets:
#                bbox_pt_h = F.relu(bbox_pt_h)
#                if self.training:
#                    pt_reg_h.append(bbox_pt_h)
#                else:
#                    pt_reg_h.append(bbox_pt_h * self.fpn_strides[l])
#            else:
#                pt_reg_h.append(torch.exp(bbox_pt_h))
        prediction = {
            "logits": logits,
            "bbox_reg_size": bbox_reg_size,
#            "bbox_reg_insize": bbox_reg_insize,
            "center": center,
            "confs": confs,
#            "quarters_w": quarters_w,
#            "quarters_h": quarters_h,
#            "p1": p1,
#            "p2": p2,
        }
        return prediction

