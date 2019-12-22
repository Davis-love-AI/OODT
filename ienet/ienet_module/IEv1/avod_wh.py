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
class AVODWH_WH_HEAD(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(AVODWH_WH_HEAD, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.AVOD.NUM_CLASSES 
        self.fpn_strides = cfg.MODEL.AVOD.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.AVOD.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.AVOD.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.AVOD.USE_DCN_IN_TOWER
        self.pt_tower_define = cfg.MODEL.AVOD.INDEPENDENT_BRANCH
        self.attention_define = cfg.MODEL.AVOD.ATTENTION_ON

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
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        
                
        self.bbox_pt = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        )
        
        if self.attention_define:
            self.atten_feature = Self_Attn(in_channels, 'relu')
        
        
        # initialization
        init_modules = []
        init_modules.append(self.cls_tower)
        init_modules.append(self.bbox_tower)
        init_modules.append(self.cls_logits)
        init_modules.append(self.bbox_pred)
        init_modules.append(self.centerness)
        init_modules.append(self.bbox_pt)
        
        if pt_tower is not None: 
            init_modules.append(self.pt_tower)
        if self.attention_define:
            init_modules.append(self.atten_feature)
        
        for modules in init_modules:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)


        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.AVOD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales_ltrb = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        self.scales_wh = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        pt_reg = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)


            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))
                
            
            if self.attention_define:
                pt, p_pt = self.atten_box(cls_tower + box_tower)
            else:
                pt = 0
                p_pt = None
                
            if self.pt_tower_define:
                pt_tower = self.pt_tower(feature)
                pt_tower = pt_tower + pt
                bbox_pt = self.scales_wh[l](self.bbox_pt(pt_tower))
            else:
                if pt != 0:
                    bbox_pt = self.scales_wh[l](self.bbox_pt(pt))
                else:
                    bbox_pt = self.scales_wh[l](self.bbox_pt(box_tower))


            bbox_pred = self.scales_ltrb[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                bbox_pt = F.relu(bbox_pt)
                if self.training:
                    bbox_reg.append(bbox_pred)
                    pt_reg.append(bbox_pt)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
                    pt_reg.append(bbox_pt * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
                pt_reg.append(torch.exp(bbox_pt))
                
       

        prediction = {
            "logits": logits,
            "bbox_reg": bbox_reg,
            "centerness": centerness,
            "pt_reg": pt_reg,
            "p_pt": p_pt,
        }
        return prediction