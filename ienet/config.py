# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_ienet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.AVOD = CN()
    _C.MODEL.AVOD.NUM_CLASSES = 15 
    _C.MODEL.AVOD.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.AVOD.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.AVOD.PRIOR_PROB = 0.01
    _C.MODEL.AVOD.INFERENCE_TH = 0.3
    _C.MODEL.AVOD.NMS_TH = 0.5
    _C.MODEL.AVOD.PRE_NMS_TOP_N = 1000
    
    # Focal loss parameter: alpha
    _C.MODEL.AVOD.LOSS_ALPHA = 0.25
    # Focal loss parameter: gamma
    _C.MODEL.AVOD.LOSS_GAMMA = 2.0
    
    # the number of convolutions used in the cls and bbox tower
    _C.MODEL.AVOD.NUM_CONVS = 4
    
    # if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
    _C.MODEL.AVOD.CENTER_SAMPLING_RADIUS = 0.0
    # IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
    _C.MODEL.AVOD.IOU_LOSS_TYPE = "iou"
    
    _C.MODEL.AVOD.NORM_REG_TARGETS = False
    _C.MODEL.AVOD.CENTERNESS_ON_REG = False
    
    _C.MODEL.AVOD.USE_DCN_IN_TOWER = False
    _C.MODEL.AVOD.INDEPENDENT_BRANCH = False
    _C.MODEL.AVOD.ATTENTION_ON = False
    _C.MODEL.AVOD.HEAD_NAME = "avod"
    _C.MODEL.AVOD.ROTA_AUG_ON = True
    _C.MODEL.AVOD.IN_CHANNELS = 256
    root = '/home/omnisky/Pengming_workspace/'
    root += 'disk_2T/DOTA/DOTA_dataset/{}'
    _C.MODEL.AVOD.DATASET = root
