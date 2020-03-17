# -*- coding: utf-8 -*-
from detectron2.utils.registry import Registry
IENET_HEAD_REGISTRY = Registry("IENET_HEAD")
IENET_HEAD_IN_REGISTRY = Registry("IENET_HEAD_INFERENCE")
IENET_HEAD_LO_REGISTRY = Registry("IENET_HEAD_LOSS")

def build_avod_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.AVOD.HEAD_NAME + '_HEAD'
    return IENET_HEAD_REGISTRY.get(name)(cfg, input_shape)




def build_avod_postprocessor(cfg):
    """
    Build an IEnet head defined by `cfg.MODEL.AVOD.HEAD_NAME`.
    """
    
    name = cfg.MODEL.AVOD.HEAD_NAME + '_IN'
    pre_nms_thresh = cfg.MODEL.AVOD.INFERENCE_TH
    pre_nms_top_n = cfg.MODEL.AVOD.PRE_NMS_TOP_N
    nms_thresh = cfg.MODEL.AVOD.NMS_TH
    fpn_post_nms_top_n = cfg.TEST.DETECTIONS_PER_IMAGE
#    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    return IENET_HEAD_IN_REGISTRY.get(name)(
            pre_nms_thresh=pre_nms_thresh,
            pre_nms_top_n=pre_nms_top_n,
            nms_thresh=nms_thresh,
            fpn_post_nms_top_n=fpn_post_nms_top_n,
            min_size=0,
            num_classes=cfg.MODEL.AVOD.NUM_CLASSES,
#            bbox_aug_enabled=bbox_aug_enabled
        )
    




def build_avod_loss_evaluator(cfg):
    """
    Build an IEnet head defined by `cfg.MODEL.AVOD.HEAD_NAME`.
    """
    
    name = cfg.MODEL.AVOD.HEAD_NAME + '_LO'
    return IENET_HEAD_LO_REGISTRY.get(name)(cfg)