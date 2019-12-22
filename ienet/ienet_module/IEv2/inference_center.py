# -*- coding: utf-8 -*-


import torch


from detectron2.layers import batched_nms_rotated, nms_rotated, cat

from projects.Avod.dataset.dataset_tool import (
        batch_hbb_hw2poly, 
        batch_polygonToRotRectangle,
        batch_get_best_begin_point
    )

#from projects.Avod.structures import RotatedBoxes
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    RotatedBoxes
)
import numpy as np
from ienet.ienet_module.build import IENET_HEAD_IN_REGISTRY
from dataset.dataset_tool import boxlist_nms_poly

@IENET_HEAD_IN_REGISTRY.register()
class AVODWH_CENTER_IN(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(AVODWH_CENTER_IN, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes 
        self.bbox_aug_enabled = bbox_aug_enabled
        self.num_after_nms = []

    def forward_for_single_feature_map(
            self, locations, box_cls, box_regression, center, confs,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """

        

        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        center = center.view(N, 2, H, W).permute(0, 2, 3, 1)
        center = center.reshape(N, -1, 2)
        confs = confs.view(N, 1, H, W).permute(0, 2, 3, 1)
        confs = confs.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        box_cls = box_cls * confs[:, :, None]
        
        pred_boxes = []
        pred_boxes_pt = []
        scores = []
        pred_classes = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] 

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            
            per_center = center[i]
            per_center = per_center[per_box_loc]
            
            
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_center = per_center[top_k_indices]
                per_locations = per_locations[top_k_indices]

            pbr_w = per_box_regression[:, 0] + per_box_regression[:, 1]
            pbr_h = per_box_regression[:, 2] + per_box_regression[:, 3]
            pbr_w_half = pbr_w / 2
            pbr_h_half = pbr_h / 2
            detections = torch.stack([
                per_locations[:, 0] + per_center[:, 0] - pbr_w_half,
                per_locations[:, 1] + per_center[:, 1] - pbr_h_half,
                per_locations[:, 0] + per_center[:, 0] + pbr_w_half,
                per_locations[:, 1] + per_center[:, 1] + pbr_h_half,
            ], dim=1)
    
            pt_detections = torch.stack([
                per_locations[:, 0] + per_center[:, 0] - pbr_w_half,
                per_locations[:, 1] + per_center[:, 1] - pbr_h_half,
                per_box_regression[:, 0],
                per_box_regression[:, 2],
            ], dim=1)


            detections_wh = BoxMode.convert(
                        detections, 
                        BoxMode.XYXY_ABS, 
                        BoxMode.XYWH_ABS
                    )
            poly = batch_hbb_hw2poly(
                        detections, 
                        detections_wh, 
                        pt_detections,
                        dtype='tensor',
                    )
            poly = poly.view(-1, 8)

            rot_box = batch_polygonToRotRectangle(poly)
            rot_box = rot_box.view(-1, 5)
            score = torch.sqrt(per_box_cls)
            pred_boxes.append(rot_box)
            scores.append(score)
            pred_classes.append(per_class)
            


        return pred_classes, scores, pred_boxes

    def forward(self, locations, prediction, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
#        sampled_boxes = []
        self.num_after_nms = []
        box_cls = prediction['logits']
        box_regression = prediction['bbox_reg_size']
        centerness = prediction['center']
        confs = prediction["confs"]
        
        pred_classes = []
        scores = []
        pred_boxes = []
        for _, (l, o, b, c, s) in enumerate(zip(locations, box_cls, box_regression, centerness, confs)):
            sampled_boxes = self.forward_for_single_feature_map(
                    l, o, b, c, s, image_sizes
                )
            pred_classes.append(sampled_boxes[0])
            scores.append(sampled_boxes[1])
            pred_boxes.append(sampled_boxes[2])
            

        pred_classes_lists = list(zip(*pred_classes))
        scores_lists = list(zip(*scores))
        pred_boxes_lists = list(zip(*pred_boxes))
        
        pred_classes_lists = [cat(boxlist) for boxlist in pred_classes_lists]
        scores_lists = [cat(boxlist) for boxlist in scores_lists]
        pred_boxes_lists = [cat(boxlist) for boxlist in pred_boxes_lists]

            
        if not self.bbox_aug_enabled:
            results = self.select_over_all_levels(
                    pred_boxes_lists, scores_lists,
                    pred_classes_lists, image_sizes
                )

        return results

    def select_over_all_levels(self, bboxlist, scorelist, 
                               cls_list, image_sizes):
        # num_images = len(image_sizes)
        results = []
        for i, (boxes, labels, scores) in enumerate(
                zip(bboxlist, cls_list, scorelist)):
            # skip the background
            
            keep = batched_nms_rotated(
                boxes, scores, labels, self.nms_thresh
            )   

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

             
            number_of_detections = boxes.size(0)
            
            
            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = scores.clone()
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
            result = Instances(image_sizes[i])
            result.pred_boxes = RotatedBoxes(boxes)
            result.scores = scores
            result.pred_classes = labels
            results.append(result)
        return results


