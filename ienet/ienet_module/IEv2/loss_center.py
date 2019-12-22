# -*- coding: utf-8 -*-

import torch
from torch.nn import functional as F
from torch import nn
import os
from fvcore.nn import smooth_l1_loss, sigmoid_focal_loss_star_jit
from ienet.layers import IOULoss, smooth_l1_loss_weight, SigmoidFocalLoss


from ienet.ienet_module.build import IENET_HEAD_LO_REGISTRY

INF = 100000000




def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor

@IENET_HEAD_LO_REGISTRY.register()
class AVODWH_CENTER_LO(object):
    """
    This class computes the AVOD losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.AVOD.LOSS_GAMMA,
            cfg.MODEL.AVOD.LOSS_ALPHA
        )

        self.fpn_strides = cfg.MODEL.AVOD.FPN_STRIDES 
        self.center_sampling_radius = cfg.MODEL.AVOD.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.AVOD.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.AVOD.NORM_REG_TARGETS


        self.box_iou_func = IOULoss(self.iou_loss_type)
        self.box_size_l1_func = smooth_l1_loss_weight()
        self.bce_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)

        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask
    
    @torch.no_grad()
    def prepare_targets(self, points, targets):
        if len(self.fpn_strides) == 5:
            object_sizes_of_interest = [
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, INF],
            ]

        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, pt_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            pt_targets[i] = torch.split(pt_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        pt_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)
            
            pt_targets_per_level = torch.cat([
                pt_targets_per_im[level]
                for pt_targets_per_im in pt_targets
            ], dim=0)
            

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
                pt_targets_per_level = pt_targets_per_level / self.fpn_strides[level]
 
            reg_targets_level_first.append(reg_targets_per_level)
            pt_targets_level_first.append(pt_targets_per_level)

        return labels_level_first, reg_targets_level_first, pt_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        pt_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for per_im_target in targets:
            bboxes = per_im_target['pt_hbb']
            pt_in_bboxes = per_im_target['pt_inbox']
            pt_bboxes = per_im_target['poly'].reshape(-1, 8)
            labels_per_im = per_im_target['labels']
            area = per_im_target['area']
            
            center_x = (bboxes[:, 2] + bboxes[:, 0]) / 2
            center_y = (bboxes[:, 3] + bboxes[:, 1]) / 2
            
            center_w = bboxes[:, 2] - bboxes[:, 0]
            center_h = bboxes[:, 3] - bboxes[:, 1]
#            
            
            point_w1 = pt_in_bboxes[:, 2]
            point_h1 = pt_in_bboxes[:, 3]
            point_w2 = center_w - point_w1
            point_h2 = center_h - point_h1
            
            point_w1 = point_w1[None].repeat(xs.size(0), 1)
            point_w2 = point_w2[None].repeat(xs.size(0), 1)
            point_h1 = point_h1[None].repeat(ys.size(0), 1)
            point_h2 = point_h2[None].repeat(ys.size(0), 1)
            
            x1 = pt_bboxes[:, 0][None] - xs[:, None]
            y1 = pt_bboxes[:, 1][None] - ys[:, None]
            x2 = pt_bboxes[:, 2][None] - xs[:, None]
            y2 = pt_bboxes[:, 3][None] - ys[:, None]
            x3 = pt_bboxes[:, 4][None] - xs[:, None]
            y3 = pt_bboxes[:, 5][None] - ys[:, None]
            x4 = pt_bboxes[:, 6][None] - xs[:, None]
            y4 = pt_bboxes[:, 7][None] - ys[:, None]
            
            center_x = center_x[None] - xs[:, None]
            center_y = center_y[None] - ys[:, None]
            
            center_targets_per_im = torch.stack([center_x, center_y], dim=2)
#            point_targets_per_im = torch.stack([point_w1, point_w2,
#                                                point_h1, point_h2], dim=2)
            point_targets_per_im = torch.stack([x1, y1, 
                                                x2, y2, 
                                                x3, y3, 
                                                x4, y4,
                                                point_w1, point_w2,
                                                point_h1, point_h2], dim=2)
            

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = point_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = point_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

#            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            
            center_targets_per_im = center_targets_per_im[range(len(locations)), locations_to_gt_inds]
            
            point_targets_per_im = point_targets_per_im[range(len(locations)), locations_to_gt_inds]

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(center_targets_per_im)
            pt_targets.append(point_targets_per_im)

        return labels, reg_targets, pt_targets

    def compute_targets(self, reg_targets, center_targets, pred_reg, pred_center):
        reg_targets_size = reg_targets[:, [0, 1]] + reg_targets[:, [2, 3]]
        reg_targets_size_half = reg_targets_size * 0.5
        pred_reg_size = pred_reg[:, [0, 1]] + pred_reg[:, [2, 3]]
        pred_reg_size_half = pred_reg_size * 0.5
        lt = reg_targets_size_half[:, 0] - center_targets[:, 0]
        tt = reg_targets_size_half[:, 1] + center_targets[:, 1]
        rt = reg_targets_size_half[:, 0] + center_targets[:, 0]
        bt = reg_targets_size_half[:, 1] - center_targets[:, 1]
        ltrb_targets = torch.stack([lt, tt, rt, bt], dim=1)
        lp = pred_reg_size_half[:, 0] - pred_center[:, 0]
        tp = pred_reg_size_half[:, 1] + pred_center[:, 1]
        rp = pred_reg_size_half[:, 0] + pred_center[:, 0]
        bp = pred_reg_size_half[:, 1] - pred_center[:, 1]
        ltrb_pred = torch.stack([lp, tp, rp, bp], dim=1)

        
        return ltrb_targets, ltrb_pred
   
    def compute_centerness_targets(self, reg_targets):
        N = reg_targets.size(0)
        reg_targets = reg_targets.reshape(N, 4, 2)
        reg_targets = torch.pow(reg_targets, 2)
        reg_targets = torch.sum(reg_targets, 2)
        reg_targets = torch.sqrt(reg_targets)
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, prediction, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        
        box_cls = prediction["logits"]
        bbox_reg_size = prediction["bbox_reg_size"]
        centerness = prediction["center"]
        confs = prediction["confs"]

        
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        
        labels, reg_targets, box_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        bbox_reg_size_flatten = []
        centerness_flatten = []
        confs_flatten = []
        labels_flatten = []
        center_targets_flatten = []
        box_targets_flatten = []
        
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            bbox_reg_size_flatten.append(bbox_reg_size[l].permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_flatten.append(centerness[l].permute(0, 2, 3, 1).reshape(-1, 2))
            confs_flatten.append(confs[l].reshape(-1))

            labels_flatten.append(labels[l].reshape(-1))
            box_targets_flatten.append(box_targets[l].reshape(-1, 12))
            center_targets_flatten.append(reg_targets[l].reshape(-1, 2))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        bbox_reg_size_flatten = torch.cat(bbox_reg_size_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        confs_flatten = torch.cat(confs_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        box_targets_flatten = torch.cat(box_targets_flatten, dim=0)
        center_targets_flatten = torch.cat(center_targets_flatten, dim=0)
        

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        bbox_reg_size_flatten = bbox_reg_size_flatten[pos_inds]
        center_targets_flatten = center_targets_flatten[pos_inds]
        box_targets_flatten = box_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        confs_flatten = confs_flatten[pos_inds]
        

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)


        
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int(),
        ) / num_pos_avg_per_gpu
        


        if pos_inds.numel() > 0:
            box_targets_ltrb = box_targets_flatten[:, 0:8]
            confidance_targets = self.compute_centerness_targets(box_targets_ltrb)
            
            decode = self.compute_targets(box_targets_flatten[:, 8:12],
                                          center_targets_flatten,
                                          bbox_reg_size_flatten,
                                          centerness_flatten)
            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(confidance_targets.sum()).item() / float(num_gpus)
                
            reg_size_loss = self.box_size_l1_func(
                bbox_reg_size_flatten,
                box_targets_flatten[:, 8:12],
                confidance_targets
            ) / sum_centerness_targets_avg_per_gpu
                    
            reg_iou_loss = self.box_iou_func(
                decode[1],
                decode[0],
                confidance_targets
            ) / sum_centerness_targets_avg_per_gpu
                    
                    
            center_loss = self.box_size_l1_func(
                centerness_flatten,
                center_targets_flatten
            ) / num_pos_avg_per_gpu
                    
            conf_loss = self.bce_loss_func(
                confs_flatten,
                confidance_targets
            ) / num_pos_avg_per_gpu
                    

        else:
            reg_size_loss = bbox_reg_size_flatten.sum()
            reg_iou_loss = bbox_reg_size_flatten.sum()
            center_loss = centerness_flatten.sum()
            reduce_sum(confs_flatten.new_tensor([0.0]))
            conf_loss = confs_flatten.sum()


        losses = {
            "loss_cls": cls_loss,
            "reg_iou_loss": reg_iou_loss,
            "reg_size_loss": reg_size_loss * 0.1,
            "center_loss": center_loss,
            "conf_loss": conf_loss,
            }

        return losses


