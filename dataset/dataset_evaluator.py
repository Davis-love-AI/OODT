# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from detectron2.evaluation import DatasetEvaluator
from .point_tool import polygonToRotRectangle, hrbb2hw_obb

import math
import cv2 as cv
from DOTA_devkit import polyiou
from DOTA_devkit.dota_evaluation_task1 import voc_eval as dota_val
from .dataset_tool import preprocess_annotation
from .eval_tool import decode_result
from detectron2.layers.rotated_boxes import pairwise_iou_rotated

class DotaVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name, eval_type = 'hw'):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self.eval_type = eval_type
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "labelxml_voc", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", meta.split + ".txt")
        self._class_names = meta.thing_classes
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings
        self._predictions_om = defaultdict(list)
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
##            obb_boxes = instances.pred_obb_boxes.tensor.numpy()
#            if self.eval_type == 'hw':
#                boxes = instances.pred_boxes.tensor.numpy()
#            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            decode_result(self._predictions, self._predictions_om, classes, 
                          image_id, scores, instances, 
                           self.eval_type)
            
            

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        all_predictions_om = comm.gather(self._predictions_om, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        predictions_om = defaultdict(list)
        
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions
        for predictions_per_rank in all_predictions_om:
            for clsid, lines in predictions_per_rank.items():
                predictions_om[clsid].extend(lines)
        del all_predictions_om

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007
            )
        )
        prefix="pascal_dota_eval_" + self.eval_type
        
        if self.eval_type == 'hw':
            dirname = 'projects/Avod/output/' + prefix
            
            if not os.path.exists(dirname):
                os.mkdir(dirname)
                
            res_file_template = os.path.join(dirname, "Task1_{}.txt")
    
            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])
    
                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))
                    
        with tempfile.TemporaryDirectory(prefix=prefix) as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")
    
            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])
    
                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))
                    
    
                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=True,
                        eval_type=self.eval_type,
                    )
                    aps[thresh].append(ap * 100)
                    
        ret = OrderedDict()
        x = aps[50]
        cls_ret = OrderedDict()
        for clsid, ix in enumerate(x):
            cls_ret[self._class_names[clsid]] = ix
        
        ret["Cls_50"] = cls_ret
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP    ": np.mean(list(mAP.values())), "AP50  ": mAP[50], "AP75  ": mAP[75]}
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text.lower().strip()

        obj_struct["difficult"] = int(obj.find("difficult").text)
        bb = obj.find("bndbox")
        
        OBB_box = [
            float(bb.find("center_x").text),
            float(bb.find("center_y").text),
            float(bb.find("box_width").text),
            float(bb.find("box_height").text),
        ]
        
        point_box = [
                float(bb.find("x1").text),
                float(bb.find("y1").text),
                float(bb.find("x2").text),
                float(bb.find("y2").text),
                float(bb.find("x3").text),
                float(bb.find("y3").text),
                float(bb.find("x4").text),
                float(bb.find("y4").text),
            ]
        
        OBB_box = polygonToRotRectangle(point_box)
        
        theta = float(bb.find("box_ang").text)
        
        theta = math.degrees(theta)
        
        rot_box = [OBB_box[0], OBB_box[1],
                   OBB_box[2], OBB_box[3],
                   -theta]
          
        
        if theta > 90.0:
            theta -= 180
        elif theta < -90.0:
            theta += 180
            
        if theta == 90.0:
            theta = -90
        
        
        hbb_box = ((OBB_box[0], OBB_box[1]),
                   (OBB_box[2], OBB_box[3]),
                   theta)
        hbb_box = cv.boxPoints(hbb_box)
        hbb_box = np.int0(hbb_box)
        pt_x_y_min = hbb_box.min(axis= 0)
        pt_x_y_max = hbb_box.max(axis= 0)
    
        hrbb_box = np.hstack((pt_x_y_min, pt_x_y_max))
        
        obj_struct["obbox"] = hbb_box.reshape(-1)
        
        obj_struct["rotbox"] = rot_box
        obj_struct["hrbb_box"] = hrbb_box
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, eval_type='hw'):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    reshape_num = 4
    box_type = "obbox"
    if eval_type == 'hbb':
        box_type = "hrbb_box"
        reshape_num = 4
    if eval_type == 'hw':
        box_type = "obbox"
        reshape_num = 8
    else:
        box_type = "rotbox"
        reshape_num = 5
        
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]

        bbox = np.array([x[box_type] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, reshape_num)
    

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)
        

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            if eval_type == 'hbb': 
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih
    
                # union
                uni = (
                    (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                    - inters
                )
    
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            if eval_type == 'hw':
                # compute overlaps
                # intersection
    
                # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
                # pdb.set_trace()
                BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
                BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
                BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
                BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
                bb_xmin = np.min(bb[0::2])
                bb_ymin = np.min(bb[1::2])
                bb_xmax = np.max(bb[0::2])
                bb_ymax = np.max(bb[1::2])
    
                ixmin = np.maximum(BBGT_xmin, bb_xmin)
                iymin = np.maximum(BBGT_ymin, bb_ymin)
                ixmax = np.minimum(BBGT_xmax, bb_xmax)
                iymax = np.minimum(BBGT_ymax, bb_ymax)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
    
                # union
                uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                       (BBGT_xmax - BBGT_xmin + 1.) *
                       (BBGT_ymax - BBGT_ymin + 1.) - inters)
    
                overlaps = inters / uni
    
                BBGT_keep_mask = overlaps > 0
                BBGT_keep = BBGT[BBGT_keep_mask, :]
                BBGT_keep_index = np.where(overlaps > 0)[0]
                # pdb.set_trace()
                def calcoverlaps(BBGT_keep, bb):
                    overlaps = []
                    for index, GT in enumerate(BBGT_keep):
    
                        overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                        overlaps.append(overlap)
                    return overlaps
                if len(BBGT_keep) > 0:
                    overlaps = calcoverlaps(BBGT_keep, bb)
    
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                    # pdb.set_trace()
                    jmax = BBGT_keep_index[jmax]
            else:
                BBGT_gpu = torch.from_numpy(BBGT).view(-1, 5).float().cuda()
                bb_gpu = torch.from_numpy(bb).view(-1, 5).float().cuda()
                overlaps = pairwise_iou_rotated(BBGT_gpu, bb_gpu).cpu()
                overlaps = overlaps.view(-1).numpy()
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
