# -*- coding: utf-8 -*-

import torch
from projects.Avod.dataset.dataset_tool import (
        batch_hbb_hw2poly, 
        batch_polygonToRotRectangle,
        polygonToRotRectangle
    )
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    RotatedBoxes
)
import math
import cv2 as cv
import numpy as np
from detectron2.layers import nms_rotated, batched_nms_rotated

poly_rotated_box = [
        [690.0, 91.0 , 687.0, 84.0, 701.0, 78.0, 704.0, 85.0],
        [419.0, 745.0, 420.0, 756.0, 337.0, 753.0, 337.0, 742.0],
    ]
polys_from_rot = [
        [701.,  77., 704.,  85., 689.,  91., 686.,  83.],
        [418., 734., 420., 756., 337., 763., 335., 741.]
    ]
poly_rotated_box = np.array(polys_from_rot)


def get_all_groundthurth(poly):
    poly = poly.reshape(4, 2)
    
    OBB_box = polygonToRotRectangle(poly.reshape(-1))
            
    theta = float(OBB_box[4])
    
    theta = math.degrees(theta)
    #            theta = theta + rota
    
    if theta > 90.0:
        theta -= 180
    elif theta < -90.0:
        theta += 180
    
    if theta == -90:
        theta = 90
        
    obb_bndbox = ((OBB_box[0], OBB_box[1]),
               (OBB_box[2], OBB_box[3]),
               theta)
    hbb_box = cv.boxPoints(obb_bndbox)
    hbb_box = np.int0(hbb_box)
    
    pt_x_y_min = hbb_box.min(axis= 0)
    pt_x_y_max = hbb_box.max(axis= 0)
    
    hrbb_box = np.hstack((pt_x_y_min, pt_x_y_max))
    
    
    
    hrbb_center = [(hrbb_box[0] + hrbb_box[2]) / 2,
                   (hrbb_box[1] + hrbb_box[3]) / 2]
    
    
    if theta < 0:
        pt_h = hbb_box[1][1] - hrbb_box[1]
        pt_w = hbb_box[2][0] - hrbb_box[0] 
    else:
        pt_h = hbb_box[0][1] - hrbb_box[1]
        pt_w = hbb_box[1][0] - hrbb_box[0]
    
    pt_inbox = [hrbb_center[0] - (pt_w / 2), 
                hrbb_center[1] - (pt_h / 2), 
                pt_w, pt_h]
    
    obb_box = [
            OBB_box[0], OBB_box[1],
            OBB_box[2], OBB_box[3],
            theta
        ]

    return hrbb_box, pt_inbox, obb_box

hbbs = []
pt_ins = []
obbs = []
for poly in poly_rotated_box:
    box = get_all_groundthurth(poly)
    hbbs.append(box[0])
    pt_ins.append(box[1])
    obbs.append(box[2])
    
hbbs = np.array(hbbs)
pt_ins = np.array(pt_ins)
obbs = np.array(obbs)
hbbs = torch.from_numpy(hbbs).to(torch.device('cuda'))
pt_ins = torch.from_numpy(pt_ins).to(torch.device('cuda'))
obbs = torch.from_numpy(obbs).to(torch.device('cuda'))

hbbs_wh = BoxMode.convert(
                hbbs, 
                BoxMode.XYXY_ABS, 
                BoxMode.XYWH_ABS
            )
polys = batch_hbb_hw2poly(
        hbbs.float(), hbbs_wh.float(), 
        pt_ins.float(), 'tensor'
    )
rotated_boxes = batch_polygonToRotRectangle(polys)
#pred_boxes = RotatedBoxes(rotated_boxes)
#rotated_box = batch_polygonToRotRectangle(poly_rotated_box)
scores = torch.Tensor([0.89, 0.92])
pred_classes = torch.Tensor([0, 0])
rotated_boxes[0, 4]= 90.0
rotated_boxes[1, 4]= 90.0
keep = batched_nms_rotated(rotated_boxes, scores.cuda(), pred_classes,  0.5)