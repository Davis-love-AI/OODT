# -*- coding: utf-8 -*-
import cv2
import math
import sys
import numpy as np
import torch
import torchgeometry as tgm
from DOTA_devkit.poly_nms_gpu.nms_wrapper import poly_nms_gpu

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

CLASSES = (
         'plane', 
         'baseball-diamond', 
         'bridge', 
         'ground-track-field', 
         'small-vehicle', 
         'large-vehicle', 
         'ship', 
         'tennis-court', 
         'basketball-court', 
         'storage-tank',  
         'soccer-ball-field', 
         'roundabout', 
         'harbor', 
         'swimming-pool', 
         'helicopter', 
#         'container-crane',
    ) 
  
class_to_ind = dict(zip(CLASSES, range(len(CLASSES))))

def boxlist_nms_poly(boxlist, score, nms_thresh, max_proposals=-1):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist

    boxes = boxlist.cpu().numpy()
    score = score.cpu().numpy()
#    if boxes.shape[0] == 0:
#        return boxlist
    det = np.append(boxes, score[:, None], axis=1).astype(np.float32)
#    keep = _box_nms(boxes, score, nms_thresh)
    keep = poly_nms_gpu(det, np.float(nms_thresh))
    if max_proposals > 0:
        keep = keep[: max_proposals]
    
    return keep

def polygonToRotRectangle(bbox):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
    """
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(2,4),order='F')
    angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])

    center = [[0],[0]]

    for i in range(4):
        center[0] += bbox[0,i]
        center[1] += bbox[1,i]

    center = np.array(center,dtype=np.float32)/4.0

    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose(),bbox-center)

    xmin = np.min(normalized[0,:])
    xmax = np.max(normalized[0,:])
    ymin = np.min(normalized[1,:])
    ymax = np.max(normalized[1,:])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    return [float(center[0]),float(center[1]),w,h,angle]

def batch_polygonToRotRectangle(bbox):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
    """
    bbox = torch.stack([bbox[:, 0::2], bbox[:, 1::2]], dim=1)
    angle = torch.atan2(-(bbox[:, 0,1]-bbox[:, 0,0]), 
                        bbox[:, 1,1]-bbox[:, 1,0])

    center = torch.zeros(bbox.size(0), 2, 1, dtype=bbox.dtype, device=bbox.device)

    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = center / 4.0
    R = torch.stack([torch.cos(angle), -torch.sin(angle), 
                     torch.sin(angle), torch.cos(angle)], dim=1)
    R = R.reshape(-1, 2, 2)

    normalized = torch.matmul(R.transpose(2,1), bbox - center)
    
    if bbox.size(0) == 0:
        return torch.empty((0, 5), dtype=bbox.dtype, device=bbox.device)

    xmin = torch.min(normalized[:, 0,:], dim=1)[0]
    xmax = torch.max(normalized[:, 0,:], dim=1)[0]
    ymin = torch.min(normalized[:, 1,:], dim=1)[0]
    ymax = torch.max(normalized[:, 1,:], dim=1)[0]

    w = xmax - xmin 
    h = ymax - ymin

    center = center.squeeze(-1)
    center_x = center[:, 0]
    center_y = center[:, 1]
    new_box = torch.stack([center_x, center_y, w, h, -tgm.rad2deg(angle)], dim=1)
    return new_box
 
def batch_hbb_hw2poly(proposal_xy, proposal_wh, hw, dtype='np'):
    hrbb_x_min = proposal_xy[:, 0]
    hrbb_y_min = proposal_xy[:, 1]
    hrbb_x_max = proposal_xy[:, 2]
    hrbb_y_max = proposal_xy[:, 3] 

    h = hw[:, 3]
    w = hw[:, 2]  
    h2 = proposal_wh[:, 3] - h
    w2 = proposal_wh[:, 2] - w

    
    x1 = (hrbb_x_min + w)[:, None]
    y1 = hrbb_y_min[:, None]
    x2 = hrbb_x_max[:, None]
    y2 = (hrbb_y_min + h2)[:, None]
    x3 = (hrbb_x_min + w2)[:, None]
    y3 = hrbb_y_max[:, None]
    x4 = hrbb_x_min[:, None]
    y4 = (hrbb_y_min + h)[:, None]
    
    if dtype == 'tensor':
        obb_bbox = torch.cat([
                x1,y1,x2,y2,x3,y3,x4,y4
            ], axis=1)#.astype(np.int64)
    else:
        obb_bbox = np.concatenate([
                x1,y1,x2,y2,x3,y3,x4,y4
            ], axis=1)#.astype(np.int64)
    
    return obb_bbox    

def dots4ToRec4(poly):
    xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
                            max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
                             min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
                             max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    return xmin, ymin, xmax, ymax

def get_groundtruth(annopath, ids, index, keep_difficult=True):
    img_id = ids[index]
    anno = ET.parse(annopath % img_id).getroot()
    anno = preprocess_annotation(anno, keep_difficult)
    
    return anno

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))
   
def get_best_begin_point(coordinate):
    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1],
                                                                                           dst_coordinate[
                                                                                               1]) + cal_line_length(
            combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
#    if force_flag != 0:
#        print("choose one direction!")
    return  combinate[force_flag]

def batch_cal_line_length(point1, point2):
    math1 = torch.pow(point1[:, 0] - point2[:, 0], 2)
    math2 = torch.pow(point1[:, 1] - point2[:, 1], 2)
    return torch.sqrt(math1 + math2)
 

def batch_get_best_begin_point(coordinate):
    x1 = coordinate[:, 0]
    y1 = coordinate[:, 1]
    x2 = coordinate[:, 2]
    y2 = coordinate[:, 3]
    x3 = coordinate[:, 4]
    y3 = coordinate[:, 5]
    x4 = coordinate[:, 6]
    y4 = coordinate[:, 7]
    xmin = torch.min(torch.stack([x1, x2, x3, x4], dim=1), dim=1)[0]
    ymin = torch.min(torch.stack([y1, y2, y3, y4], dim=1), dim=1)[0]
    xmax = torch.max(torch.stack([x1, x2, x3, x4], dim=1), dim=1)[0]
    ymax = torch.max(torch.stack([y1, y2, y3, y4], dim=1), dim=1)[0]
    combinate = torch.stack([torch.stack([torch.stack([x1, y1], dim=1), 
                                          torch.stack([x2, y2], dim=1), 
                                          torch.stack([x3, y3], dim=1), 
                                          torch.stack([x4, y4], dim=1)], dim=1), 
                             torch.stack([torch.stack([x2, y2], dim=1), 
                                          torch.stack([x3, y3], dim=1), 
                                          torch.stack([x4, y4], dim=1), 
                                          torch.stack([x1, y1], dim=1)], dim=1),
                             torch.stack([torch.stack([x3, y3], dim=1), 
                                          torch.stack([x4, y4], dim=1), 
                                          torch.stack([x1, y1], dim=1), 
                                          torch.stack([x2, y2], dim=1)], dim=1),
                             torch.stack([torch.stack([x4, y4], dim=1), 
                                          torch.stack([x1, y1], dim=1), 
                                          torch.stack([x2, y2], dim=1), 
                                          torch.stack([x3, y3], dim=1)], dim=1)], dim=1)
    dst_coordinate = torch.stack([torch.stack([xmin, ymin], dim=1), 
                                  torch.stack([xmax, ymin], dim=1),
                                  torch.stack([xmax, ymax], dim=1),
                                  torch.stack([xmin, ymax], dim=1),], dim=1)
    force_i = 100000000.0
    force = torch.full((coordinate.size(0),), force_i, 
               dtype=coordinate.dtype, 
               device=coordinate.device)
    force_flag = torch.zeros(coordinate.size(0),
                             dtype=coordinate.dtype,
                             device=coordinate.device)
    combinate_final = dst_coordinate.clone()
    for i in range(4):
        temp_force = batch_cal_line_length(
            combinate[:, i][:, 0], 
            dst_coordinate[:, 0]) + batch_cal_line_length(
                combinate[:, i][:, 1],
                dst_coordinate[:, 1]) + batch_cal_line_length(
                    combinate[:, i][:, 2], 
                    dst_coordinate[:, 2]) + batch_cal_line_length(
                        combinate[:, i][:, 3], 
                        dst_coordinate[:, 3])
                        
        mask = temp_force < force
        force[mask] = temp_force[mask]
        force_flag[mask] = i
        combinate_final[mask] = combinate[mask, i]
        # if temp_force < force:
        #     force = temp_force
        #     force_flag = i
#    if force_flag != 0:
#        print("choose one direction!")
    return  combinate_final

def preprocess_annotation(target):
    polyes = []
    gt_classes = []
    difficult_boxes = []
#    TO_REMOVE = 1
    
    

    for obj in target.iter("object"):
        difficult = int(obj.find("difficult").text) == 1
        name = obj.find("name").text.lower().strip()
        bb = obj.find("bndbox")
        # Make pixel indexes 0-based
        # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
        
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
        
        poly = np.array(point_box)
        poly = poly.reshape(4, 2)
        polyes.append(poly)
        gt_classes.append(name)
        difficult_boxes.append(difficult)
        
        
    size = target.find("size")
    im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

    res = {
        "boxes": polyes,
        "labels": gt_classes,
        "difficult": difficult_boxes,
        "im_info": im_info,
    }
    return res   


