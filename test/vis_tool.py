from ienet import add_ienet_config
from dataset import (
        register_all_dota_voc,
        DatasetMapper, 
        DotaVOCDetectionEvaluator,
        load_dota_instances
    )
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    RotatedBoxes,
)
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import random
import cv2
import math
import numpy as np
from dataset.dataset_tool import polygonToRotRectangle
import torch
import cv2 as cv
import torchgeometry as tgm

from fvcore.common.file_io import PathManager
from PIL import Image, ImageOps

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

def filter_empty_instances(instances, by_box=True, by_mask=True):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_pt_hbb_boxes.nonempty())
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m]  

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    
    angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def read_image(file_name, format=None, rota=0):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
            
        if rota != 0:
            image = image.rotate(rota)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image

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

def hrbb_anchor2hw_obb(proposal, hw):
    hrbb_x_min = proposal[0] - proposal[2]/2
    hrbb_y_min = proposal[1] - proposal[3]/2
    hrbb_x_max = proposal[0] + proposal[2]/2
    hrbb_y_max = proposal[1] + proposal[3]/2  

    h = hw[3]
    w = hw[2]  
    h2 = proposal[3] - h
    w2 = proposal[2] - w
    
    obb_pt_1 = np.array([hrbb_x_min + w, hrbb_y_min])
    obb_pt_2 = np.array([hrbb_x_max, hrbb_y_min + h2])
    obb_pt_3 = np.array([hrbb_x_min + w2, hrbb_y_max])
    obb_pt_4 = np.array([hrbb_x_min, hrbb_y_min + h])
    
    obb_bbox = np.array([
            obb_pt_1,
            obb_pt_2,
            obb_pt_3,
            obb_pt_4
        ], dtype=np.int64)
    
    return obb_bbox
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    
    angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def normTo90(rotate):
    theta = float(rotate[4])
    if theta > 90.0:
        theta -= 180
    elif theta < -90.0:
        theta += 180
        
    if theta == 90.0:
        theta = -90
    rotate[4] = theta
    return rotate

def convRotaToPolyAndHbb(rotate):
    rotate = normTo90(rotate)
    theta = float(-rotate[4])
    obb_bndbox = ((rotate[0], rotate[1]),
               (rotate[2], rotate[3]),
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
    
    hbb_box = get_best_begin_point(hbb_box.reshape(-1, 2))
    polygons = [np.asarray(hbb_box).reshape(-1, 2)]
    polygons = [p.reshape(-1) for p in polygons]
    
    return hrbb_box, pt_inbox, polygons

def transform_dota_instance_annotations(annotation, image_size, rota):

    
    poly = np.array(annotation["boxes"])
    poly = poly.reshape(4, 2)
    image_size_half = image_size / 2
    if rota != 0:
        ro = []
        for p in poly:
            ro.append(rotate(
                    [int(image_size_half[0]) - 1, 
                     int(image_size_half[1]) - 1], p, -rota))
        
        poly = np.array(ro)
    
    OBB_box = polygonToRotRectangle(poly.reshape(-1).astype(np.int64))

    theta = float(OBB_box[4])
    
    theta = math.degrees(theta)

    theta = -theta    
 

    
    obb_box = [
            OBB_box[0], OBB_box[1],
            OBB_box[2], OBB_box[3],
            theta
        ]
    annotation["boxes"] = obb_box
    
    return annotation


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


def dota_annotations_to_instances(annos, image_size):
  
    target = Instances(image_size)
    
    obb_boxes = [obj["boxes"] for obj in annos]
    obb_boxes = target.gt_boxes = RotatedBoxes(obb_boxes)
    obb_boxes.clip(image_size)
    
    pt_hbb, pt_inbox, polygons = [], [], []
    
    rotate_boxes = obb_boxes.tensor.numpy()
    data = [
        convRotaToPolyAndHbb(rotate_box) for rotate_box in rotate_boxes
    ]
    for d in data:
        pt_hbb.append(d[0])
        pt_inbox.append(d[1])
        polygons.append(d[2])
    

    target.gt_pt_inbox_boxes = Boxes(pt_inbox)
    

    target.gt_pt_hbb_boxes = Boxes(pt_hbb)
    

    classes = [obj["category_id"] + 1 for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes


    masks = PolygonMasks(polygons)
    target.gt_masks = masks
    
    if len(target) > 2000:
        mask = random.sample(list(range(0, len(target))), 2000)
        target = target[mask]


    return target