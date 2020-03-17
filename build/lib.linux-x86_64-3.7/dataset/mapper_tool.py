# -*- coding: utf-8 -*-

import numpy as np
import torch
import math
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    RotatedBoxes,
)
from .dataset_tool import polygonToRotRectangle, get_best_begin_point
import cv2 as cv
import random


from fvcore.common.file_io import PathManager
from PIL import Image, ImageOps

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

# --- copy from detectron2----
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

def transform_dota_instance_annotations(annotation, image_size, rota, transforms):
    
    """
    Apply transforms to box, segmentation and keypoints of annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    
    poly = np.array(annotation["boxes"])
    poly = poly.reshape(4, 2)
    image_size_half = np.array(image_size) / 2
    if rota != 0:
        ro = []
        for p in poly:
            ro.append(rotate([int(image_size_half[0]) - 1, 
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
    annotation["boxes"] = transforms.apply_rotated_box(np.array([obb_box]))[0]
       
    return annotation


def dota_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """ 
    
    target = Instances(image_size)
    
    obb_boxes = [obj["boxes"] for obj in annos]
    obb_boxes = target.gt_boxes = RotatedBoxes(obb_boxes)
    # obb_boxes.clip(image_size)
    
    pt_hbb, pt_inbox, polygons = [], [], []
    
    rotate_boxes = obb_boxes.tensor.numpy()
    data = [
        convRotaToPolyAndHbb(rotate_box) for rotate_box in rotate_boxes
    ]
    for d in data:
        pt_hbb.append(d[0])
        pt_inbox.append(d[1])
        polygons.append(d[2])
    
    pt_inbox = torch.as_tensor(pt_inbox).to(dtype=torch.float)
    target.gt_pt_inbox_boxes = Boxes(pt_inbox)
    
    pt_hbb = torch.as_tensor(pt_hbb).to(dtype=torch.float)
    target.gt_pt_hbb_boxes = Boxes(pt_hbb)
    
    # for sigmoid_focal_loss_jit the category id should start with 0
    # for SigmoidFocalLoss in layers the category id should start with 1
    classes = [obj["category_id"] + 1 for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes


    # masks = PolygonMasks(polygons)
    masks_areas = target.gt_pt_hbb_boxes.area()
    # masks = torch.as_tensor(masks.polygons).to(dtype=torch.float)
    # target.gt_poly = masks.view(-1, 8)
    target.gt_areas = masks_areas.to(dtype=torch.float)
    
    # if len(target) > 3000:
    #     mask = random.sample(list(range(0, len(target))), 3000)
    #     target = target[mask]
        
    return target