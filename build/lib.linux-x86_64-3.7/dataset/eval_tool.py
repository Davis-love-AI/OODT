# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from collections import defaultdict   
from .point_tool import polygonToRotRectangle, hrbb2hw_obb  
import math           
def decode_result(_predictions, _predictions_om, classes, image_id, scores, instances, eval_type):
    boxes = instances.pred_boxes.tensor.numpy()
    for cls, rot_boxe, score in zip(classes, boxes, scores):

        
       
        hbb_box = ((rot_boxe[0], rot_boxe[1]),
                   (rot_boxe[2], rot_boxe[3]),
                   -rot_boxe[4])
        hbb_box = cv.boxPoints(hbb_box)
        hbb_box = np.int0(hbb_box)
        
        pt_x_y_min = hbb_box.min(axis= 0)
        pt_x_y_max = hbb_box.max(axis= 0)
    
        hrbb_box = np.hstack((pt_x_y_min, pt_x_y_max))
        
        poly = hbb_box.reshape(-1).astype(float)
        
        rot_box = [rot_boxe[0], rot_boxe[1],
                   rot_boxe[2], rot_boxe[3],
                   rot_boxe[4]]
        
        x1, y1, x2, y2, x3, y3, x4, y4 = poly
        xmin, ymin, xmax, ymax = hrbb_box
        if eval_type == 'hw':
            # The inverse of data loading logic in `datasets/pascal_voc.py` refer to dota_dateset
            _predictions[cls].append(
                    f"{image_id} {score:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f}"
                )
            fake_id = image_id.split('_')[0]
            _predictions_om[cls].append(
                    f"{fake_id} {score:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f}"
                )
        if eval_type == 'rot':
            x, y, w, h, a = rot_box
            # The inverse of data loading logic in `datasets/pascal_voc.py`
#            xmin += 1
#            ymin += 1
            _predictions[cls].append(
                    f"{image_id} {score:.3f} {x:.1f} {y:.1f} {w:.1f} {h:.1f} {a:.1f}"
                )
        if eval_type == 'hbb':
            
            _predictions[cls].append(
                f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
            )
            
