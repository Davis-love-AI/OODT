# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
from collections import defaultdict                
def decode_result(classes, image_id, scores, instances, eval_type):
    _predictions = defaultdict(list) 
    rot_boxes = instances.pred_boxes.tensor.numpy()
    for cls, rot_boxe, score in zip(classes, rot_boxes, scores):
        theta = rot_boxe[-1]
            
        if theta > 90.0:
            theta -= 180
        elif theta < -90.0:
            theta += 180
            
        if theta == -90:
            theta = 90
       
        hbb_box = ((rot_boxe[0], rot_boxe[1]),
                   (rot_boxe[2], rot_boxe[3]),
                   theta)
        hbb_box = cv.boxPoints(hbb_box)
        hbb_box = np.int0(hbb_box)
        
        bb_cover = hbb_box.reshape(-1).astype(float)
        
        rot_box = [rot_boxe[0], rot_boxe[1],
                   rot_boxe[2], rot_boxe[3],
                   theta]
        
        x1, y1, x2, y2, x3, y3, x4, y4 = bb_cover
        if eval_type == 'hw':
            # The inverse of data loading logic in `datasets/pascal_voc.py` refer to dota_dateset
            _predictions[cls].append(
                    f"{image_id} {score:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f}"
                )
        if eval_type == 'rot':
            x, y, w, h, a = rot_box
            # The inverse of data loading logic in `datasets/pascal_voc.py`
#            xmin += 1
#            ymin += 1
            _predictions[cls].append(
                    f"{image_id} {score:.3f} {x:.1f} {y:.1f} {w:.1f} {h:.1f} {a:.1f}"
                )
            
    return _predictions