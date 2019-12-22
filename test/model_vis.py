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
import copy
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import random
import cv2
import math
import numpy as np
from projects.Avod.test.vis_tool import (
        polygonToRotRectangle,
        hrbb_anchor2hw_obb,
        rotate,
        transform_dota_instance_annotations,
        dota_annotations_to_instances,
        filter_empty_instances,
        batch_polygonToRotRectangle,
        batch_hbb_hw2poly,
        get_best_begin_point,
        batch_get_best_begin_point,
        read_image
    )

from projects.Avod.test.test_vis_tool import (
        Visualizer,
        _create_text_labels
    )
import cv2 as cv
from detectron2.utils.colormap import random_color
from PIL import Image
import tqdm

cfg = get_cfg()
add_ienet_config(cfg)
# file = 'projects/Avod/configs/ienet_R_50_FPN_s1x_test.yaml'
file = 'projects/Avod/configs/ienet_R_101_FPN_demo_s1x.yaml'
cfg.merge_from_file(file)
model_w = "/home/omnisky/Pengming_workspace/disk_2T/DOTA/training/ienet/avod_center_demo/model_0119999.pth"
cfg.MODEL.WEIGHTS = model_w

# predictor = DefaultPredictor(cfg)
size= "./"
root = cfg.MODEL.AVOD.DATASET
register_all_dota_voc(root)
split = "test"
dir_root = os.path.join(root.format(size), split) 

dataset_dicts = load_dota_instances(dir_root, split)
dota_metadata = MetadataCatalog.get("dota_val_mini_800")

image_shape = np.array([800, 800])
# dataset cheack
datas = []
rota = 0
rotaed_aug = [0, 90, 180, 270]
dataset_dicts_copy = copy.deepcopy(dataset_dicts)
for i in tqdm.tqdm(range(0, len(dataset_dicts_copy))):
    d = dataset_dicts_copy[i]
    # rotaed_aug = [0, 90, 180, 270]
    # rota = random.sample(rotaed_aug, 1)[0]
    rota = 0
    data = d["annotations"] 
    annos = [
            transform_dota_instance_annotations(
                obj, image_shape, rota
            )
                for obj in d["annotations"]
            ]
    instances = dota_annotations_to_instances(
        annos, image_shape
    )
    instances = filter_empty_instances(instances)
    if len(instances) == 0:
        datas.append(i)
    # polygons = torch.from_numpy(
    #                 np.array(instances.gt_masks.polygons)
    #             ).squeeze(1).float()
    
       

i=0
cv2.namedWindow("dota", cv2.WINDOW_NORMAL)
# cv2.namedWindow("dota_point", cv2.WINDOW_NORMAL)
# cv2.namedWindow("dota_poly", cv2.WINDOW_NORMAL)
rotaed_aug = [0, 90, 180, 270]
rota = 0
dataset_dicts_copy = copy.deepcopy(dataset_dicts)
while True:
    d = dataset_dicts_copy[i]
    # img = Image.open(d["file_name"]).convert("RGB")
    im = read_image(d["file_name"], format='BGR', rota=rota)
    
    # rota = random.sample(rotaed_aug, 1)[0]
    # img = img.rotate(rota)
        
    # im = cv.cvtColor(np.asarray(img),cv.COLOR_RGB2BGR) 
    # im_poly = im.copy()
    outputs = predictor(im)
    # annos = [
    #         transform_dota_instance_annotations(
    #             obj, image_shape, rota
    #         )
    #             for obj in d["annotations"]
    #         ]
    # instances = dota_annotations_to_instances(
    #     annos, image_shape
    # )
    
    v = Visualizer(im[:, :, ::-1],
                   metadata=dota_metadata, 
                   scale=1  # remove the colors of unsegmented pixels
    )
    # output = outputs["instances"].to("cpu")
    # polygon = output.pred_masks.numpy()
    # scores = output.scores.numpy()
    # classes = instances.gt_classes.numpy()
    # labels = _create_text_labels(classes, 
    #                                class_names=v.metadata.get("thing_classes", None))
    # pt_inbox_boxes = instances.gt_pt_inbox_boxes.tensor
    # pt_inbox_boxes = torch.from_numpy(pt_inbox_boxes)
    
    # pt_hbb_boxes = instances.gt_pt_hbb_boxes.tensor
    # pt_hbb_boxes = torch.from_numpy(pt_hbb_boxes)
    # pt_hbb_boxes_wh = BoxMode.convert(
    #             pt_hbb_boxes, 
    #             BoxMode.XYXY_ABS, 
    #             BoxMode.XYWH_ABS
    #         )
    # poly = batch_hbb_hw2poly(
    #             pt_hbb_boxes, 
    #             pt_hbb_boxes_wh, 
    #             pt_inbox_boxes,
    #             dtype='tensor',
    #         )
    # polys = poly.view(-1, 8)
    # polys = batch_get_best_begin_point(polys)
    # polys = polys.reshape(-1, 8)
    # rot_box = batch_polygonToRotRectangle(polys)
    # rot_box = rot_box.view(-1, 5)
    # obb = batch_polygonToRotRectangle(pt_hbb_boxes)
    # v = v.draw_instance_predictions(output)
    # v = v.overlay_rotated_instances(instances.gt_boxes.tensor)
    # v = v.overlay_rotated_instances(rot_box, labels)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # num_instances = len(labels)
    # assigned_colors = [random_color(rgb=True, maximum=255) for _ in range(num_instances)]
    # for poly, l, c in zip(polygon, labels, assigned_colors):
    #     poly = poly.reshape(4, 2)
    #     color = tuple(c.tolist())
    #     cv2.putText(im, l, tuple(poly[0]), font, 0.5, color, 2)
    #     cv2.drawContours(im,[poly.astype(np.int64)],0,color,2)
    # cv2.imshow("dota", im)
    mi = v.get_image()[:, :, ::-1]
    
    # for poly in instances.gt_masks.polygons:
    #     # poly = poly.numpy().reshape(4, 2).astype(np.int64)
    #     poly = poly[0].reshape(4, 2).astype(np.int64)
    #     cv2.drawContours(im,[poly.astype(np.int64)],0,(255,255,0),2)
    #     cv2.circle(im,tuple(poly[0]),4,(255,0,0),5)
    #     cv2.circle(im,tuple(poly[1]),4,(255,0,255),5)
    
    # for poly in polys:
    #     poly = poly.numpy().reshape(4, 2).astype(np.int64)
    #     poly = np.array(get_best_begin_point(poly))
    #     cv2.drawContours(im_poly,[poly.astype(np.int64)],0,(255,255,0),2)
    #     cv2.circle(im_poly,tuple(poly[0]),4,(255,0,0),5)
    #     cv2.circle(im_poly,tuple(poly[1]),4,(255,0,255),5)
        
    cv2.imshow("dota", mi)
    # cv2.imshow("dota_point", im)
    # cv2.imshow("dota_poly", im_poly)


    k = cv2.waitKey(0)
    
    if k == ord('n'):
        i+=1
        continue
        #print("image: {}".format(i))
    elif k == ord('m'):
        i-=1
        continue
    elif k == ord('q'):
        cv2.destroyAllWindows()
        print('done')
        break
   
cv2.destroyAllWindows()