# -*- coding: utf-8 -*-

from DOTA_devkit.DOTA import DOTA

root = '/home/omnisky/Pengming_workspace/disk_2T/DOTA/DOTA_dataset/train'

dataset = DOTA(root)

cats = ['soccer-ball-field', 'roundabout', 'bridge', 
       'ground-track-field', 'basketball-court', 
       'harbor', 'helicopter']

imgs = []
for cat in cats:
    imgids = dataset.getImgIds(catNms=[cat])
    imgs.append(imgids)
    
ids = []    
for img in imgs:
    for i in img:
        if i not in ids:
            ids.append(i)
            
import os
import shutil  
import glob

all_train_img = glob.glob(root + '/images/*.png')
all_train_txt = glob.glob(root + '/labelTxt/*.txt')
det_lot = '/home/omnisky/Pengming_workspace/disk_2T/DOTA/DOTA_dataset/val_mini_lot'
det_litter = '/home/omnisky/Pengming_workspace/disk_2T/DOTA/DOTA_dataset/val_mini_litter'

if not os.path.exists(det_lot+'/images'):
    os.mkdir(det_lot+'/images')

if not os.path.exists(det_lot+'/labelTxt'):
    os.mkdir(det_lot+'/labelTxt')
    
if not os.path.exists(det_litter+'/images'):
    os.mkdir(det_litter+'/images')
    
if not os.path.exists(det_litter+'/labelTxt'):
    os.mkdir(det_litter+'/labelTxt')
    
    
for j, file in enumerate(all_train_img):
    name = file.split('/')[-1].split('.')[0]
    if name not in ids:
        shutil.copyfile(all_train_img[j], det_lot+'/images/'+name+'.png')
        shutil.copyfile(all_train_txt[j], det_lot+'/labelTxt/'+name+'.txt')
    else:
        shutil.copyfile(all_train_img[j], det_litter+'/images/'+name+'.png')
        shutil.copyfile(all_train_txt[j], det_litter+'/labelTxt/'+name+'.txt')
        
        
        
    