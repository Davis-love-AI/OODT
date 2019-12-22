import glob
import os
import numpy as np
from collections import defaultdict
txt_per_class = glob.glob("*.txt")

txt_per_image = defaultdict(list)
idd = []
for txt in txt_per_class:
    class_name = txt.split('/')[-1].split('.')[0].split('Task1_')[1]
    f = open(txt, 'r')
    lines = f.readlines()
    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 8)
    for i, bb in enumerate(BB):
        x1, y1, x2, y2, x3, y3, x4, y4 = bb
        txt_per_image[image_ids[i]].append(
            f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f} {class_name} 0"
        )


images = list(txt_per_image.keys())
out = 'image_label/'
os.mkdir(out)      
for image in images:
    lines = txt_per_image.get(image, [""])
    with open(out+image+'.txt', 'w') as f:
        f.write("\n".join(lines))

