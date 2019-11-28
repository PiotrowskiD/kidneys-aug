import os
import random
from glob import glob
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm

DATA_DEST = '../data'
paths = glob('../visualization/case_*/*')
paths = [Path(x) for x in paths]


for path in tqdm(paths):
    for img_path in os.listdir(path):
        if not img_path.startswith("img"):
            continue
        img_path = os.path.join(path, img_path)
        seg_img = cv2.imread(img_path.replace("img", "seg", 1), cv2.IMREAD_GRAYSCALE)
        out_name = os.path.basename(img_path).replace("img", "img_" + str(img_path.split("\\")[6]))
        if 255 not in seg_img:
            if random.random() < 0.3:
                rnd = random.random()
                if rnd < 0.1:
                    shutil.copy(img_path, os.path.join(DATA_DEST, "test/" + out_name))
                    shutil.copy(img_path.replace("img", "seg", 1), os.path.join(DATA_DEST, "testannot/" + out_name))
                elif rnd < 0.2:
                    shutil.copy(img_path, os.path.join(DATA_DEST, "val/" + out_name))
                    shutil.copy(img_path.replace("img", "seg", 1), os.path.join(DATA_DEST, "valannot/" + out_name))
                else:
                    shutil.copy(img_path, os.path.join(DATA_DEST, "train/" + out_name))
                    shutil.copy(img_path.replace("img", "seg", 1), os.path.join(DATA_DEST, "trainannot/" + out_name))

        else:
            rnd = random.random()
            if rnd < 0.1:
                shutil.copy(img_path, os.path.join(DATA_DEST, "test/" + out_name))
                shutil.copy(img_path.replace("img", "seg", 1), os.path.join(DATA_DEST, "testannot/" + out_name))
            elif rnd < 0.2:
                shutil.copy(img_path, os.path.join(DATA_DEST, "val/" + out_name))
                shutil.copy(img_path.replace("img", "seg", 1), os.path.join(DATA_DEST, "valannot/" + out_name))
            else:
                shutil.copy(img_path, os.path.join(DATA_DEST, "train/" + out_name))
                shutil.copy(img_path.replace("img", "seg", 1), os.path.join(DATA_DEST, "trainannot/" + out_name))


