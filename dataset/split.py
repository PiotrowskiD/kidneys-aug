import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
from glob import glob
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm

from configs import config
DATA_DEST = config.DATA_PATH + "2"
paths = glob(config.ORIGINAL_DATA + '/visualization2/case_*/*')

for img_path in tqdm(paths):

    if not os.path.basename(img_path).startswith("img"):
        continue
    seg_img = cv2.imread(img_path.replace("img", "seg", 1))
    dir_name = str(Path(img_path).parent.as_posix()).split("/")[-1]
    out_name = os.path.basename(img_path).replace("img", "img_" + dir_name)
    if 255 not in seg_img:
        if random.random() < 0.1:
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


