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
DATA_DEST = config.DATA_PATH
paths = glob(config.ORIGINAL_DATA + '/visualization/case_*/*')
print(DATA_DEST)
for img_path in tqdm(paths):

    if not os.path.basename(img_path).startswith("img"):
        continue
    seg_img = cv2.imread(img_path.replace("img", "seg", 1))
    dir_name = str(Path(img_path).parent.as_posix()).split("/")[-1]
    case_no = int(dir_name[-5:])
    out_name = os.path.basename(img_path).replace("img", "img_" + dir_name)
    if 255 not in seg_img and random.random() > 0.2:
        continue

    if case_no < 10:
        shutil.copy(img_path, os.path.join(DATA_DEST, "test/" + out_name))
        shutil.copy(img_path.replace("img", "seg", 1), os.path.join(DATA_DEST, "testannot/" + out_name))
    elif case_no < 20:
        shutil.copy(img_path, os.path.join(DATA_DEST, "val/" + out_name))
        shutil.copy(img_path.replace("img", "seg", 1), os.path.join(DATA_DEST, "valannot/" + out_name))
    else:
        shutil.copy(img_path, os.path.join(DATA_DEST, "train/" + out_name))
        shutil.copy(img_path.replace("img", "seg", 1), os.path.join(DATA_DEST, "trainannot/" + out_name))




