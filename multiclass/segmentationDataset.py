from pathlib import Path
from typing import List
<<<<<<< Updated upstream

import cv2
from torch.utils.data import Dataset
from catalyst import utils
import numpy as np

=======
import numpy as np
from torch.utils.data import Dataset
from catalyst import utils
import cv2
>>>>>>> Stashed changes

class SegmentationDataset(Dataset):
    def __init__(
            self,
            images: List[Path],
            masks: List[Path] = None,
            transforms=None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:


        ####
        result = {}
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if np.shape(image)[1] != 512:
            image = cv2.resize(image, (512, 512))


        if self.masks is not None:
            mask = cv2.imread(self.masks[idx])
            if np.shape(mask)[1] != 512:
                mask = cv2.resize(mask, (512, 512))
            # extract certain classes from mask (e.g. cars)
            mask_kidney = mask[:, :, 0] == 255
            mask_tumor = mask[:, :, 1] == 255
            masks = [mask_kidney, mask_tumor]
            mask = np.stack(masks, axis=-1).astype('float')
            result["mask"] = mask

        result["image"] = image
        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result
