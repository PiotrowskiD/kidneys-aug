from pathlib import Path

from tqdm import tqdm


from starter_code import utils
from starter_code import visualize

DATAPATH = Path(r'C:\Users\darek\Documents\kits19\data')

if __name__ == '__main__':

    for idx in tqdm(range(210)):
        visualize.visualize(
            idx, r'C:\Users\darek\Documents\kits19\visualization\case_{:05d}\multiclass_mask'.format(idx),
            data_path=DATAPATH,
            separate=True
        )
