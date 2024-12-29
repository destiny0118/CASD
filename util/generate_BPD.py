import gzip
import os
import pickle

from PIL import Image
from PIL import ImageFile
import numpy as np
import util as util

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_dataset(folder, new_folder, new_size=(256, 256), crop_bord=0):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for name in os.listdir(folder):
        old_name = os.path.join(folder, name)
        new_name = os.path.join(new_folder, name + ".pkl.gz")

        BP1_img = np.load(old_name)
        BPD1_img = util.draw_dis_from_map(BP1_img)[0]

        with gzip.open(new_name, "wb") as f:
            pickle.dump(BPD1_img, f)
        # with gzip.open(new_name, "wt") as f:
        #     np.savetxt(f, BPD1_img)

        # np.save(new_name, BPD1_img)

        print('generate BPD %s succefully' % old_name)


BP_dir = 'dataset/fashion/trainK'
BPD_dir = 'dataset/fashion/train_BPD'
resize_dataset(BP_dir, BPD_dir)
