import gzip
import os
import pickle

from PIL import Image
from PIL import ImageFile
import numpy as np
import util as util
import h5py

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_dataset(folder, new_folder, new_size=(256, 256), crop_bord=0):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for name in os.listdir(folder):
        old_name = os.path.join(folder, name)
        new_name = os.path.join(new_folder, name + ".npz")

        BP1_img = np.load(old_name)
        BPD1_img = util.draw_dis_from_map(BP1_img)[0]
        # BPD1_img_tmp = BPD1_img.astype(np.uint8)

        # print(np.all(BPD1_img_tmp == BPD1_img_tmp))

        np.savez_compressed(new_name, data=BPD1_img)

        print('generate BPD %s succefully' % old_name)

        # with gzip.open(new_name, "wb") as f:
        #     pickle.dump(BPD1_img, f)
        # with gzip.open(new_name, "wt") as f:
        #     np.savetxt(f, BPD1_img)

        # np.save(new_name, BPD1_img)

        # # 保存为 HDF5 格式
        # with h5py.File("data.h5", "w") as f:
        #     f.create_dataset("dataset", data=data, compression="gzip", compression_opts=9)
        #
        # # 加载数据
        # with h5py.File("data.h5", "r") as f:
        #     loaded_data = f["dataset"][:]
        # print(loaded_data.shape)


BP_dir = 'dataset/fashion/trainK'
BPD_dir = 'dataset/fashion/train_BPD'
resize_dataset(BP_dir, BPD_dir)

# if __name__ == '__main__':
#     resize_dataset(BP_dir, BPD_dir)
