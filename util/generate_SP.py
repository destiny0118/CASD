import numpy as np
import os
from PIL import Image

SP_input_nc = 8
finesize = [256, 256]


def generate_SP(folder, new_folder):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for name in os.listdir(folder):
        # print(name.split('_')[-1])
        if name.split('_')[-1][:3] == 'vis':
            continue
        old_name = os.path.join(folder, name)
        old_name = old_name[:-4] + '.png'

        new_name = os.path.join(new_folder, name + ".npz")

        SP1_data = Image.open(old_name)
        SP1_data = np.array(SP1_data)

        SP1 = np.zeros((SP_input_nc, finesize[0], finesize[1]), dtype='uint8')
        SP1_20 = np.zeros((20, finesize[0], finesize[1]), dtype='uint8')
        nc = 20
        for id in range(nc):
            SP1_20[id] = (SP1_data == id).astype('uint8')
        SP1[0] = SP1_20[0]
        SP1[1] = SP1_20[9] + SP1_20[12]
        SP1[2] = SP1_20[2] + SP1_20[1]
        SP1[3] = SP1_20[3]
        SP1[4] = SP1_20[13] + SP1_20[4]
        SP1[5] = SP1_20[5] + SP1_20[6] + SP1_20[7] + SP1_20[10] + SP1_20[11]
        SP1[6] = SP1_20[14] + SP1_20[15]
        SP1[7] = SP1_20[8] + SP1_20[16] + SP1_20[17] + SP1_20[18] + SP1_20[19]

        np.savez_compressed(new_name, data=SP1)


if __name__ == '__main__':
    data = np.load('../dataset/fashion/train_SP/fashionMENDenimid0000008001_1front.png.npz')
    print(data.keys())
    a = data['data']
    for key in data.keys():
        print(f"Key: {key}, Data: {data[key]}")
    BP_dir = '../dataset/fashion/semantic_merge3'
    BPD_dir = '../dataset/fashion/train_SP'
    generate_SP(BP_dir, BPD_dir)

BP_dir = 'dataset/fashion/semantic_merge3'
BPD_dir = 'dataset/fashion/train_SP'
generate_SP(BP_dir, BPD_dir)
