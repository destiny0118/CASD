import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True
def resize_dataset(folder, new_folder, new_size=(256, 256), crop_bord=0):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for name in os.listdir(folder):
        if name.endswith(".npz"):
            continue
        old_name = os.path.join(folder, name)
        new_name = os.path.join(new_folder, name)
        # new_name=new_name[:-4]


        img = Image.open(old_name)
        # if img.size[0]!=750 or img.size[1]!=1101:
        #     img=img.resize((750,1101))
        #     img.save(new_name)

        # img = img.convert("RGB")      PNG转为JPG
        w, h =img.size
        if crop_bord == 0:
            pass
        else:
            img = img.crop((crop_bord, 0, w-crop_bord, h))
        img = img.resize(new_size)
        # # img.save(new_name, quality=100)
        img.save(new_name)
        # print('resize %s succefully' % old_name)


old_dir = '../dataset/fashion/test'
root_dir = '../dataset/fashion/test_resize_176_256'

if __name__ == '__main__':


# fashionWOMENTees_Tanksid0000570208_2side
    resize_dataset(old_dir, root_dir, new_size=(176, 256))

