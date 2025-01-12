import sys


# DPTN
# fashionMENSweatersid0000217901_4full_2_fashionMENSweatersid0000217901_3back_vis.jpg
# fashionMENSweatersid0000217901_3back.jpg
# Exploring Dual-task Correlation for Pose Guided Person Image Generation
def DPTN_process(imgPath):
    path=imgPath.split('_2_')
    path=path[1]
    path=path.split('_vis')
    path=path[0]+path[1]

    return path


def PT2_process(image):
    image = image.split('to')[1]
    # image=image[:-4]
    parts = image.split('_')
    if (len(parts) == 8):
        image = "fashion" + "".join(parts[0:2]) + "_" + "".join(parts[2:-2]) + "_" + parts[-2] + parts[-1]
    else:
        # fashionWOMENSweaters_id0000778102_2side
        image = "fashion" + "".join(parts[0:-2]) + "_" + parts[-2] + parts[-1]



if __name__ == '__main__':
    DPTN_process("fashionMENSweatersid0000217901_4full_2_fashionMENSweatersid0000217901_3back_vis.jpg")
