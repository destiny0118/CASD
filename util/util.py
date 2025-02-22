from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import os
import collections
from skimage.draw import circle, line_aa



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


LIMB_SEQ = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
            [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
            [0, 15], [15, 17]]

# draw dis img
LIMB_SEQ_DIS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
                [9, 10], [1, 11], [11, 12], [12, 13]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
          'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1


def map_to_cord(pose_map, threshold=0.1):
    all_peaks = [[] for i in range(18)]
    pose_map = pose_map[..., :18]

    if torch.is_tensor(pose_map):
        pose_map = pose_map.cpu()
    try:
        y, x, z = np.where(np.logical_and(pose_map == 1.0, pose_map > threshold))
    except:
        print(np.where(np.logical_and(pose_map == 1.0, pose_map > threshold)))
        print(pose_map.shape)
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(18):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)


def draw_pose_from_map(pose_map, threshold=0.1, **kwargs):
    # CHW -> HCW -> HWC
    pose_map = pose_map[0].cpu().transpose(1, 0).transpose(2, 1).numpy()

    cords = map_to_cord(pose_map, threshold=threshold)
    return draw_pose_from_cords(cords, pose_map.shape[:2], **kwargs)


def draw_dis_from_map(pose_map, threshold=0.1, **kwargs):
    # CHW -> HCW -> HWC
    # print(pose_map.shape)
    if torch.is_tensor(pose_map):
        pose_map = pose_map[0].cpu().transpose(1, 0).transpose(2, 1).numpy()
        # print(pose_map.shape)
    cords = map_to_cord(pose_map, threshold=threshold)
    return draw_dis_from_cords(cords, pose_map.shape[:2], **kwargs)



# draw pose from map
def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3,), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


# point to line distance
def get_distance_from_point_to_line(point, line_point1, line_point2):

    if line_point1 == line_point2:
        point_array = np.array(point)
        point1_array = np.array(line_point1)
        aa = np.expand_dims(np.expand_dims(point1_array, -1), -1)
        aa = np.repeat(aa, point.shape[1], 1)
        aa = np.repeat(aa, point.shape[2], 2)
        return np.linalg.norm(point_array - aa)
    A = line_point2[0] - line_point1[0]
    B = line_point1[1] - line_point2[1]
    C = (line_point1[0] - line_point2[0]) * line_point1[1] + \
        (line_point2[1] - line_point1[1]) * line_point1[0]
    distance = np.abs(A * point[1] + B * point[0] + C) / (np.sqrt(A ** 2 + B ** 2))
    distance = np.exp(-0.1 * distance)
    return distance




# draw dis from map
def draw_dis_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    dis = np.zeros(shape=img_size + (12,), dtype=np.float64)
    y = np.linspace(0, img_size[0] - 1, img_size[0])
    x = np.linspace(0, img_size[1] - 1, img_size[1])
    xv, yv = np.meshgrid(x, y)
    point = np.concatenate([np.expand_dims(yv, 0), np.expand_dims(xv, 0)], 0)

    for i, (f, t) in enumerate(LIMB_SEQ_DIS):
        from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
        to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
        if from_missing or to_missing:
            continue
        dis[:, :, i] = get_distance_from_point_to_line(point, [pose_joints[f][0], pose_joints[f][1]],
                                                       [pose_joints[t][0], pose_joints[t][1]])
    return dis, np.mean(dis, -1)



def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
