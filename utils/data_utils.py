import os
from PIL import Image
import numpy as np
import random


def get_paths(root, suffixs):
    paths = []
    for root, _, fs in os.walk(root):
        if root[0] != '.':
            for f in fs:
                for suffix in suffixs:
                    suffix = str.lower(suffix)
                    if f.endswith(suffix):
                        paths.append(os.path.join(root, f))
                        break
    return paths

def read_img_array(filename, to_gray=False):
    img = Image.open(filename).convert('RGB')
    if to_gray:
        img = Image.open(filename).convert('L')
    return np.array(img)

def save_img_array(array, filename, mode='RGB'):
    img = Image.fromarray(np.uint8(array))
    if mode != None:
        img.convert(mode).save(filename)
    else:
        img.save(filename)

def raw2spike(raw_seq, h, w):
    raw_seq = np.array(raw_seq).astype(np.uint8)
    img_size = h*w
    img_num = len(raw_seq)//(img_size//8)
    spk_seq = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0,h*w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id*img_size//8
        id_end = id_start + img_size//8
        cur_info = raw_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        spk_seq[img_id, :, :] = np.flipud((result == comparator))

    return spk_seq
