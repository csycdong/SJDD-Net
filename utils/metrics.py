import cv2
import math
import numpy as np
from skimage.metrics import structural_similarity


def compute_psnr(clean_img, img):
    """compute the psnr
    """
    clean_img = np.array(clean_img).astype(np.float32)
    img = np.array(img).astype(np.float32)
    mse = np.mean((clean_img - img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_ssim(clean_img, img, data_range=255, multichannel=True):
    return structural_similarity(clean_img, img, data_range=data_range, multichannel=multichannel)
