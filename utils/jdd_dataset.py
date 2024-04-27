import os
import glob
import random
import numpy as np
from torch.utils import data

import utils.spk_utils as utils


class JDDDataset(data.Dataset):
    def __init__(self, root, noise=0, dataset=None, patch_size=128, color_mode='rggb', upside_down=False):
        self.raws = glob.glob(os.path.join(root, 'spk', str(noise), dataset if dataset else '*', '*.dat'), recursive=True)
        self.gt = glob.glob(os.path.join(root, 'gt', dataset if dataset else '*', '*.npy'), recursive=True)
        self.raws.sort()
        self.gt.sort()
        self.patch_size = patch_size
        self.color_mode = color_mode
        self.noise = noise
        self.upside_down = upside_down

    def __len__(self): 
        return len(self.raws)

    def __getitem__(self, index):
        """
        return : ndarray
            spk: (n, h, w)
            gt: (3, h, w)
            mask: (4, h, w)
        """
        gt = np.load(self.gt[index])
        _, _, h, w = gt.shape
        raw_f = open(self.raws[index], 'rb')
        raw = np.fromstring(raw_f.read(), 'B')
        raw_f.close()
        spk_seq = utils.raw2spike(raw, h, w, self.upside_down)
        mask = self._generate_bayer_mask(h, w)
        spk, gt, mask = self._preprocess_data(spk_seq, gt, mask)
        if self.noise == 'r':
            sigma = self.raws[index].split('-')[-1].replace('.dat', '')
            sigma = float(sigma)
            return spk, gt, mask, sigma
        else:
            return spk, gt, mask

    def _preprocess_data(self, spk_seq, gt, mask):
        if self.patch_size != 0:
            _, _, ih, iw = gt.shape
            x = random.randrange(0, iw - self.patch_size + 1)
            y = random.randrange(0, ih - self.patch_size + 1)
            spk_crop = spk_seq[:, y:y+self.patch_size, x:x + self.patch_size] /1.
            gt_crop = gt[:, :, y:y+self.patch_size, x:x+self.patch_size] / 255.
            mask_crop = mask[:,y:y+self.patch_size, x:x+self.patch_size]
            return self._flip(spk_crop, gt_crop.squeeze(0), mask_crop)
        else:
            spk_crop = spk_seq /1.
            gt_crop = gt / 255.
            mask_crop = mask
            return spk_crop, gt_crop.squeeze(0), mask_crop
        

    def _flip(self, spk, gt, mask):
        if random.random() > 0.5:
            spk = np.flip(spk, axis=1)
            gt = np.flip(gt, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() > 0.5:
            spk = np.flip(spk, axis=2)
            gt = np.flip(gt, axis=2)
            mask = np.flip(mask, axis=2)
        return spk.copy(), gt.copy(), mask.copy()
    
    def _generate_bayer_mask(self, h, w):
        num = []
        flag = 0
        for c in self.color_mode:
            if c == 'r':
                num.append(0)
            elif c == 'g' and flag == 0:
                num.append(1)
                flag = 1
            elif c == 'g' and flag == 1:
                num.append(2)
            elif c == 'b':
                num.append(3)
        mask = np.zeros((4, h, w))
        rows_1 = slice(0, h, 2)
        rows_2 = slice(1, h, 2)
        cols_1 = slice(0, w, 2)
        cols_2 = slice(1, w, 2)
        mask[num[0], rows_1, cols_1] = 1
        mask[num[1], rows_1, cols_2] = 1
        mask[num[2], rows_2, cols_1] = 1
        mask[num[3], rows_2, cols_2] = 1
        return mask


class PreDeomosaicJDDDataset(data.Dataset):
    def __init__(self, root, noise=0, dataset=None, patch_size=128, color_mode='rggb', multi=False):
        self.input = glob.glob(os.path.join(
            root, '3dri-m' if multi else '3dri-s', str(noise), dataset if dataset else '*', '*.npy'), recursive=True)
        self.gt = glob.glob(os.path.join(root, 'gt', dataset if dataset else '*', '*.npy'), recursive=True)
        self.input.sort()
        self.gt.sort()
        self.patch_size = patch_size
        self.color_mode = color_mode
        self.noise = noise

    def __len__(self): 
        return len(self.input)

    def __getitem__(self, index):
        """
        return : ndarray
            input: ({1,n}, 3, h, w)
            gt: (3, h, w)
        """
        input = np.load(self.input[index])
        gt = np.load(self.gt[index])
        input, gt = self._preprocess_data(input, gt)
        if self.noise == 'r':
            sigma = self.input[index].split('-')[-1].replace('.npy', '')
            sigma = float(sigma)
            return input, gt, sigma
        else:
            return input, gt

    def _preprocess_data(self, input, gt):
        if self.patch_size != 0:
            _, ih, iw, _ = input.shape
            x = random.randrange(0, iw - self.patch_size + 1)
            y = random.randrange(0, ih - self.patch_size + 1)
            input_crop = input[:, y:y+self.patch_size, x:x + self.patch_size, :] / 255.
            gt_crop = gt[:, :, y:y+self.patch_size, x:x+self.patch_size] / 255.
            return self._flip(input_crop.transpose((0, 3, 1, 2)), gt_crop.squeeze(0))
        else:
            input_crop = input / 255.
            gt_crop = gt / 255.
            return input_crop.transpose((0, 3, 1, 2)), gt_crop.squeeze(0)
        
    def _flip(self, input, gt):
        if random.random() > 0.5:
            input = np.flip(input, axis=2)
            gt = np.flip(gt, axis=1)
        if random.random() > 0.5:
            input = np.flip(input, axis=3)
            gt = np.flip(gt, axis=2)
        return input.copy(), gt.copy()


class RealJDDDataset(data.Dataset):
    def __init__(self, root, noise=0, dataset=None, patch_size=128, color_mode='bggr', upside_down=True):
        # print(os.path.join(root, 'spk', str(noise), dataset if dataset else '*', '*.dat'))
        # exit()
        self.raws = glob.glob(os.path.join(root, 'spk', str(noise), dataset if dataset else '*', '*.dat'), recursive=True)
        self.raws.sort()
        self.patch_size = patch_size
        self.color_mode = color_mode
        self.noise = noise
        self.upside_down = upside_down

    def __len__(self): 
        return len(self.raws)

    def __getitem__(self, index):
        """
        return : ndarray
            spk: (n, h, w)
            gt: (3, h, w)
            mask: (4, h, w)
        """
        h = 1000
        w = 1000
        raw_f = open(self.raws[index], 'rb')
        raw = np.fromstring(raw_f.read(), 'B')
        raw_f.close()
        spk_seq = utils.raw2spike(raw, h, w, self.upside_down)
        mask = self._generate_bayer_mask(h, w)
        spk, mask = self._preprocess_data(spk_seq, mask)
        return spk, mask

    def _preprocess_data(self, spk_seq, mask):
        spk_crop = spk_seq[:39, :, :] /1.
        mask_crop = mask
        return spk_crop, mask_crop

    def _generate_bayer_mask(self, h, w):
        num = []
        flag = 0
        for c in self.color_mode:
            if c == 'r':
                num.append(0)
            elif c == 'g' and flag == 0:
                num.append(1)
                flag = 1
            elif c == 'g' and flag == 1:
                num.append(2)
            elif c == 'b':
                num.append(3)
        mask = np.zeros((4, h, w))
        rows_1 = slice(0, h, 2)
        rows_2 = slice(1, h, 2)
        cols_1 = slice(0, w, 2)
        cols_2 = slice(1, w, 2)
        mask[num[0], rows_1, cols_1] = 1
        mask[num[1], rows_1, cols_2] = 1
        mask[num[2], rows_2, cols_1] = 1
        mask[num[3], rows_2, cols_2] = 1
        return mask