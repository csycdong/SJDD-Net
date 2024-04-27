import numpy as np
import os
import torch


def tfi_torch(spk_seq, device):
    b, n, h, w = spk_seq.shape
    last_index = torch.zeros((b, h, w)).to(device).float()
    cur_index = torch.zeros((b, h, w)).to(device).float()
    c_frames = torch.zeros_like(spk_seq).to(device).float()
    for i in range(n - 1):
        last_index = cur_index
        cur_index = spk_seq[:,i+1,:,:] * (i + 1) + (1 - spk_seq[:,i+1,:,:]) * last_index
        c_frames[:,i,:,:] = cur_index - last_index
    last_frame = c_frames[:,n-1,:,:]
    last_frame[last_frame==0] = n
    c_frames[:,n-1,:,:] = last_frame
    last_interval = n * torch.ones((b, h, w)).to(device).float()
    for i in range(n - 2, -1, -1):
        last_interval = spk_seq[:,i+1,:,:] * c_frames[:,i,:,:] + (1 - spk_seq[:,i+1,:,:]) * last_interval
        tmp_frame = c_frames[:,i,:,:]
        tmp_frame[tmp_frame==0] = last_interval[tmp_frame==0]
        c_frames[:,i,:,:] = tmp_frame
    return 1. / c_frames


def tfp_torch(spk_seq, wsize, device):
    b, n, h, w = spk_seq.shape
    clips = torch.zeros((b, n - wsize + 1, wsize, h, w)).to(device).float()
    for i in range(n - wsize + 1):
        clips[:,i,:,:,:] = spk_seq[:,i:i+wsize,:,:]
    c_frames = torch.mean(clips, axis=2)
    return c_frames


def tfi(spk_seq, gamma):
    n, h, w = spk_seq.shape
    last_index = np.zeros((1, h, w))
    cur_index = np.zeros((1, h, w))
    c_frames = np.zeros_like(spk_seq).astype(np.float64)
    for i in range(n - 1):
        last_index = cur_index
        cur_index = spk_seq[i+1,:,:] * (i + 1) + (1 - spk_seq[i+1,:,:]) * last_index
        c_frames[i,:,:] = cur_index - last_index
    last_frame = c_frames[n-1:,:]
    last_frame[last_frame==0] = n
    c_frames[n-1,:,:] = last_frame
    last_interval = n * np.ones((1, h, w))
    for i in range(n - 2, -1, -1):
        last_interval = spk_seq[i+1,:,:] * c_frames[i,:,:] + (1 - spk_seq[i+1,:,:]) * last_interval
        tmp_frame = np.expand_dims(c_frames[i,:,:], 0)
        tmp_frame[tmp_frame==0] = last_interval[tmp_frame==0]
        c_frames[i] = tmp_frame
    return ((1 / c_frames)**gamma * 255).astype(np.uint8)


def tfp(spk_seq, win_size, gamma):
    half_win = win_size // 2
    n, h, w = spk_seq.shape
    c_frames = np.zeros((n - win_size + 1, h, w)).astype(np.float64)
    for i in range(half_win, n - half_win):
        c_frame = np.mean(spk_seq[i - half_win:i+half_win+1,:,:], axis=0)
        c_frames[i-half_win,:,:] = c_frame
    return (c_frames**gamma * 255).astype(np.uint8)


def interval(spk_seq, offset=0, type=np.uint16):
    n, h, w = spk_seq.shape
    last_index = np.zeros((1, h, w))
    cur_index = np.zeros((1, h, w))
    c_frames = np.zeros_like(spk_seq).astype(np.float64)
    for i in range(n - 1):
        last_index = cur_index
        cur_index = spk_seq[i+1,:,:] * (i + 1) + (1 - spk_seq[i+1,:,:]) * last_index
        c_frames[i,:,:] = cur_index - last_index
    last_frame = c_frames[n-1:,:]
    last_frame[last_frame==0] = n + 1
    c_frames[n-1,:,:] = last_frame
    last_interval = (n + 1) * np.ones((1, h, w))
    for i in range(n - 2, -1, -1):
        last_interval = spk_seq[i+1,:,:] * c_frames[i,:,:] + (1 - spk_seq[i+1,:,:]) * last_interval
        tmp_frame = np.expand_dims(c_frames[i,:,:], 0)
        tmp_frame[tmp_frame==0] = last_interval[tmp_frame==0]
        c_frames[i] = tmp_frame
    c_frames = c_frames + offset
    return c_frames.astype(type)


def raw2spike(raw_seq, h, w, upside_down=False):
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
        spk_seq[img_id, :, :] = np.flipud((result == comparator)) if upside_down else (result == comparator)
    return spk_seq


def spike2raw(spk_seq, save_path, upside_down=False):
    """
        spk_seq: Numpy array (sfn x h x w)
        save_path: full saving path (string)
        Rui Zhao
    """
    sfn, h, w = spk_seq.shape
    base = np.power(2, np.linspace(0, 7, 8))
    if os.path.exists(save_path):
        os.remove(save_path)
    fid = open(save_path, 'wb')
    for img_id in range(sfn):
        # 模拟相机的倒像
        spike = np.flipud(spk_seq[img_id, :, :]) if upside_down else spk_seq[img_id, :, :]
        # numpy按自动按行排，数据也是按行存的
        spike = spike.flatten()
        spike = spike.reshape([int(h*w/8), 8])
        data = spike * base
        data = np.sum(data, axis=1).astype(np.uint8)
        fid.write(data.tobytes())
    fid.close()
    return