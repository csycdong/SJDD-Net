import argparse
import numpy as np
import os
import time
import torch
from torch.utils import data

from networks.jdd import JDDNet
from utils.jdd_dataset import JDDDataset
from utils.metrics import compute_psnr, compute_ssim
from utils.data_utils import save_img_array


tag = 'SJDD-Net'


def eval(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.noise == -1:
        noise = 'r' 
    elif args.noise == -2:
        noise = 'gp'
    else:
        noise = args.noise
    # check GPU number
    gpu_num = torch.cuda.device_count()
    if args.gpu_num > gpu_num:
        args.gpu_num = gpu_num
        print('GPU number has been ajusted to', gpu_num)
    use_cuda = (args.gpu_num != 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print('Use GPU:')
        for i in range(args.gpu_num):
            print(torch.cuda.get_device_name(i))
    # set paths
    result_path = os.path.join(args.result_path, tag, args.eval_set, str(noise))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # set dataloader
    eval_dataset = JDDDataset(root=os.path.join(args.data_root, 'eval'), dataset=args.eval_set, noise=noise, patch_size=0)
    eval_loader = data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)
    # load the model
    model = JDDNet(n=39)
    if args.gpu_num > 1:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(args.gpu_num)])
    else:
        model = torch.nn.DataParallel(model)
    model.cuda()
    # resume checkpoint from file
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=None if use_cuda  else 'cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state'], strict=False)
            del checkpoint
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    # evaluation
    model.eval()
    psnr = 0.0
    ssim = 0.0
    for test_index, data_batch in enumerate(eval_loader):
        x = data_batch[0]
        gt = data_batch[1]
        mask = data_batch[2]
        padding_size = 8
        row_pad = padding_size - x.shape[2] % padding_size if x.shape[2] % padding_size != 0 else 0
        col_pad = padding_size - x.shape[3] % padding_size if x.shape[3] % padding_size != 0 else 0
        x =  torch.Tensor(np.pad(x, ((0, 0), (0, 0), (0, row_pad), (0, col_pad)), 'edge')).to(device).float() # b n h w
        mask = torch.Tensor(np.pad(mask, ((0, 0), (0, 0), (0, row_pad), (0, col_pad)), 'constant', constant_values=0)).to(device).float() # b 4 h w
        gt = gt.to(device).float() # b n h w 3
        with torch.no_grad():
            output, noise_map = model(x, mask)
            output = output[:,:,:gt.shape[2], :gt.shape[3]]
            noise_map = noise_map[:,:,:gt.shape[2], :gt.shape[3]]
            output = output.clamp(0., 1.).permute(0, 2, 3, 1).squeeze(0).cpu().numpy() *255.0
            gt = gt.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() * 255.0
            tmp_psnr = compute_psnr(gt, output)
            tmp_ssim = compute_ssim(gt, output, data_range=255, multichannel=True)
            psnr += tmp_psnr
            ssim += tmp_ssim
            save_img_array(gt, os.path.join(result_path, '_'.join([str(test_index+1), 'gt.png'])), mode='RGB')
            save_img_array(output, os.path.join(result_path, '_'.join([str(test_index+1), 'output.png'])), mode='RGB')
            print("batch [%d/%d]:" % (test_index + 1, len(eval_loader)), "PSNR =", tmp_psnr, "    SSIM =", tmp_ssim)
    print("PSNR =", psnr / len(eval_loader), "    SSIM =", ssim / len(eval_loader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch SJDD-Net Evaluation")  
    parser.add_argument('--gpu_id', type=str, default='0',  help='the id of GPU.')   
    parser.add_argument('--noise', type=int, default=-1, help='noise level (default: 0)')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='gpu number for training, 0 means cpu training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='num_workers parameter for dataloader (default: 4)')
    parser.add_argument('--pin_memory', nargs='?', type=bool, default=True,
                        help='pin_memory parameter for data_loader (default: True)')
    parser.add_argument('--eval_set', type=str, default='test',
                        help='dataset for evaluation (default: test)')
    parser.add_argument('--data_root', nargs='?', type=str, default='data',
                        help='root path of the dataset (default: data)')
    parser.add_argument('--result_path', nargs='?', type=str, default='results',
                        help='path to save test results (default: results)')
    parser.add_argument('--resume', nargs='?', type=str, default='weights/r/best_model.pth',
                         help='path to previous saved model to restart from')
    args = parser.parse_args()
    eval(args)
