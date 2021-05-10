import numpy as np
import torch
from torch.autograd import Variable
import time
import os
from model import *
from loss import *
from datasets.dataset import Data
import argparse
import cfg
from PIL import Image

import sys
sys.path.insert(1, '../../data_generator/')
from reconstruct_tiling import reconstruct_tiling


def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def test(model, args):
    val_root = cfg.config_val[args.dataset]['data_root']
    val_lst = cfg.config_val[args.dataset]['data_lst']
    val_name_lst = os.path.join(val_root, 'val_pair.lst')
    mean_bgr = np.array(cfg.config_val[args.dataset]['mean_bgr'])
    val_img = Data(val_root, val_lst, mean_bgr=mean_bgr)
    valloader = torch.utils.data.DataLoader(val_img, batch_size=1, shuffle=False, num_workers=0)
    nm = np.loadtxt(val_name_lst, dtype=str)
    save_dir = os.path.join(args.res_dir, args.model.split('/')[-1].split('.')[0] + '_fuse')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.cuda:
        model.cuda()

    model.eval()
    start_time = time.time()
    all_t = 0
    for i, (data, label) in enumerate(valloader):
        if args.cuda:
            data = data.cuda()
            data = Variable(data)

            label = label.cuda()
            label = Variable(label)

        tm = time.time()

        with torch.no_grad():
            out = model(data)

        fuse = torch.sigmoid(out[-1]).cpu().numpy()[0, 0, :, :]
        fuse = fuse * 255
        fuse = Image.fromarray(fuse).convert('RGB')
        fuse.save(os.path.join(save_dir, '{}.png'.format(nm[i][0].split('/')[1].split('.')[0])))

        all_t += time.time() - tm

    print('Overall Time use: ', time.time() - start_time)

def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = BDCN()
    model.load_state_dict(torch.load('%s' % (args.model)))

    test(model, args)

    # Reconstruct image
    reconstruct_tile_path = os.path.join(args.res_dir, args.model.split('/')[-1].split('.')[0] + '_fuse')
    reconstruct_tiling(args.original_image_path, reconstruct_tile_path, args.tile_save_image_path, w_size=500)


def parse_args():
    parser = argparse.ArgumentParser('Test BDCN')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_val.keys(), default='HistoricalMap2020', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true', help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default=None, help='the model to test')
    parser.add_argument('--res-dir', type=str, default='results', help='the dir to store result')
    parser.add_argument('--original_image_path', type=str, default='../../data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-INPUT_color_border.jpg', help='the file path of original image')
    parser.add_argument('--tile_save_image_path', type=str, default='../best_models_prediction/BDCN_pretrain_val.png', help='the file path of original image')

    return parser.parse_args()

if __name__ == '__main__':
    main()
