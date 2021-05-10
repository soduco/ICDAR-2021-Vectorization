import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import time
import re
import os
import sys
import cv2
from model import *
from loss import *
from datasets.dataset import Data
import argparse
import cfg
from matplotlib import pyplot as plt
from PIL import Image
from skimage import measure
import copy
import pdb


def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))


def test(model, args):
    test_root = cfg.config_test[args.dataset]['data_root']
    test_lst = cfg.config_test[args.dataset]['data_lst']
    test_name_lst = os.path.join(test_root, 'test_pair.lst')
    mean_bgr = np.array(cfg.config_test[args.dataset]['mean_bgr'])
    test_img = Data(test_root, test_lst, mean_bgr=mean_bgr, augment=False)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=1, shuffle=False, num_workers=0)
    nm = np.loadtxt(test_name_lst, dtype=str)

    save_res = True
    save_dir = os.path.join(args.res_dir, args.model.split('/')[-1].split('.')[0] + '_fuse')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.cuda:
        model.cuda()

    model.eval()
    start_time = time.time()
    all_t = 0
    for i, (data, label) in enumerate(testloader):
        if args.cuda:
            data = data.cuda()
            data = Variable(data)

            label = label.cuda()
            label = Variable(label)

        tm = time.time()

        with torch.no_grad():
            out = model(data)

        if not os.path.exists(os.path.join(save_dir, 'fuse')):
            os.mkdir(os.path.join(save_dir, 'fuse'))
        
        # if not os.path.exists(os.path.join(save_dir, 'fuse_prob')):
        #     os.mkdir(os.path.join(save_dir, 'fuse_prob'))

        # fuse_prob = torch.sigmoid(out[0][-1])
        # fuse_prob = fuse_prob.cpu().numpy()[0, 0, :, :]
        # fuse_prob = fuse_prob * 255
        # fuse_prob = fuse_prob.astype(np.uint8)
        # fuse_prob = Image.fromarray(fuse_prob).convert('RGB')
        # fuse_prob.save(os.path.join(save_dir, 'fuse_prob', '{}.png'.format(nm[i][0].split('/')[2].split('.')[0])))

        # for j in range(len(out[0])):
        #     fuse = out[0][j].cpu().numpy()[0, 0, :, :]
        #     fuse = fuse * 255
        #     fuse = Image.fromarray(fuse).convert('RGB')
        #     fuse.save(os.path.join(save_dir, 'fuse', '{}_{}.png'.format(nm[i][0].split('/')[1].split('.')[0], j)))

        fuse = torch.sigmoid(out[1])
        fuse = (fuse > 0.5).type(torch.uint8)

        # # C0 - C7
        # # fuse[0, 0, :, :][:-1, :-1]
        # # fuse[0, 7, :, :][1:, 1:]
        # m = nn.ZeroPad2d(1)
        # c_0 = m(fuse[0, 0, :, :])[1:-1, 1:-1]
        # c_7 = m(fuse[0, 7, :, :])[2:, 2:]
        # fuse_1 = c_0 * c_7

        # # C1 - C6
        # c_1 = m(fuse[0, 1, :, :])[1:-1, 1:-1]
        # c_6 = m(fuse[0, 6, :, :])[2:, 2:]
        # fuse_2 = c_1 * c_6

        # # C2 - C5
        # c_2 = m(fuse[0, 2, :, :])[1:-1, 1:-1]
        # c_5 = m(fuse[0, 5, :, :])[2:, 2:]
        # fuse_3 = c_2 * c_5

        # # C3 - C4
        # c_3 = m(fuse[0, 3, :, :])[1:-1, 1:-1]
        # c_4 = m(fuse[0, 4, :, :])[2:, 2:]
        # fuse_4 = c_3 * c_4

        # fuse = fuse_1 + fuse_2 + fuse_3 + fuse_4

        fuse = torch.sum(fuse, axis=1)
        fuse = (fuse > 0).type(torch.uint8)
        fuse = fuse.cpu().numpy()[0,:, :]
        fuse = fuse * 255
        fuse = Image.fromarray(fuse).convert('RGB')
        fuse.save(os.path.join(save_dir, 'fuse', '{}.png'.format(nm[i][0].split('/')[1].split('.')[0])))

        all_t += time.time() - tm

    print('Overall Time use: ', time.time() - start_time)

def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    bdcn = BDCN()
    unet = UNet(n_channels=args.channels, n_classes=args.classes)
    model = pixel_connect_network(bdcn, unet)
    model.load_state_dict(torch.load('%s' % (args.model)))

    test(model, args)

def parse_args():
    parser = argparse.ArgumentParser('test BDCN')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
        default='HistoricalPatch', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default=None,
        help='the model to test')
    parser.add_argument('--res-dir', type=str, default='results',
        help='the dir to store result')
    parser.add_argument('--channels', type=int, default=1,
        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=8,
        help='number of classes for watershed level')
    return parser.parse_args()

if __name__ == '__main__':
    main()
