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
from border_calibration import border_cali


def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def test(model, args):
    test_root = cfg.config_test[args.dataset]['data_root']
    test_lst = cfg.config_test[args.dataset]['data_lst']
    test_name_lst = os.path.join(test_root, 'test_pair.lst')
    mean_bgr = np.array(cfg.config_test[args.dataset]['mean_bgr'])
    test_img = Data(test_root, test_lst, mean_bgr=mean_bgr)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=1, shuffle=False, num_workers=0)
    nm = np.loadtxt(test_name_lst, dtype=str)
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

        fuse = out[-1].cpu().numpy()[0, 0, :, :]
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

    model = HED()
    model.load_state_dict(torch.load('%s' % (args.model)))

    test(model, args)

    # Reconstruct image
    reconstruct_tile_path = os.path.join(args.res_dir, args.model.split('/')[-1].split('.')[0] + '_fuse')
    reconstruct_tiling(args.original_image_path, reconstruct_tile_path, args.tile_save_image_path, w_size=500)

    # Boarder Calibration
    input_border_path = args.EPM_border
    border_cali(args.tile_save_image_path, input_border_path, args.tile_save_image_path)
    print('Save reconstruction calibration image into {}'.format(args.tile_save_image_path))


def parse_args():
    parser = argparse.ArgumentParser('Test HED')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(), default='HistoricalMap2020', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true', help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default=None, help='the model to test')
    parser.add_argument('--res-dir', type=str, default='results_test', help='the dir to store result')
    parser.add_argument('--original_image_path', type=str, default='../../data_generator/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-INPUT_color_border.jpg', help='the file path of original image')
    parser.add_argument('--tile_save_image_path', type=str, default='../best_models/best_models_prediction/HED_pretrain_test.png', help='the file path of original image')
    parser.add_argument('--EPM_border', type=str, default=r'../../data_generator/epm_mask/BHdV_PL_ATL20Ardt_1898_0004-TEST-EPM-BORDER-MASK_content.png', help='The mask of the EPM_border')

    return parser.parse_args()

if __name__ == '__main__':
    main()
