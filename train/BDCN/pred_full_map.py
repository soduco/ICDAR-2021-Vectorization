import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import time
import os
from model import *
from loss import *
from datasets.dataset import Data
import argparse
import cfg
from PIL import Image
import matplotlib.pyplot as plt
import pdb
import pickle

import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../data_generator/')
sys.path.insert(1, '../../evaluation/')

from create_tiling import generate_tiling
from reconstruct_tiling import reconstruct_tiling
from border_calibration import border_cali
from prepare_training_text import create_testing_list_file


def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def test(model, weight, save_path, args):
    list_file_name = 'test_pair.lst'
    test_name_lst = os.path.join(save_path, list_file_name)
    mean_bgr = np.array(cfg.config_test[args.dataset]['mean_bgr'])
    test_img = Data(save_path, list_file_name, mean_bgr=mean_bgr)
    testloader = torch.utils.data.DataLoader(test_img, batch_size=1, shuffle=False, num_workers=0)
    nm = np.loadtxt(test_name_lst, dtype=str)

    save_dir = os.path.join(save_path, weight.split('/')[-1].split('.')[0] + '_fuse')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.cuda:
        model.cuda()

    model.eval()
    start_time = time.time()
    all_t = 0
    for i, (data, _) in enumerate(testloader):
        if args.cuda:
            data = data.cuda()
            data = Variable(data)

        tm = time.time()

        with torch.no_grad():
            out = model(data)

        if not os.path.exists(os.path.join(save_dir)):
            os.mkdir(os.path.join(save_dir))

        fuse = out[-1].cpu().numpy()[0, 0, :, :]
        fuse = fuse * 255
        fuse = Image.fromarray(fuse).convert('RGB')
        fuse.save(os.path.join(save_dir, '{}.png'.format(nm[i].split('/')[2].split('.')[0])))
        all_t += time.time() - tm
    print('Overall Time use: ', time.time() - start_time)


def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = BDCN()
    
    # Crop image into batches
    original_image_path = args.original_image_path
    image_root_path = os.path.dirname(original_image_path)
    tile_path = os.path.join(image_root_path, 'data_tiles')

    if not os.path.exists(os.path.join(tile_path)):
        os.mkdir(os.path.join(tile_path))

    # create data tiles
    generate_tiling(original_image_path, tile_path, args.batch_image_size)

    # Create list files
    create_testing_list_file(tile_path)
    
    # Create save path of the construction image path
    tile_save_path = os.path.join(image_root_path, 'tile_save')
    if not os.path.exists(tile_save_path):
        os.mkdir(tile_save_path)

    weight = args.model
    model.load_state_dict(torch.load('%s' % (weight)))

    # Predict the weight and save prediction in results/
    test(model, weight, image_root_path, args)

    # reconstruction tillings
    weight_name_epoch = weight.split('/')[-1].split('.')[0]

    tile_save_image_path = os.path.join(tile_save_path, weight_name_epoch+'_reconstruct.png')

    reconstruct_tile_path = os.path.join('.', image_root_path, weight_name_epoch+'_fuse')
    if not os.path.exists(reconstruct_tile_path):
        os.mkdir(reconstruct_tile_path)

    # Reconstruct the batches into full image
    reconstruct_tiling(original_image_path, reconstruct_tile_path, tile_save_image_path)

    # Boarder Calibration
    if args.EPM_border:
        input_EPM_path = tile_save_image_path
        input_border_path = args.input_border_path
        border_cali(input_EPM_path, input_border_path, tile_save_image_path)
        print('Done border calibration.')

    print('Save reconstruction image into {}'.format(tile_save_image_path))


def parse_args():
    parser = argparse.ArgumentParser('Predict and reconstruct full map.')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
        default='HistoricalMap2020', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('-m', '--model', type=str, default=None,
        help='the model to test')
    parser.add_argument('--original_image_path', type=str, default=None,
        help='the dir to store result /lrde/home2/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/benchmark/HED/predict_new_image/1926_0004/map_avg_fill.png')
    parser.add_argument('--EPM_border', type=str, default=None,
        help='The mask of the data generator ../../data_generator/all_the_mask/input_mask_fix_epm.png')
    parser.add_argument('--batch_image_size', type=int, default=500,
        help='The size of the image.')

    return parser.parse_args()

if __name__ == '__main__':
    main()
