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
import pickle
from plot_curve import plot_graph

import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../data_generator/')
sys.path.insert(1, '../../evaluation/')

from reconstruct_tiling import reconstruct_tiling
from border_calibration import border_cali
from epm2labelmap import epm2labelmap
from eval_shape_detection import shape_detection

import pdb

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def test(model, weight, args):
    train_root = cfg.config[args.dataset]['data_root']
    train_lst = cfg.config[args.dataset]['data_lst']
    train_name_lst = os.path.join(train_root, 'train_pair.lst')
    mean_bgr = np.array(cfg.config[args.dataset]['mean_bgr'])
    train_img = Data(train_root, train_lst, mean_bgr=mean_bgr)
    trainloader = torch.utils.data.DataLoader(train_img, batch_size=1, shuffle=False, num_workers=0)
    nm = np.loadtxt(train_name_lst, dtype=str)
    save_dir = os.path.join(args.res_dir, weight.split('/')[-1].split('.')[0] + '_fuse')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.cuda:
        model.cuda()

    model.eval()
    start_time = time.time()
    all_t = 0
    for i, (data, label) in enumerate(trainloader):
        if args.cuda:
            data = data.cuda()
            data = Variable(data)

            label = label.cuda()
            label = Variable(label)

        tm = time.time()

        with torch.no_grad():
            out = model(data)

        if not os.path.exists(os.path.join(save_dir)):
            os.mkdir(os.path.join(save_dir))

        fuse = torch.sigmoid(out).cpu().numpy()[0, 0, :, :]
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

    model = UNet(n_channels=args.channels, n_classes=args.classes)
    original_image_path = args.original_image_path

    # Create save path of the construction image path
    tile_save_path = os.path.join(args.res_dir, 'tile_save')
    if not os.path.exists(tile_save_path):
        os.mkdir(tile_save_path)

    # Create Evaluation results folder
    eval_save_path = os.path.join(args.eval_dir, 'UNET_scratch_results')
    if not os.path.exists(eval_save_path):
        os.mkdir(eval_save_path)

    Precision_val = []
    Recall_val = []
    F_score_val = []
    COCO_score_val = []

    plot_dict = {'precision': None,
                 'recall': None,
                 'F-score': None,
                 'COCO_PQ': None
                 }

    weight_list = os.listdir(args.weight_folder)[30: 120] # 15 epochs -> 60 epochs
    max_coco_pq_score = 0

    for weight_name in weight_list:
        if weight_name.endswith('.pth'):
            # Load the weight
            weight = os.path.join(args.weight_folder, weight_name)
            model.load_state_dict(torch.load('%s' % (weight)))
            # Predict the weight and save prediction in results/
            test(model, weight, args)

            # reconstruction tillings
            weight_name_epoch = weight_name.split('.')[0]
            tile_path = os.path.join('.', args.res_dir, weight_name_epoch+'_fuse')

            if not os.path.exists(tile_path):
                os.makedirs(tile_path)

            tile_save_path = os.path.join(args.res_dir, 'tile_save')

            if not os.path.exists(tile_save_path):
                os.makedirs(tile_save_path)

            tile_save_image_path = os.path.join('.', tile_save_path, weight_name_epoch+'_reconstruct.png')

            # Reconstruct the batches into full image
            reconstruct_tiling(original_image_path, tile_path, tile_save_image_path, w_size=500)

            # # Boarder Calibration
            input_EPM_path = tile_save_image_path
            input_border_path = args.EPM_border
            border_cali(input_EPM_path, input_border_path, tile_save_image_path)
            print('Save reconstruction calibration image into {}'.format(tile_save_image_path))

            # Evaluation process
            input_path = tile_save_image_path
            output_path = str(os.path.join(eval_save_path, weight_name_epoch))

            # Create folder of the evaluation /results/
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_path = os.path.join(output_path, 'epm_label_map.tif')

            # Transfer epm to label map
            epm2labelmap(input_path, output_path, args.EPM_threshold, debug_labels=None)

            # Calculate Precision, recall and F-measure
            input_gt_path = args.label_path
            input_contenders_path = [output_path]
            input_mask = args.validation_mask
            output_dir = str(os.path.join(eval_save_path, weight_name_epoch))
            auc_threshold = args.auc_threshold

            precisions, recalls, f_score, coco_matrix = shape_detection(input_gt_path, input_contenders_path, input_mask, output_dir, auc_threshold)
            coco_gp = coco_matrix[list(coco_matrix.keys())[0]]['PQ']
            Precision_val.append(np.mean(precisions))
            Recall_val.append(np.mean(recalls))
            F_score_val.append(np.mean(f_score))
            COCO_score_val.append(coco_gp)
            
            # Measure the current
            current = coco_gp
            if current >= max_coco_pq_score:
                max_coco_pq_score = current
                current_best_weight = weight_name_epoch
            else:
                pass

            print('Current best average COCO_PQ results: {}'.format(max_coco_pq_score))
            print('Current best weight: {}'.format(current_best_weight))

    plot_dict['precision'] = Precision_val
    plot_dict['recall'] = Recall_val
    plot_dict['F-score'] = F_score_val
    plot_dict['COCO_PQ'] = COCO_score_val

    plot_dict_path = os.path.join(eval_save_path, 'UNET_scratch_plot_info.pkl')
    with open(plot_dict_path, 'wb') as f:
        pickle.dump(plot_dict, f, pickle.HIGHEST_PROTOCOL)

    plot_graph(plot_dict_path, eval_save_path)


def parse_args():
    AUC_THRESHOLD_DEFAULT = 0.5
    parser = argparse.ArgumentParser('Evaluation plot for UNET.')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
        default='HistoricalMap2020', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--res_dir', type=str, default='results_train',
        help='the dir to store result')
    parser.add_argument('--original_image_path', type=str, default=r'../../data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-TRAIN-INPUT_color_border.jpg',
        help='the dir to store result')
    parser.add_argument('--weight_folder', type=str, default=r'./params/',
        help='the dir of weights')
    parser.add_argument('--EPM_border', type=str, default=r'../../data_generator/epm_mask/BHdV_PL_ATL20Ardt_1926_0004-TRAIN-EPM-BORDER-MASK_content.png',
        help='The mask of the EPM_border')
    parser.add_argument('--EPM_threshold', type=int, default=0.5,
        help='Threshold to create binary image of EPM')
    parser.add_argument('--eval_dir', type=str, default=r'./results_train/evaluation',
        help='The evaluation folder')
    parser.add_argument('--label_path', type=str, default=r'../../data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-TRAIN-GT_LABELS_target.png',
        help='The evaluation folder')
    parser.add_argument('--validation_mask', type=str, default=r'../../data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-TRAIN-MASK_content.png',
        help='Validation mask to evaluate the results')
    parser.add_argument('--auc-threshold', type=float,
        help='Threshold value (float) for AUC: 0.5 <= t < 1.'f' Default={AUC_THRESHOLD_DEFAULT}', default=AUC_THRESHOLD_DEFAULT)
    parser.add_argument('--channels', type=int, default=3,
        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
        help='number of classes in the output')

    return parser.parse_args()

if __name__ == '__main__':
    main()
