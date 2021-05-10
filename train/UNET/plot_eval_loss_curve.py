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
import matplotlib.pyplot as plt

from loss import cross_entropy_loss2d
import pickle
import pdb

def sigmoid(x):
    return 1./(1+np.exp(np.array(-1.*x)))

def test(model, args):
    val_root = cfg.config_val[args.dataset]['data_root']
    val_lst = cfg.config_val[args.dataset]['data_lst']
    mean_bgr = np.array(cfg.config_val[args.dataset]['mean_bgr'])
    val_img = Data(val_root, val_lst, mean_bgr=mean_bgr)
    valloader = torch.utils.data.DataLoader(val_img, batch_size=1, shuffle=False, num_workers=0)

    if args.cuda:
        model.cuda()

    model.eval()
    start_time = time.time()
    all_t = 0
    val_mean_loss = []
    tm = time.time() - start_time

    for val_images, val_labels in valloader:
        if args.cuda:
            val_images, val_labels = val_images.cuda(), val_labels.cuda()
        val_images, val_labels = Variable(val_images), Variable(val_labels)

        with torch.no_grad():
            out = model(val_images)

        val_loss = cross_entropy_loss2d(out, val_labels)
        val_mean_loss.append(val_loss)
        
        all_t += time.time() - tm
    print('Overall Time use: ', time.time() - start_time)
    return torch.mean(torch.stack(val_mean_loss)).detach().cpu().numpy()

def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = UNet(n_channels=args.channels, n_classes=args.classes)

    val_loss_list = []
    weight_list = os.listdir(args.weight_folder)[0: 50]
    for weight_name in weight_list:
        if weight_name.endswith('.pth'):
            # Load the weight
            weight = os.path.join(args.weight_folder, weight_name)
            model.load_state_dict(torch.load('%s' % (weight)))
            # Predict the weight and save prediction in results/
            val_loss = test(model, args)
            print('validation loss {}'.format(val_loss))
            val_loss_list.append(val_loss)
    
    print('The best validation weight is at epoch:{}'.format(np.argmin(val_loss_list)))
    print('The best validation value is at {}'.format(np.min(val_loss_list)))

    with open(os.path.join('validation_eval_plot', 'val_loss_list.txt'), "wb") as fp:
        pickle.dump(val_loss_list, fp)

    length = len(val_loss_list)
    plt.figure(figsize=(10, 10))
    plt.title('Validation plot for every epochs.')
    x = np.linspace(0, length, length, endpoint=False)
    plt.xlabel('Epochs -->', fontsize=18)
    plt.ylabel('Validation loss -->', fontsize=18)
    plt.plot(x, val_loss_list)
    plt.xticks(list(range(1, length, 10)))
    plt.savefig(os.path.join('validation_eval_plot', 'validation_loss_plot.png'))


def parse_args():
    parser = argparse.ArgumentParser('Validation loss plot.')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config_test.keys(),
        default='HistoricalMap2020', help='The dataset to train')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--weight_folder', type=str, default=r'./params/',
        help='the dir of weights')
    parser.add_argument('--side-weight', type=float, default=0.5,
        help='the loss weight of sideout, default 0.5')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--channels', type=int, default=3,
        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=1,
        help='number of classes in the output')
    return parser.parse_args()

if __name__ == '__main__':
    main()
