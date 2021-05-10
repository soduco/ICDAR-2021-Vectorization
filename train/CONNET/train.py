import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import time
import re
import os
from model import *
from loss import *
from datasets.dataset import Data
import cfg
import log
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import pdb

def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))


def train(model, args):
    data_root = cfg.config[args.dataset]['data_root']
    data_lst = cfg.config[args.dataset]['data_lst']
    mean_bgr = np.array(cfg.config[args.dataset]['mean_bgr'])
    crop_size = args.crop_size
    train_img = Data(data_root, data_lst, mean_bgr=mean_bgr, crop_size=crop_size)
    trainloader = torch.utils.data.DataLoader(train_img, batch_size=args.batch_size, shuffle=True, num_workers=20)
    n_train = len(trainloader)
    logger = args.logger

    # Change it to adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    # Validation evaluation
    val_root = cfg.config_val[args.dataset]['data_root']
    val_lst = cfg.config_val[args.dataset]['data_lst']
    mean_bgr = np.array(cfg.config_val[args.dataset]['mean_bgr'])
    val_img = Data(val_root, val_lst, mean_bgr=mean_bgr, crop_size=crop_size)
    valloader = torch.utils.data.DataLoader(val_img, batch_size=args.batch_size, shuffle=False, num_workers=20)

    start_time = time.time()
    if args.cuda:
        model.cuda()

    if args.resume:
        logger.info('resume from %s' % args.resume)
        state = torch.load(args.resume)
        optimizer.load_state_dict(state['solver'])
        model.load_state_dict(state['param'])

    batch_size = args.batch_size

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    epochs = args.epochs
    batch_index = 0
    for epoch in range(epochs):
        model.train()
        mean_loss = []
        mean_bdcn_loss = []
        mean_connect_loss = []

        batch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', leave=False) as pbar:
            for images, labels in trainloader:
                # Set the gradient in the model into 0
                optimizer.zero_grad()

                if batch_index != batch_size:
                    # If batchsize not equal to batch index , calculate the current loss
                    if args.cuda:
                        images, labels = images.cuda(), labels.cuda()
                    images, labels = Variable(images), Variable(labels)
                    out = model(images)
                    loss_pred, connectivity_loss, total_loss = network_loss(out, labels, args)
                    mean_bdcn_loss.append(loss_pred)
                    mean_connect_loss.append(connectivity_loss)

                    # Back calculating loss
                    total_loss.backward()

                    batch_index += 1

                    # Update batch loss
                    batch_loss += total_loss

                if batch_index == batch_size:
                    # If batchsize equal to batch index , backward the loss and update the loss function
                    # Set batch index to 0
                    batch_index = 0

                    # update parameter, gradient descent, back propagation
                    optimizer.step()

                    # Update the pbar
                    pbar.update(images.shape[0])

                    batch_loss = 0
                    mean_loss.append(total_loss)

                # Add loss (batch) value to tqdm
                pbar.set_postfix(**{'loss (batch)': total_loss.item(),
                                    'bdcn_loss': loss_pred.item(),
                                    'con_loss': connectivity_loss.item()
                                    })

        # Adjust learning rate
        if (epoch+1) % args.step_size == 0:
            adjust_learning_rate(optimizer, epoch+1, args.step_size, args.gamma)

        # Save BDCN weights
        if (epoch+1) % args.snapshots == 0:
            torch.save(model.state_dict(), '%s/Connet_%d.pth' % (args.param_dir, epoch+1))
            state = {'step': epoch+1, 'param':model.state_dict(),'solver':optimizer.state_dict()}
            torch.save(state, '%s/Connet_%d.pth.tar' % (args.param_dir, epoch+1))

        tm = time.time() - start_time

        # Use generative model to test samples
        model.eval()
        val_mean_loss = []
        val_mean_bdcn_loss = []
        val_mean_connect_loss = []
        for val_images, val_labels in valloader:
            if args.cuda:
                val_images, val_labels = val_images.cuda(), val_labels.cuda()
            val_images, val_labels = Variable(val_images), Variable(val_labels)

            with torch.no_grad():
                out = model(val_images)

            val_loss_pred, val_connectivity_loss, val_total_loss = network_loss(out, val_labels, args)

            val_mean_loss.append(val_total_loss)
            val_mean_bdcn_loss.append(val_loss_pred)
            val_mean_connect_loss.append(val_connectivity_loss)

        # Add scalar to tensorboard Loss/Train
        writer.add_scalars('Loss/train/val', {
                            'Train loss': torch.mean(torch.stack(mean_loss)),
                            'Validation loss': torch.mean(torch.stack(val_mean_loss))
        }, epoch)

        logger.info('lr: %e, train_loss: %f, val_bdcn_loss: %f, val_connect_loss: %f, validation loss: %f, time using: %f' %
                (optimizer.param_groups[0]['lr'],
                torch.mean(torch.stack(mean_loss)),
                torch.mean(torch.stack(val_mean_bdcn_loss)),
                torch.mean(torch.stack(val_mean_connect_loss)),
                torch.mean(torch.stack(val_mean_loss)),
                tm))
        
        start_time = time.time()


def main():
    args = parse_args()

    # Choose the GPUs
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger = log.get_logger(args.log)
    args.logger = logger
    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)
    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info(cfg.config[args.dataset])
    logger.info('*'*80)

    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)

    torch.manual_seed(50)
    bdcn = BDCN(pretrain=args.pretrain, logger=logger)

    if args.model:
        init_weights = torch.load(args.model)
        bdcn.load_state_dict(init_weights)
        print('-- Load weight to bdcn network {}'.format(args.model))
        print('\n')

    unet = UNet(n_channels=args.channels, n_classes=args.classes)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            torch.nn.init.zeros_(m.bias)
    
    # Xavier initialization
    unet.apply(weights_init)
    logger.info('Kaiming initliazation done.')

    model = pixel_connect_network(bdcn, unet)

    if args.complete_pretrain:
        model.load_state_dict(torch.load(args.complete_pretrain))
    train(model, args)

def parse_args():
    parser = argparse.ArgumentParser(description='Train pixel connectivity network')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
        default='HistoricalMap2020', help='The dataset to train')
    parser.add_argument('--param-dir', type=str, default='params',
        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-6,
        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=None,
        help='whether resume from some, default is None')
    parser.add_argument('--model', type=str, default=None,
        help='Pre-load model')
    parser.add_argument('-p', '--pretrain', type=str, default='../../pretrain_weight/vgg16-397923af.pth',
        help='init net from pretrained model default is None')
    parser.add_argument('--epochs', type=int, default=100,
        help='Epoch to train network, default is 1000')
    parser.add_argument('--max-iter', type=int, default=40000,
        help='max iters to train network, default is 40000')
    parser.add_argument('--iter-size', type=int, default=10,
        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=1,
        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step-size', type=int, default=100,
        help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    parser.add_argument('--batch-size', type=int, default=1,
        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
        help='the size of image to crop, default not crop')
    parser.add_argument('--complete-pretrain', type=str, default=None,
        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--side-weight', type=float, default=0.5,
        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='the decay of learning rate, default 0.1')
    parser.add_argument('--channels', type=int, default=1,
        help='number of channels for unet')
    parser.add_argument('--classes', type=int, default=8,
        help='number of classes for connectivity')
    return parser.parse_args()

if __name__ == '__main__':
    main()