import skimage
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.util import view_as_windows
from PIL import Image
import argparse
import pdb

def generate_tiling(image_path, gt_path, save_image_path, save_gt_path, w_size=500):
    # Generate tiling images
    win_size = int(w_size)
    pad_px = win_size // 2

    # Read image
    in_img = np.array(Image.open(image_path))
    img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px), (0,0)], 'edge')
    tiles = view_as_windows(img_pad, (win_size,win_size,3), step=pad_px)
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            tt = tiles[row, col, 0, ...].copy()
            # If you want black boarder, set the value to 25 (Suggest not using balck boarder)
            # bordersize=1005
            # tt[:bordersize,:, 2] = 255
            # tt[-bordersize:,:, 2] = 255
            # tt[:,:bordersize, 2] = 255
            # tt[:,-bordersize:, 2] = 255
            skio.imsave(os.path.join(save_image_path, f"t_r{row:02d}_c{col:02d}.jpg"), tt)

    # Read ground truth
    in_img = np.array(Image.open(gt_path))
    img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px)], 'constant', constant_values=0)
    tiles = view_as_windows(img_pad, (win_size,win_size), step=pad_px)
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            tt = tiles[row, col, ...].copy()
            # If you want black boarder, set the value to 255 (Suggest not using balck boarder)
            # bordersize=100
            # tt[:bordersize,:, 2] = 255
            # tt[-bordersize:,:, 2] = 255
            # tt[:,:bordersize, 2] = 255
            # tt[:,-bordersize:, 2] = 255
            skio.imsave(os.path.join(save_gt_path, f"t_r{row:02d}_c{col:02d}.png"), tt)


def main():
    parser = argparse.ArgumentParser(description='Create Tillings.')
    parser.add_argument('input_path', help='Path of original image.')
    parser.add_argument('gt_path', help='Path of ground truth.')
    parser.add_argument('save_image_path', help='Directory of images for saving the tilings.')
    parser.add_argument('save_gt_path', help='Directory of ground truth for saving the tilings.')
    parser.add_argument('width', help='Width of the tillings')

    args = parser.parse_args()
    generate_tiling(args.input_path, args.gt_path, args.save_image_path, args.save_gt_path, args.width)

if __name__ == '__main__':
    main()
