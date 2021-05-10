from PIL import Image
from skimage.segmentation import find_boundaries
import numpy as np
import matplotlib.pyplot as plt
import argparse

import pdb

def epm_border_gen(mask_input, save_boundary_path):
    msk_img = np.array(Image.open(mask_input))
    boundary_img = find_boundaries(msk_img, mode='inner').astype(np.uint8)*255
    b_img = Image.fromarray(boundary_img)
    b_img.save(save_boundary_path)


def main():
    parser = argparse.ArgumentParser(description='Border Calibration.')
    parser.add_argument('input_mask_path', help='The path of input mask.')
    parser.add_argument('save_boundary_path', help='Saving path of output boundary.')

    args = parser.parse_args()
    epm_border_gen(args.input_mask_path, args.save_boundary_path)

if __name__ == '__main__':
    main()

    # mask = '/lrde/home2/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-MASK_content.png'
    # epm_border = epm_border_gen(mask)
