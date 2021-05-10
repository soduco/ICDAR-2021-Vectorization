import argparse
import os

import sys

sys.path.insert(1, '../evaluation/')

from eval_shape_detection import shape_detection

def gen_watershed_image(reconstruction_image_path, dynamic, area, ws_path, out_path):
    os.environ["LD_LIBRARY_PATH"]=os.path.join(os.environ["HOME"], "lib")
    os.system('./histmapseg/build/bin/histmapseg 'f'{reconstruction_image_path}'' 'f'{dynamic}'' 'f'{area}'' 'f'{ws_path}'' 'f'{out_path}')

def evaluate_watershed(label_path, input_contenders_path, input_mask, output_dir, auc_threshold):
    shape_detection(label_path, [input_contenders_path], input_mask, output_dir, auc_threshold)

def main():
    AUC_THRESHOLD_DEFAULT = 0.5
    parser = argparse.ArgumentParser(description='Grid search watershed')
    parser.add_argument('--reconstruction_image_path', type=str, default=None, help='Reconstruction image path')
    parser.add_argument('--ws_path', type=str, default=None, help='ws path.')
    parser.add_argument('--out_path', type=str, default=None, help='output path.')
    parser.add_argument('--dynamic', type=int, default=None, help='Watershed dynamic.')
    parser.add_argument('--area', type=int, default=None, help='Watershed area.')

    parser.add_argument('--label_path', type=str, default=r'/lrde/home2/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-GT_LABELS_target.png', help='The label path')
    parser.add_argument('--input_mask', type=str, default=r'/lrde/home2/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/data_generator/img_gt/BHdV_PL_ATL20Ardt_1898_0004-TEST-MASK_content.png', help='Validation mask to evaluate the results')
    parser.add_argument('--output_dir', type=str, default=None, help='The label path')
    parser.add_argument('--auc-threshold', type=float, help='Threshold value (float) for AUC: 0.5 <= t < 1.'f' Default={AUC_THRESHOLD_DEFAULT}', default=AUC_THRESHOLD_DEFAULT)

    args = parser.parse_args()

    gen_watershed_image(args.reconstruction_image_path, args.dynamic, args.area, args.ws_path, args.out_path)
    evaluate_watershed(args.label_path, args.ws_path, args.input_mask, args.output_dir, args.auc_threshold)


if __name__ == '__main__':
    main()
