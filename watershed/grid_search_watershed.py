import subprocess
import argparse
import os
import pathos.pools as pp
from functools import partial
import glob
import numpy as np

import sys

sys.path.insert(1, '../evaluation/')
from eval_shape_detection import shape_detection

import pdb

def prepare_watershed_eval(watershed_base_dir, reconstruction_image_path, label_path, input_mask, auc_threshold):
    # Apply waterhsed by using subprocess
    # subprocess.run('ls')
    model_name = str(reconstruction_image_path.split('/')[-1].split('.')[0].split('_')[0] + '_' + reconstruction_image_path.split('/')[-1].split('.')[0].split('_')[1])

    # Create directory of model watershed results
    watershed_base_dir = os.path.join(watershed_base_dir, model_name)
    if not os.path.exists(watershed_base_dir):
        os.mkdir(watershed_base_dir)

    # Create test ws for watershed
    watershed_ws_dir = os.path.join(watershed_base_dir, 'test_ws')
    if not os.path.exists(watershed_ws_dir):
        os.mkdir(watershed_ws_dir)
    
    # Create test_out for saving watershed output
    watershed_out_dir = os.path.join(watershed_base_dir, 'test_out')
    if not os.path.exists(watershed_out_dir):
        os.mkdir(watershed_out_dir)

    os.environ["LD_LIBRARY_PATH"]=os.path.join(os.environ["HOME"], "lib")   

    os.system('parallel -j 20 ./histmapseg/build/bin/histmapseg 'f'{reconstruction_image_path}'' {1} {2} 'f'{watershed_ws_dir}''/labelmap-ws-dyn_min_{1}_area_min_{2}.tif 'f'{watershed_out_dir}''/labelmap-output-dyn_min_{1}_area_min_{2}.tif ::: $(seq 0 10) ::: 50 100 200 300 400 500')

    # Read and evaluate the watershed output
    w_lst = os.listdir(watershed_ws_dir)

    # Pack the value
    output_base_dir = os.path.join(watershed_base_dir, 'eval_ws')
    if not os.path.exists(output_base_dir):
        os.mkdir(output_base_dir)

    pack = watershed_ws_dir, label_path, input_mask, output_base_dir, auc_threshold 

    def shape_detection_multipro(pack, watershed_file):
        watershed_dir, label_path, input_mask, output_base_dir, auc_threshold = pack
        
        input_contenders_path = os.path.join(watershed_dir, watershed_file)
        print(input_contenders_path)
        output_dir = os.path.join(output_base_dir, input_contenders_path.split('/')[-1].split('.')[0])
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        shape_detection(label_path, [input_contenders_path], input_mask, output_dir, auc_threshold)

    pool = pp.ProcessPool(processes=30) # Pool
    partial_func = partial(shape_detection_multipro, pack)
    results = pool.amap(partial_func, w_lst) # do an asynchronous map, then get the results
    results = results.get()


def grid_search(input_results_dir):
    results = []
    # Read the results files
    params = os.listdir(input_results_dir)
    for p in params:
        s_params = p.split('_')
        file_name = glob.glob(input_results_dir + p + '/' + '*.txt')[0]
        with open(file_name) as f:
            content = f.readlines()
        auc = content[3].split(' ')[4]
        if len(s_params) == 1:
            results.append((s_params[2], auc))
        else:
            results.append(np.array((s_params[2], s_params[5], auc)))

    results = np.array(results)
    if len(s_params) == 1:
        optim_params = results[np.argmax(results[:,1])]
        print('Optimum solution after grid search: threshold {} -> F1: {}'.format(optim_params[2], optim_params[1]))
    else:
        optim_params = results[np.argmax(results[:,2])]
        worst_params = results[np.argmin(results[:,2])]
        print('Optimum solution after grid search: dynamic minimum {} and area minimum {} -> COCO PQ: {}'.format(optim_params[0], optim_params[1], optim_params[2]))
        print('Worst value after grid search: dynamic minimum {} and area minimum {} -> COCO PQ: {}'.format(worst_params[0], worst_params[1], worst_params[2]))
    
def main():
    AUC_THRESHOLD_DEFAULT = 0.5
    parser = argparse.ArgumentParser(description='Grid search watershed')
    parser.add_argument('--watershed_dir', type=str, default=r'./output/',
        help='watershed_results.')
    parser.add_argument('--reconstruction_image_path', type=str, default=r'../benchmark/UNET/results/tile_save/unet_30_reconstruct.png', 
        help='Reconstruction image path')
    parser.add_argument('--label_path', type=str, default=r'../data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-GT_LABELS_target.png',
        help='The label path')
    parser.add_argument('--validation_mask', type=str, default=r'../data_generator/img_gt/BHdV_PL_ATL20Ardt_1926_0004-VAL-MASK_content.png',
        help='Validation mask to evaluate the results')
    parser.add_argument('--auc-threshold', type=float, 
        help='Threshold value (float) for AUC: 0.5 <= t < 1.'f' Default={AUC_THRESHOLD_DEFAULT}', default=AUC_THRESHOLD_DEFAULT)
    parser.add_argument('--input_results_dir', type=str, default=r'./output/UNET/eval_ws/',
        help='Evalaution results directory')

    args = parser.parse_args()
    prepare_watershed_eval(args.watershed_dir, args.reconstruction_image_path, args.label_path, args.validation_mask, args.auc_threshold)

    if not os.path.exists(args.input_results_dir):
        os.makedirs(args.input_results_dir)

    grid_search(args.input_results_dir)

if __name__ == '__main__':
    main()
