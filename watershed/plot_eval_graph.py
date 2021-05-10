import argparse
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

import pdb

def full_combination_plot(input_results_dir, output_file_path):
    # Read the results files
    params = os.listdir(input_results_dir)
    fig = plt.figure(figsize=(15, 15))
    sns.set_theme()
    for p in params:
        results_csv_1 = glob.glob(input_results_dir + p + '/' + '*.csv')[0]
        label = str(results_csv_1.split('/')[-1].split('_')[2] + '_' + results_csv_1.split('/')[-1].split('_')[-2])
        
        df_1 = pandas.read_csv(results_csv_1)
        sns_plot = sns.lineplot(data=df_1, x="IoU", y="F-score", label="{}".format(label))
        sns_plot.set(ylim=(0.0, 1))
    plt.savefig(output_file_path)


def sensitivity_plot(input_results_dir, best_dynamic, best_minimum_area, output_file_sen_1, output_file_sen_2):
    # Read the results files
    params = os.listdir(input_results_dir)

    fig = plt.figure(figsize=(15, 15))
    sns.set_theme()
    for p in params:
        dyn_val = int(p.split('_')[2])
        if dyn_val == best_dynamic:
            results_csv_1 = glob.glob(input_results_dir + p + '/' + '*.csv')[0]
            label = str(results_csv_1.split('/')[-1].split('_')[2] + '_' + results_csv_1.split('/')[-1].split('_')[-2])
            df_1 = pandas.read_csv(results_csv_1)
            sns_plot = sns.lineplot(data=df_1, x="IoU", y="F-score", label="{}".format(label))
            sns_plot.set(ylim=(0.0, 1))
    plt.savefig(output_file_sen_1)

    fig = plt.figure(figsize=(15, 15))
    sns.set_theme()
    for p in params:
        area_val = int(p.split('_')[-1])
        if area_val == best_minimum_area:
            results_csv_1 = glob.glob(input_results_dir + p + '/' + '*.csv')[0]
            label = str(results_csv_1.split('/')[-1].split('_')[2] + '_' + results_csv_1.split('/')[-1].split('_')[-2])
            df_1 = pandas.read_csv(results_csv_1)
            sns_plot = sns.lineplot(data=df_1, x="IoU", y="F-score", label="{}".format(label))
            sns_plot.set(ylim=(0.0, 1))
    plt.savefig(output_file_sen_2)

def main():
    parser = argparse.ArgumentParser(description='Plot evaluation graph and visualize the sensitivity.')
    parser.add_argument('--input_results_dir', type=str, default=r'./ws_eval/',
    help='Evalaution results directory.')
    parser.add_argument('--output_file_path', type=str, default=r'./eval_merge/merge_eval_plot.png',
    help='Evaluation merge results.')
    parser.add_argument('--output_file_path_sen_1', type=str, default=r'./eval_merge/dynamic_sensitivty_plot.png',
    help='Evaluation of dynamic plot.')
    parser.add_argument('--output_file_path_sen_2', type=str, default=r'./eval_merge/minimum_area_sensitivty_plot.png',
    help='Evaluation of minimum area sensitivity plot.')
    parser.add_argument('--best_dynamic', type=int, default=0,
    help='Optimimum value of dynamics calculated by grid search.')
    parser.add_argument('--best_minimum_area', type=int, default=50,
    help='Optimimum value of dynamics calculated by grid search.')

    args = parser.parse_args()
    full_combination_plot(args.input_results_dir, args.output_file_path)
    sensitivity_plot(args.input_results_dir, args.best_dynamic, args.best_minimum_area, args.output_file_path_sen_1, args.output_file_path_sen_2)

if __name__ == '__main__':
    main()