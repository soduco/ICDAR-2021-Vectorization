import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def plot_graph(input_path, output_path):
    with open(input_path, 'rb') as f:
        plot_dict = pickle.load(f)

    # Plot precision, recall and F-measure score
    length = len(plot_dict['precision'])

    # plt.figure(figsize=(10, 10))
    # plt.title('Precision plot for every epochs.')
    # x = np.linspace(0, length, length, endpoint=False)
    # plt.xlabel('Epochs -->', fontsize=18)
    # plt.ylabel('Precision -->', fontsize=18)
    # plt.plot(x, plot_dict['precision'])
    # plt.ylim(0.0, 1.0)
    # plt.xticks(list(range(1, length, 10)))
    # plt.savefig(os.path.join(output_path, 'Precision.png'))

    # plt.figure(figsize=(10, 10))
    # plt.title('Recall plot for every epochs.')
    # x = np.linspace(0, length, length, endpoint=False)
    # plt.xlabel('Epochs -->', fontsize=18)
    # plt.ylabel('Recall -->', fontsize=18)
    # plt.plot(x, plot_dict['recall'])
    # plt.xticks(list(range(1, length, 10)))
    # plt.ylim(-0.0, 1.0)
    # plt.savefig(os.path.join(output_path, 'Recall.png'))

    # plt.figure(figsize=(10, 10))
    # plt.title('F-score plot for every epochs.')
    # x = np.linspace(0, length, length, endpoint=False)
    # plt.xlabel('Epochs -->', fontsize=18)
    # plt.ylabel('F-score -->', fontsize=18)
    # plt.plot(x, plot_dict['F-score'])
    # plt.xticks(list(range(1, length, 10)))
    # plt.ylim(0.0, 1.0)
    # plt.savefig(os.path.join(output_path, 'F_score.png'))

    plt.figure(figsize=(10, 10))
    plt.title('COCO-PQ plot for every epochs.')
    x = np.linspace(16, length+16, length, endpoint=False)
    plt.xlabel('Epochs -->', fontsize=18)
    plt.ylabel('COCO-PQ -->', fontsize=18)
    plt.plot(x, plot_dict['COCO_PQ'])
    plt.xticks(list(range(16, length+16, 5)))
    plt.axvline(x=55, label='Selected weight at epoch {}'.format(55), c='r')
    plt.ylim(0.0, 0.5)
    plt.legend()
    plt.savefig(os.path.join(output_path, 'COCO-PQ_HED_SCRATCH.png'))


def main():
    parser = argparse.ArgumentParser(description='Takes an edge probability map and produces a label map. Edge pixels > 0.')
    parser.add_argument('input_path', type=str, default=None, help='Input path of the saving dictionary into numpy format.')
    parser.add_argument('output_path', type=str, default=None, help='Path of saving curve.')

    args = parser.parse_args()
    plot_graph(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
