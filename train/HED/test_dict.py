
import pdb
import pickle

if __name__ == '__main__':
    input_dict = '/lrde/home2/ychen/code_for_ICDAR/ICDAR_paper/icdar21-paper-map-object-seg/evaluation/HED_scratch_results/HED_scratch_plot_info.pkl'
    with open(input_dict, 'rb') as f:
        x = pickle.load(f)
    pdb.set_trace()