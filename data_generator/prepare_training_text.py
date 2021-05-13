import os
from pathlib import Path
import numpy as np
import pdb


def create_training_list_file():
	# Write training files
	dir = str(Path(os.getcwd()))	
	data_dir = dir + '/train_data_tiles'
	gt_dir = dir + '/train_gt_tiles'
	
	data_files = os.listdir(data_dir)
	gt_files = os.listdir(gt_dir)

	train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
	num_train = int(np.floor(train_ratio * len(data_files)))
	# num_val =  int(np.floor(val_ratio * len(data_files)))

	train_data_files = data_files[0: num_train]
	train_gt_data_files = gt_files[0: num_train]

	with open(os.path.join(dir, 'train_pair.lst'),'w+') as out:
		for t, g in zip(train_data_files, train_gt_data_files):
			out.write('{} {}\n'.format('train_data_tiles/' + t, 'train_gt_tiles/' + g))
	
	val_data_files = data_files[num_train:]
	val_gt_data_files = gt_files[num_train:]

	with open(os.path.join(dir, 'val_pair.lst'),'w+') as out:
		for t, g in zip(val_data_files, val_gt_data_files):
			out.write('{} {}\n'.format('train_data_tiles/' + t, 'train_gt_tiles/' + g))

	# Test files are used for whole map prediction and evaluation
	test_data_files = data_files
	test_gt_data_files = gt_files

	with open(os.path.join(dir, 'test_pair.lst'),'w+') as out:
		for t, g in zip(data_files, gt_files):
			out.write('{} {}\n'.format('train_data_tiles/' + t, 'train_gt_tiles/' + g))

def create_testing_list_file(tiles_folder):
	# Write training files
	data_files = os.listdir(tiles_folder)
	data_parent_path = os.path.join(str(os.path.dirname(tiles_folder)), 'test_pair.lst')
	with open(data_parent_path, 'w+') as out:
		for t in data_files:
			out.write('{}\n'.format('/data_tiles/' + t))

def create_list_file(image_tiles_folder, gt_tiles_folder, list_name):
	# Write training files
	data_files = os.listdir(image_tiles_folder)
	gt_files = os.listdir(gt_tiles_folder)
	data_parent_path = os.path.join(str(os.path.dirname(image_tiles_folder)), list_name)
	with open(data_parent_path, 'w+') as out:
		for t, g in zip(data_files, gt_files):
			out.write('{} {}\n'.format('test_data_tiles/' + t, 'test_gt_tiles/' + g))

if __name__ == '__main__':
	# create_training_list_file
	# create_training_list_file()
	image_tiles_folder = './data_generator/test_data_tiles'
	gt_tiles_folder = './data_generator/test_gt_tiles'
	list_name = 'test_pair.lst'
	create_list_file(image_tiles_folder, gt_tiles_folder, list_name)
