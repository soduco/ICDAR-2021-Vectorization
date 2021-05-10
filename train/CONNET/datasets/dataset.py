import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import random
from io import BytesIO
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
import imgaug.augmenters as iaa
import imgaug as ia
import pdb


def load_image_with_cache(path, cache=None, lock=None):
	if cache is not None:
		if path not in cache:
			with open(path, 'rb') as f:
				cache[path] = f.read()
		return Image.open(BytesIO(cache[path]))
	return Image.open(path)


class Data(data.Dataset):
	def __init__(self, root, lst,
		mean_bgr = np.array([168.1872554, 195.97301654, 204.64264255]),
		crop_size=None, rgb=True, scale=None, mode='Train', augment=False):
		self.mean_bgr = mean_bgr
		self.root = root
		self.lst = lst
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}
		self.mode = mode
		self.augment = augment

		lst_dir = os.path.join(self.root, self.lst)

		with open(lst_dir, 'r') as f:
			self.files = f.readlines()
			self.files = [line.strip().split(' ') for line in self.files]

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		img_file = self.root + data_file[0]
		# print(img_file)
		if not os.path.exists(img_file):
			img_file = img_file.replace('jpg', 'png')
		# img = Image.open(img_file)
		img = load_image_with_cache(img_file, self.cache)
		img = img.convert('RGB')
		# load gt image
		gt_file = self.root + data_file[1]
		# gt = Image.open(gt_file)
		gt = load_image_with_cache(gt_file, self.cache)
		if gt.mode == '1':
			gt  = gt.convert('L')
		return self.transform(img, gt)

	def transform(self, img, gt):
		gt = np.array(gt, dtype=np.float32)

		if len(gt.shape) == 3:
			gt = gt[:, :, 0]
		gt /= 255.

		if self.mode=='Train':
			struct1 = ndimage.generate_binary_structure(2, 2)
			gt = binary_dilation(gt, structure=struct1).astype(np.uint8)
			img = np.array(img, dtype=np.float32)
		else:
			img = np.array(img, dtype=np.float32)

		if self.rgb:
			img = img[:, :, ::-1] # RGB->BGR

		img -= self.mean_bgr
		img = img.astype(np.uint8)

		if self.augment:
			img, gt = data_augmentation(img, gt)
		else:
			img = np.transpose(img, (2, 0, 1))

		img = torch.from_numpy(img.copy()).float()
		gt = torch.from_numpy(np.array([gt])).float()
		return img, gt

def data_augmentation(images, labels):
	# Data augmentation
	seq = iaa.Sequential([
		iaa.Affine(rotate = (-15, 15),
				translate_px = {"x": (-15, 15),
								"y": (-15, 15)})
	], random_order = True)

	# Image shape: (1, 500, 500, 3)
	# GT image:    (1, 500, 500, 1)
	# Flip images and labels
	images = np.expand_dims(images, axis=0)
	labels = np.expand_dims(labels, axis=(0, -1))

	# images: (1, 500, 500, 3)
	# labels: (1, 500, 500, 1)
	images, labels = seq(images=images, segmentation_maps=labels)
	# images = np.squeeze(images, axis=0)

	images = images[0,:,:,:]
	# images = Image.fromarray(images, mode='RGB')
	# images.save('/lrde/home2/ychen/code_for_ICDAR/icdar21-paper-map-object-seg/BDCN_Connet/augment_image/image.png')
	labels = np.squeeze(labels)
	# cv2.imwrite('/lrde/home2/ychen/code_for_ICDAR/icdar21-paper-map-object-seg/BDCN_Connet/augment_image/gt.png', labels*255)
	images = np.transpose(images, (2, 0, 1))
	# images: (3, 500, 500)
	# labels: (500, 500)
	return images, labels
