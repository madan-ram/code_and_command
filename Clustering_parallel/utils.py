# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from os.path import isfile, join
from os import listdir
import math
import sys
import itertools

def test_addDir(dir, path):
	if not os.path.exists(path+'/'+dir):
		os.makedirs(path+'/'+dir)
	return path+'/'+dir

def generate_window_locations(location, patch_shape, stride=0.5, grid_shape=5):
	"""
		Generate a list of window location based on location, patch_shape, grid_shape and stride

		location:
			(y, x) a tuple for center location of window.
		patch_shape:
			(y, x) a tuple which represent window shape.
		stride {default: 0.5}:
			a float which represent stride to be taken.
		grid_shape :
			a integer that represent grid shape where grid_x == grid_y

	"""
	assert(grid_shape%2 != 0), "grid_shape should be odd number"

	assert(stride != 0), "stride should not be <= 0"

	center_y, center_x = location
	string = ""
	mapping = {}

	# left is represented as -, center as 0  and right is +
	if grid_shape%2 != 0:
		pointer = xrange(-(grid_shape/2), (grid_shape/2)+1)
	# else:
	# 	pointer = range(-(grid_shape/2)+1, (grid_shape/2)+1)

	for i, n, in enumerate(pointer):
		mapping[i] = n
		string += str(i)
	windows_list = []	
	sequences = np.asarray(list(itertools.product(string, repeat=2)), dtype="int32")

	for y, x in sequences:
		new_center_y, new_center_x = center_y +patch_shape[0]*mapping[y]*stride, center_x+patch_shape[1]*mapping[x]*stride

		res =  np.asarray([(math.floor(new_center_y)-patch_shape[0]/2,math.floor(new_center_x)-patch_shape[1]/2), 
		(math.ceil(new_center_y)+patch_shape[0]/2, math.ceil(new_center_x)+patch_shape[1]/2)], dtype="int32")

		windows_list.append(res)

	return np.asarray(windows_list).tolist()

def getImmediateSubdirectories(dir):
	"""
		this function return the immediate subdirectory list
		eg:
			dir
				/subdirectory1
				/subdirectory2
				.
				.
		return ['subdirectory1',subdirectory2',...]
	"""

	return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

def getFiles(dir_path):
    """getFiles : gets the file in specified directory
    dir_path: String type
    dir_path: directory path where we get all files
    """
    onlyfiles = [ f for f in listdir(dir_path) if isfile(join(dir_path, f)) ]
    return onlyfiles

def get_num_batch(data_size, batch_size):
	if data_size%batch_size == 0:
		return data_size/batch_size
	return (data_size/batch_size) + 1

def feature_normalization(data, type='standardization', params = None):
	u"""
		data:
			an numpy array
		type:
			(standardization, min-max)
		params {default None}: 
			dictionary
			if params is provided it is used as mu and sigma when type=standardization else Xmax, Xmin when type=min-max
			rather then calculating those paramsanter
		two type of normalization 
		1) standardization or (Z-score normalization)
			is that the features will be rescaled so that they'll have the properties of a standard normal distribution with
				μ = 0 and σ = 1
			where μ is the mean (average) and σ is the standard deviation from the mean
				Z = (X - μ)/σ
			return:
				Z, μ, σ
		2) min-max normalization
			the data is scaled to a fixed range - usually 0 to 1.
			The cost of having this bounded range - in contrast to standardization - is that we will end up with smaller standard 
			deviations, which can suppress the effect of outliers.
			A Min-Max scaling is typically done via the following equation:
				Z = (X - Xmin)/(Xmax-Xmin)
			return Z, Xmax, Xmin
	"""
	if type == 'standardization':
		if params is None:
			mu = np.mean(data, axis=0)
			sigma =  np.std(data, axis=0)
		else:
			mu = params['mu']
			sigma = params['sigma']
		Z = (data - mu)/sigma
		return Z, mu, sigma

	elif type == 'min-max':
		if params is None:
			Xmin = np.min(data, axis=0)
			Xmax = np.max(data, axis=0)
		else:
			Xmin = params['Xmin']
			Xmax = params['Xmax']

		Xmax = Xmax.astype('float')
		Xmin = Xmin.astype('float')
		Z = (data - Xmin)/(Xmax - Xmin)
		return Z, Xmax, Xmin