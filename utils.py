# -*- coding: utf-8 -*-
import numpy as np
import os
from os.path import isfile, join
from os import listdir
import cv2


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

def create_fixed_image_shape(img, frame_size=(200, 200, 3), random_fill=True, mode='crop'):
	image_frame = None
	if mode == 'fit':
		X1, Y1, _ = frame_size
		if random_fill:
			image_frame = np.asarray(np.random.randint(0, high=255, size=frame_size), dtype='uint8')
			print image_frame.shape
		else:
			image_frame = np.zeros(frame_size, dtype='uint8')

		X2, Y2 = img.shape[1], img.shape[0]

		if X2 > Y2:
			X_new = X1
			Y_new = int(round(float(Y2*X_new)/float(X2)))
		else:
			Y_new = Y1
			X_new = int(round(float(X2*Y_new)/float(Y2)))

		img = cv2.resize(img, (X_new, Y_new))

		X_space_center = ((X1 - X_new)/2)
		Y_space_center = ((Y1 - Y_new)/2)

		# print Y_new, X_new, Y_space_center, X_space_center
		image_frame[Y_space_center: Y_space_center+Y_new, X_space_center: X_space_center+X_new, :] = img
		
	elif mode == 'crop':
		X1, Y1, _ = frame_size
		image_frame = np.zeros(frame_size, dtype='uint8')

		X2, Y2 = img.shape[1], img.shape[0]

		#increase the size of smaller length (width or hegiht)
		if X2 > Y2:
			Y_new = Y1
			X_new = int(round(float(X2*Y_new)/float(Y2)))
		else:
			X_new = X1
			Y_new = int(round(float(Y2*X_new)/float(X2)))

		img = cv2.resize(img, (X_new, Y_new))

		
		X_space_clip = (X_new - X1)/2
		Y_space_clip = (Y_new - Y1)/2

		#trim image both top, down, left and right
		if X_space_clip == 0 and Y_space_clip != 0:
			img = img[Y_space_clip:-Y_space_clip, :]
		elif Y_space_clip == 0 and X_space_clip != 0:
			img = img[:, X_space_clip:-X_space_clip]

		if img.shape[0] != X1:
			img = img[1:, :]
		if img.shape[1] != Y1:
			img = img[:, 1:]

		image_frame[: , :] = img
	return image_frame

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
			params = {}
			mu = np.mean(data, axis=0)
			sigma =  np.std(data, axis=0)
			params['mu'] = mu
			params['sigma'] = sigma
		else:
			mu = params['mu']
			sigma = params['sigma']
		Z = (data - mu)/sigma
		return Z, mu, sigma

	elif type == 'min-max':
		if params is None:
			params = {}
			Xmin = np.min(data, axis=0)
			Xmax = np.max(data, axis=0)
			params['Xmin'] = Xmin
			params['Xmax'] = Xmax
		else:
			Xmin = params['Xmin']
			Xmax = params['Xmax']

		Xmax = Xmax.astype('float')
		Xmin = Xmin.astype('float')
		Z = (data - Xmin)/(Xmax - Xmin)
		return Z, params