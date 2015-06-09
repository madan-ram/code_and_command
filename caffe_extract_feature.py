# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
from os.path import isfile, join
from os import listdir
import cv2
from sklearn import manifold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


caffe_root = '/home/invenzone/digits-1.0/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python/')
import caffe

# model = 'deploy.prototxt'
# weights = 'AlexNet_SalObjSub.caffemodel'

model = '/home/invenzone/.digits/jobs/20150608-154456-3158/deploy.prototxt'
weights = '/home/invenzone/.digits/jobs/20150608-154456-3158/snapshot_iter_9792.caffemodel'

caffe.set_mode_gpu()

# IMAGE_FILE = 'result_198.jpg'



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

def feature_normalization(data, type='standardization', param = None):
	u"""
		data:
			an numpy array
		type:
			(standardization, min-max)
		param {default None}: 
			dictionary
			if param is provided it is used as mu and sigma when type=standardization else Xmax, Xmin when type=min-max
			rather then calculating those paramanter
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
		if param is None:
			mu = np.mean(data, axis=0)
			sigma =  np.std(data, axis=0)
		else:
			mu = param['mu']
			sigma = param['sigma']
		Z = (data - mu)/sigma
		return Z, mu, sigma

	elif type == 'min-max':
		if param is None:
			Xmin = np.min(data, axis=0)
			Xmax = np.max(data, axis=0)
		else:
			Xmin = param['Xmin']
			Xmax = param['Xmax']

		Xmax = Xmax.astype('float')
		Xmin = Xmin.astype('float')
		Z = (data - Xmin)/(Xmax - Xmin)
		return Z, Xmax, Xmin

def create_fixed_image_shape(img, frame_size=(200, 200, 3)):
	X1, Y1, _ = frame_size
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
	image_frame[Y_space_center: Y_space_center+Y_new, X_space_center: X_space_center+X_new] = img
	return image_frame

# temp_file = open('result.txt', 'w')

#calculate the batch size
batch_size = 10
file_list = getFiles(sys.argv[1])
number_of_iteration =  get_num_batch(len(file_list), batch_size)

images_feature = None

for batch_index in xrange(number_of_iteration):
	images = []
	for f in file_list[(batch_index) * batch_size: (batch_index+1) * batch_size]:
		image = caffe.io.load_image(sys.argv[1]+'/'+f)

		image = create_fixed_image_shape(image, frame_size=(227, 227, 3))
		images.append(image)
	images = np.asarray(images)

	input_image = np.transpose(images, (0, 3, 1, 2))

	#load model and weights for testing
	net = caffe.Net(model, weights, caffe.TEST)

	#change the shape of blog to accomadate data
	net.blobs['data'].reshape(*input_image.shape)
	#set the data
	net.blobs['data'].data[...] = input_image

	#fedforward network to get the layer activation
	net.forward()

	#get layerwise activation here from layer fc7 (fully connected layer 7 as name specified in deploy.prototxt)
	data = net.blobs['fc7'].data
	data = data.reshape((data.shape[0], np.prod(data.shape[1:])))

	#vertically apped data to images_feature
	if images_feature == None: 
		images_feature = data
	else:
		images_feature = np.vstack((images_feature, data))

	# data_str = map(str, data)
	#print >> temp_file, ' '.join(data_str)

# temp_file.close()

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne, _ , _ = feature_normalization(tsne.fit_transform(images_feature), type='min-max')


S = 2000
s = 50
G = np.zeros((S, S, 3))
T = np.zeros((S, S))

for i in xrange(len(file_list)):
	image = cv2.imread(sys.argv[1]+'/'+file_list[i])
	image = create_fixed_image_shape(image, frame_size=(s, s, 3))

	a = X_tsne[i, 0] * (S - s)
	b = X_tsne[i, 1] * (S - s)

	if T[a,b] == 1:
		continue

	G[a:a+s, b:b+s, :] = image
	T[a:a+s, b:a+s] = 1

cv2.imwrite('test_1.png', G)



G = np.zeros((S, S, 3))
S = 2000
s = 50

a_min = 0
b_min = 0
a_max = 2000
b_max = 2000

h_a = s
h_b = s
aa, bb = np.meshgrid(np.arange(a_min, a_max, h_a),
                     np.arange(b_min, b_max, h_b))

temp = np.asarray(np.c_[aa.ravel(), bb.ravel()], dtype="float32")

X_tsne_unScaled = (X_tsne * (S - s))

for i in xrange(len(file_list)):
	fName = file_list[i]
	val = X_tsne_unScaled[i]	
	dd = np.sum((temp - val)**2, axis=1)
	index = np.argmin(dd)
	x_place, y_place = temp[index].astype('int')
	temp[index] = np.asarray([np.inf, np.inf])

	image = cv2.imread(sys.argv[1]+'/'+file_list[i])
	image = create_fixed_image_shape(image, frame_size=(s, s, 3))
	G[x_place:x_place+s, y_place:y_place+s] = image

cv2.imwrite('test_2.png', G)
