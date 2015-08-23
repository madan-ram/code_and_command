# -*- coding: utf-8 -*-
# extract feature from each images using Imagenet_model
# try diffrent k value at 10 multiple till 150
# plot graph k vs error
# finally create new directory set


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
import network
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, DBSCAN
import shutil
import os
import cPickle as pickle
import random

weights = '/home/invenzone/new_disk/inmobi_project/clustering/model/bvlc_alexnet.caffemodel'
model = '/home/invenzone/new_disk/inmobi_project/clustering/model/deploy.prototxt'

def test_addDir(path, dir):
	if not os.path.exists(path+'/'+dir):
		os.makedirs(path+'/'+dir)
	return path+'/'+dir

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

	image_frame[Y_space_center: Y_space_center+Y_new, X_space_center: X_space_center+X_new] = img
	return image_frame


feature_size = network.FEATURE_SIZE
batch_size = network.BATCH_SIZE

net = network.create_network(model, weights)

file_list = getFiles(sys.argv[1])
random.shuffle(file_list)

number_of_iteration =  get_num_batch(len(file_list), batch_size)


batch_size_cluster = 10000

if len(sys.argv) != 4:
	images_feature = None
	epoch = 20
	k = 90
	clr = MiniBatchKMeans(n_clusters=k, init='k-means++', batch_size=batch_size_cluster, compute_labels=True, max_no_improvement=None, n_init=10)
	for e in xrange(epoch):
		for batch_index in xrange(number_of_iteration):
			images = []
			for f in file_list[(batch_index) * batch_size: (batch_index+1) * batch_size]:
				image = cv2.imread(sys.argv[1]+'/'+f)

				if image is None:
					continue

				image = create_fixed_image_shape(image, frame_size=(227, 227, 3))
				images.append(image)
			images = np.asarray(images)

			input_image = np.transpose(images, (0, 3, 1, 2))

			#  data from network, But you can replace with your own data source
			data, _ = network.get_data(net, input_image)

			# vertically append data to images_feature
			if images_feature is None:
				images_feature = data
			else:
				images_feature = np.vstack((images_feature, data))

			if batch_index%10 == 0:
				print ((batch_index+1)/float(number_of_iteration)) * 100, 'completed'

			if images_feature.shape[0]/batch_size_cluster == 1:
				print 'fiting model for data of shape', images_feature.shape
				clr.partial_fit(images_feature)
				images_feature = None

		print "epoch completed so far", e

	fw = open('CENTROID.file', 'w')
	pickle.dump(clr, fw)
	fw.close()
else:
	fr = open('CENTROID.file')
	clr = pickle.load(fr)
	fr.close()

file_list_path = []
images_feature = None
print "creating final result"
for batch_index in xrange(number_of_iteration):
	images = []
	for f in file_list[(batch_index) * batch_size: (batch_index+1) * batch_size]:
		image = cv2.imread(sys.argv[1]+'/'+f)

		if image is None:
			continue

		file_list_path.append(sys.argv[1]+'/'+f)
		image = create_fixed_image_shape(image, frame_size=(227, 227, 3))
		images.append(image)
	images = np.asarray(images)

	input_image = np.transpose(images, (0, 3, 1, 2))

	#  data from network, But you can replace with your own data source
	data, _ = network.get_data(net, input_image)

	# vertically append data to images_feature
	if images_feature is None: 
		images_feature = data
	else:
		images_feature = np.vstack((images_feature, data))

	if images_feature.shape[0]/batch_size_cluster == 1:
		labels = clr.predict(images_feature)
		print len(labels), len(file_list_path), 'list lengith'
		for l, f in zip(labels, file_list_path):
			if l != -1:
				n = f.split('/')[-1]
				path = test_addDir(sys.argv[2], str(l))
				shutil.copyfile(f, path+'/'+n)
		images_feature = None
		file_list_path = []

	if batch_index%10 == 0:
		print ((batch_index+1)/float(number_of_iteration)) * 100, 'completed'



# centroid_list = xrange(10, 500, 10)
# score_list = []
# score_list_temp = []
# batch_size_cluster = 5000
# k_list = []
# total_cluster = 0
# for k in centroid_list:
# 	clr = MiniBatchKMeans(n_clusters=k, init='random', max_iter=10, batch_size=batch_size_cluster, compute_labels=True, max_no_improvement=None, n_init=1)
# 	for batch_index in xrange(number_of_iteration):
# 		images = []
# 		for f in file_list[(batch_index) * batch_size: (batch_index+1) * batch_size]:
# 			image = cv2.imread(sys.argv[1]+'/'+f)
			
# 			if image is None:
# 				continue

# 			file_list_path.append(sys.argv[1]+'/'+f)
# 			image = create_fixed_image_shape(image, frame_size=(227, 227, 3))
# 			images.append(image)
# 		images = np.asarray(images)

# 		input_image = np.transpose(images, (0, 3, 1, 2))

# 		data, _ = network.get_data(net, input_image)

# 		#vertically apped data to images_feature
# 		if images_feature is None: 
# 			images_feature = data
# 		else:
# 			images_feature = np.vstack((images_feature, data))

# 		total_cluster += images.shape[0]
# 		print (batch_index+1)/float(number_of_iteration) * 100, 'completed'

# 		if total_cluster%batch_size_cluster == 0:
# 			total_cluster = 0
# 			images_feature = None
# 			score_list_temp = []
# 			partial_fit(images_feature)
# 			score_list_temp.append(clr.score(images_feature))

# 	score_list.append(sum(score_list_temp)/float(batch_size_cluster/batch_size))
# 	k_list.append(k)
# 	print "completed", k, 'centroid cluster'

# plt.plot(k_list, score_list)
# plt.savefig('kVsScore.png')

# centroid_list = xrange(10, 500, 10)

# score_list = []
# k_list = []
# for k in centroid_list:
# 	# clr = KMeans(n_clusters=k, init='random', n_init=1, max_iter=300, tol=0.0001, precompute_distances=False, verbose=0, n_jobs=4)
# 	clr = MiniBatchKMeans(n_clusters=k, init='random', max_iter=10, batch_size=5000, compute_labels=True, max_no_improvement=None, n_init=1)
# 	clr.fit(images_feature)

# 	score_list.append(clr.score(images_feature))
# 	k_list.append(k)
# 	print "completed", k, 'centroid cluster'

# plt.plot(k_list, score_list)
# plt.savefig('kVsScore.png')

# clr = KMeans(n_clusters=75, init='random', n_init=1, max_iter=500, tol=0.0001, precompute_distances=False, verbose=0, n_jobs=4)
# clr = MeanShift(min_bin_freq=5)

# for d in [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]:
# 	clr = DBSCAN(eps=d, min_samples=5)
# 	print d
# 	labels = clr.fit_predict(images_feature)

# 	for l, f, n in zip(labels, file_list_path, file_list):
# 		if l != -1:
# 			path = test_addDir(sys.argv[2], str(l)+'_'+str(d))
# 			shutil.copyfile(f, path+'/'+n)
# 		# else:
# 		# 	print f, n
