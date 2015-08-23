# -*- coding: utf-8 -*-
# extract feature from each images using Imagenet_model
# try diffrent k value at 10 multiple till 150
# plot graph k vs error
# finally create new directory set

# python main.py <dataset location> <director where result stored>
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
from sklearn.cluster import MiniBatchKMeans
import shutil
import cPickle as pickle
import random
import Queue
import threading


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

class CreateData(threading.Thread):

    def __init__(self, q, file_list, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.q = q
        self.file_list = file_list
        self.batch_size = batch_size
        self.file_list_paths_success = []

        # create static variable
        CreateData.completed_chunks = 0

    def run(self):
        number_of_iteration =  get_num_batch(len(self.file_list), self.batch_size)
        images_feature = None
        for batch_index in xrange(number_of_iteration):
            images = []
            for f in self.file_list[(batch_index) * self.batch_size: (batch_index+1) * self.batch_size]:

                # if image are not valid formate reject it
                try:
                    image = cv2.imread(sys.argv[1]+'/'+f)
                    if image is None:
                        raise Exception('image cannot be read')
                except Exception as e:
                        print e
                    	continue


                self.file_list_paths_success.append(sys.argv[1]+'/'+f)
                image = create_fixed_image_shape(image, frame_size=(227, 227, 3))
                images.append(image)
            images = np.asarray(images)
            input_image = np.transpose(images, (0, 3, 1, 2))

            # do lock on global net object
            network_lock.acquire()

            #  data from network, But you can replace with your own data source
            data, _ = network.get_data(net, input_image)
            network_lock.release()

            if images_feature is None:
                images_feature = data
            else:
                images_feature = np.vstack((images_feature, data))

        status_lock.acquire()


        if np.isnan(np.min(images_feature)):
            print "write data from thread_id->", self.thread_id, 'feature_shape', images_feature.shape, 'Found NaN'
            # images_feature = images_feature.nan_to_num()
        else:
            print "write data from thread_id->", self.thread_id, 'feature_shape', images_feature.shape


        CreateData.completed_chunks = CreateData.completed_chunks + 1
        print 'percentage of task completed', (CreateData.completed_chunks/float(len(chunks))) * 100, "images_feature shape", images_feature.shape
        print "created data from thread_id->", self.thread_id
        status_lock.release()

        self.q.put((images_feature, self.thread_id))

class ReadData(threading.Thread):

    def __init__(self, q, thread_id):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.q = q

    def run(self):
        images_feature, _id = self.q.get(1)
        cluster_lock.acquire()


        if np.isnan(np.sum(images_feature)):
            print "read data from data id->", _id, 'feature_shape', images_feature.shape, 'Found NaN'
        else:
            print "read data from data id->", _id, 'feature_shape', images_feature.shape


        try:
            clr.partial_fit(images_feature)
        except Exception as e:
            print images_feature, e

        cluster_lock.release()

def create_possible_batch(file_list, batch_size, percentage = 0.8):
    extras = len(file_list)%batch_size
    if extras >= (batch_size * percentage):
        file_list += random.sample(file_list, batch_size - extras)
    else:
        file_list = file_list[:-extras]   
    return file_list

NETWORK_BATCH_SIZE = network.BATCH_SIZE

weights = '/home/invenzone/new_disk/inmobi_project/clustering/model/bvlc_alexnet.caffemodel'
model = '/home/invenzone/new_disk/inmobi_project/clustering/model/deploy.prototxt'

net = network.create_network(model, weights)
network_lock = threading.Lock()
cluster_lock = threading.Lock()
status_lock = threading.Lock()


file_list = getFiles(sys.argv[1])
random.shuffle(file_list)

# do max 5 batch fetch
batch_data_queue = Queue.Queue(10)



thread_objs_global = []
thread_objs_local = []
BATCH_SIZE_CLUSTER = 10000
epoch = 10
k = 90
# BATCH_SIZE_CLUSTER = 500
# epoch = 5
# k = 50
# create batch with ranom filling to have len(file_list)%BATCH_SIZE_CLUSTER == 0
file_list = create_possible_batch(file_list, BATCH_SIZE_CLUSTER, percentage = 0.8)

clr = MiniBatchKMeans(n_clusters=k, init='k-means++', batch_size=BATCH_SIZE_CLUSTER, compute_labels=True, max_no_improvement=None, n_init=10)
chunks = [file_list[x:x+BATCH_SIZE_CLUSTER] for x in xrange(0, len(file_list), BATCH_SIZE_CLUSTER)]

if len(sys.argv) != 4:
    for e in xrange(epoch):

    #     # clear local at each iteration (epoch)
    #     thread_objs_local[:] = []

    #     # create new create and read threads
    #     for i in xrange(len(chunks)):
    #         thread_objs_local.append(CreateData(batch_data_queue, chunks[i], NETWORK_BATCH_SIZE, i))
        
    #     for i in xrange(len(chunks)):
    #         thread_objs_local.append(ReadData(batch_data_queue, i))

    #     # keep reference of local in golobal variable
    #     thread_objs_global += thread_objs_local

    #     for thread in thread_objs_local:
    #         thread.start()

    # for thread in thread_objs_global:
    #     thread.join()

        # clear local at each iteration (epoch)
        thread_objs_local[:] = []

        # create new create and read threads
        for i in xrange(len(chunks)):
            thread_objs_local.append(CreateData(batch_data_queue, chunks[i], NETWORK_BATCH_SIZE, i))
        
        for i in xrange(len(chunks)):
            thread_objs_local.append(ReadData(batch_data_queue, i))

        # keep reference of local in golobal variable
        # thread_objs_global += thread_objs_local

        for thread in thread_objs_local:
            thread.start()

        for thread in thread_objs_local:
            thread.join()

        print "epoch completed->", e+1


    fw = open('CENTROID.file', 'w')
    pickle.dump(clr, fw)
    fw.close()
else:
    fr = open('CENTROID.file')
    clr = pickle.load(fr)
    fr.close()

# reset the file_list for non repatative data
file_list = getFiles(sys.argv[1])
number_of_iteration =  get_num_batch(len(file_list), NETWORK_BATCH_SIZE)
file_list_path = []
images_feature = None
print "creating final result"
for batch_index in xrange(number_of_iteration):
    images = []
    for f in file_list[(batch_index) * NETWORK_BATCH_SIZE: (batch_index+1) * NETWORK_BATCH_SIZE]:
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

    if images_feature.shape[0]/BATCH_SIZE_CLUSTER == 1:
        labels = clr.predict(images_feature)
        print len(labels), len(file_list_path), 'list length'
        for l, f in zip(labels, file_list_path):
            if l != -1:
                n = f.split('/')[-1]
                path = test_addDir(sys.argv[2], str(l))
                shutil.copyfile(f, path+'/'+n)
        images_feature = None
        file_list_path = []

    if batch_index%10 == 0:
        print ((batch_index+1)/float(number_of_iteration)) * 100, 'completed'
