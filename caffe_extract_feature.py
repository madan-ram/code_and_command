import numpy as np
import os

caffe_root = '/home/invenzone/digits-1.0/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python/')
import caffe

#note wight, image and model location should be modified
model = ''
weights = ''
IMAGE_FILE = ''


caffe.set_mode_cpu()

#requre input shape should be batch size, image channel, height, weight
input_image = caffe.io.load_image(IMAGE_FILE)
input_image = input_image.reshape((input_image.shape[0], input_image.shape[1], input_image.shape[2], 1))
input_image = np.transpose(input_image, (3, 2, 0, 1))

net = caffe.Net(model, weights, caffe.TEST)
# print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

net.blobs['data'].reshape(*input_image.shape)
net.blobs['data'].data[...] = input_image
net.forward()

#the extracted feature is found here.
print net.blobs['fc7'].data




