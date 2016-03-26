import os
import numpy as np
import caffe

FEATURE_SIZE = 4096
BATCH_SIZE = 24
FRAME_SIZE = (224, 224, 3)
def create_network(model, weights):
    #set caffe compution mode gpu or cpu
    caffe.set_mode_gpu()

    #load model and weights for testing
    net = caffe.Net(model, weights, caffe.TEST)

    return net

def get_data(net, batch_data):
    # convert list to numpy array 
    images = np.asarray(batch_data)
    # transpose data into (batch_size, channel, heught, width)
    images = np.transpose(images, (0, 3, 1, 2))

    #change the shape of blog to accomadate data
    net.blobs['data'].reshape(*images.shape)
    #set the data
    net.blobs['data'].data[...] = images

    #fedforward network to get the layer activation
    output = net.forward()
    prob = net.blobs['prob']
    data = np.random.randn(10, 10)
    #get layerwise activation here from layer fc7 (fully connected layer 7 as name specified in deploy.prototxt)
    # data = net.blobs['pool5'].data
    # data = net.blobs['fc6'].data
    # data = data.reshape((data.shape[0], np.prod(data.shape[1:])))
    return data, prob.data