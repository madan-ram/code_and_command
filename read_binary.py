# code to read the below data
# [offset] [type]          [value]          [description] 
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
# 0004     32 bit integer  60000            number of items 
# 0008     unsigned byte   ??               label 
# 0009     unsigned byte   ??               label 
# ........ 
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.
# link for data file
# http://yann.lecun.com/exdb/mnist/index.html

import struct

f_data = open('train-labels-idx1-ubyte', 'rb')

magic_number, number_of_items = struct.unpack('>ii', f_data.read(8))

labels = []
for i in  xrange(number_of_items):
	label = struct.unpack('>B', f_data.read(1))[0]
	print label
	labels.append(label)


