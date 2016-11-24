import os, sys
import cv2
import numpy as np
import random
import math
from os.path import isfile, join
from os import listdir

def get_files_abs_path(dir_path):
    """getFiles : gets the file in specified directory
    dir_path: String type
    dir_path: directory path where we get all files
    """
    onlyfiles = [ os.path.join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) ]
    return onlyfiles

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), None, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 255)

    # rotated = rotated.reshape((rotated.shape[0], rotated.shape[1], 1))
    return rotated

def read_img(path, batch_size=32, balance_class=True):
	yes_filelist = get_files_abs_path(os.path.join(path, 'yes'))
	no_filelist = get_files_abs_path(os.path.join(path, 'no'))
	if balance_class:
		random.shuffle(no_filelist)
		no_filelist = no_filelist[:len(yes_filelist)]

	filelist = [(x, 0) for x in no_filelist]
	filelist += [(x, 1) for x in yes_filelist]

	random.shuffle(filelist)

	num_batchs = int(math.ceil(len(filelist)/float(batch_size)))
	for batch_id in xrange(num_batchs):
		result_img = []
		result_label = []
		for img_path, label in filelist[batch_id * batch_size:(batch_id+1) * batch_size]:
			img = cv2.imread(img_path, 0)
			# data agumneation
			img = rotate(img, random.randint(-30, 30))

			result_img.append(img)
			result_label.append(label)

		result_img = np.asarray(result_img)
		result_img = result_img[..., None]
		yield(result_img, result_label)


def read_img_test(path, batch_size=32, balance_class=True):
	yes_filelist = get_files_abs_path(os.path.join(path, 'yes'))
	no_filelist = get_files_abs_path(os.path.join(path, 'no'))
	if balance_class:
		no_filelist = no_filelist[:len(yes_filelist)]

	filelist =  no_filelist + yes_filelist
	filelist_with_label = [(x, 0) for x in no_filelist]
	filelist_with_label += [(x, 1) for x in yes_filelist]

	num_batchs = int(math.ceil(len(filelist_with_label)/float(batch_size)))
	for batch_id in xrange(num_batchs):
		result_img = []
		result_label = []
		for img_path, label in filelist_with_label[batch_id * batch_size:(batch_id+1) * batch_size]:
			img = cv2.imread(img_path, 0)
			result_img.append(img)
			result_label.append(label)

		result_img = np.asarray(result_img)
		result_img = result_img[..., None]
		yield(result_img, result_label, filelist[batch_id * batch_size:(batch_id+1) * batch_size])

if __name__ == '__main__':
	import random
	for imgs, labels in read_img('/home/arya_01/AxisProject/AxisFilter/new_output_test'):
		""
		# for img in imgs:
		# 	print img.shape
		# 	res = rotate(img, random.randint(-30, 30))
		# 	print img.shape, res.shape
		# 	img = np.hstack((img, res))
		# 	cv2.imwrite('tmp/imgs/1.png', img)
		# 	# print labels
		# 	sys.exit(0)