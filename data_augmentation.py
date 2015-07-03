import cv2
import numpy as np
from utils import *
import sys
from scipy.ndimage.interpolation import rotate

def data_augmentation(orignial_img, size_factor_up=1.5, size_factor_down=0.5):
	result = []
	img = create_fixed_image_shape(orignial_img, frame_size=(256, 256, 3), random_fill=True, mode='fit')
	result.append(img)
	img = create_fixed_image_shape(orignial_img, frame_size=(256, 256, 3), random_fill=True, mode='crop')
	result.append(img)

	#rotate image 30 45 -30 -45
	img_30 = rotate(orignial_img, 30)
	img = create_fixed_image_shape(img_30, frame_size=(256, 256, 3), random_fill=True, mode='fit')
	result.append(img)
	img = create_fixed_image_shape(img_30, frame_size=(256, 256, 3), random_fill=True, mode='crop')
	result.append(img)

	img_45 = rotate(orignial_img, 45)
	img = create_fixed_image_shape(img_45, frame_size=(256, 256, 3), random_fill=True, mode='fit')
	result.append(img)
	img = create_fixed_image_shape(img_45, frame_size=(256, 256, 3), random_fill=True, mode='crop')
	result.append(img)

	img_m30 = rotate(orignial_img, -30)
	img = create_fixed_image_shape(img_m30, frame_size=(256, 256, 3), random_fill=True, mode='fit')
	result.append(img)
	img = create_fixed_image_shape(img_m30, frame_size=(256, 256, 3), random_fill=True, mode='crop')
	result.append(img)

	img_m45 = rotate(orignial_img, -45)
	img = create_fixed_image_shape(img_m45, frame_size=(256, 256, 3), random_fill=True, mode='fit')
	result.append(img)
	img = create_fixed_image_shape(img_m45, frame_size=(256, 256, 3), random_fill=True, mode='crop')
	result.append(img)

	#scale up by factor of 50%(size_factor_up) more
	img_large = cv2.resize(orignial_img, (int(orignial_img.shape[1]*size_factor_up),int(orignial_img.shape[0]*size_factor_up)))
	center = (img_large.shape[0]/2, img_large.shape[1]/2)
	y, x = (center[0]-orignial_img.shape[0]/2, center[1]-orignial_img.shape[1]/2)
	y1, x1 = (center[0]+orignial_img.shape[0]/2, center[1]+orignial_img.shape[1]/2)
	img_large = img_large[y:y1, x:x1]

	img = create_fixed_image_shape(img_large, frame_size=(256, 256, 3), random_fill=True, mode='fit')
	result.append(img)

	img = create_fixed_image_shape(img_large, frame_size=(256, 256, 3), random_fill=True, mode='crop')
	result.append(img)

	cv2.imshow('img_large', img)
	cv2.waitKey(0)

	#scale down by factor of 50%(size_factor_down) more
	img_small = cv2.resize(orignial_img, (int(orignial_img.shape[1]*size_factor_down),int(orignial_img.shape[0]*size_factor_down)))
	center = (orignial_img.shape[0]/2, orignial_img.shape[1]/2)

	y, x = (center[0]-img_small.shape[0]/2, center[1]-img_small.shape[1]/2)
	y1, x1 = (center[0]+img_small.shape[0]/2, center[1]+img_small.shape[1]/2)

	img_frame = np.zeros(orignial_img.shape, dtype='uint8')
	img_frame[y:y1, x:x1, :] = img_small[:y1-y,:x1-x] 


	img = create_fixed_image_shape(img_frame, frame_size=(256, 256, 3), random_fill=True, mode='fit')
	result.append(img)

	img = create_fixed_image_shape(img_frame, frame_size=(256, 256, 3), random_fill=True, mode='crop')
	result.append(img)

	return result


# for f in getFiles(sys.argv[1]):
# 	path_files = sys.argv[1]+'/'+f
# 	img = cv2.imread(path_files)
# 	result = data_augmentation(img, size_factor_up=1.5)

# 	print len(result)
