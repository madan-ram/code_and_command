import cv2
import numpy as np 
import sys
from os.path import isfile, join
import os
from os import listdir

def getFiles(dir_path):
    """getFiles : gets the file in specified directory
    dir_path: String type
    dir_path: directory path where we get all files
    """
    onlyfiles = [ f for f in listdir(dir_path) if isfile(join(dir_path, f)) ]
    return onlyfiles

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


# img = cv2.imread('done/26a.jpg')
# cv2.imshow('ads', img)
# cv2.waitKey(0)

# img = create_fixed_image_shape(img)
# print img.shape
# cv2.imshow('ads', img)
# cv2.waitKey(0)

#debuging correctness
# for f in getFiles(sys.argv[1]):
# 	img = cv2.imread(sys.argv[1]+'/'+f)
# 	print img.shape,
# 	img = create_fixed_image_shape(img)
# 	print img.shape
