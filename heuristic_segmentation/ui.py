import cv2
import numpy as np
import sys, os
import time
from utils import fit_image_into_frame
from get_boundingbox import process_extract


K = 0
_min = 0
inverse_sigma = 0.1

def callback_trackbar_change(x):
	global K, _min, inverse_sigma
	inverse_sigma = cv2.getTrackbarPos('inverse_sigma','segmentation')
	K = cv2.getTrackbarPos('K','segmentation')
	_min = cv2.getTrackbarPos('min','segmentation')
	print inverse_sigma, K, _min
	

if __name__ == '__main__':
	fp = sys.argv[1]
	img = cv2.imread(fp, 0)
	img = fit_image_into_frame(img, frame_size=(346, 800, 1), random_fill=False, fill_color=[255], mode='fit')

	cv2.namedWindow('segmentation')
	cv2.createTrackbar('inverse_sigma','segmentation', 1, 10, callback_trackbar_change)
	cv2.createTrackbar('K','segmentation', 0, 5000, callback_trackbar_change)
	cv2.createTrackbar('min','segmentation', 0, 500, callback_trackbar_change)

	cp_img = img.copy()
	cp_img = cv2.cvtColor(cp_img, cv2.COLOR_GRAY2BGR)
	while(1):
		cv2.imshow('segmentation', cp_img)
		key = cv2.waitKey(20) & 0xFF
		if  key == 27:
			break

		if key == ord("c"):
			cp_img = img.copy()
			cp_img = cv2.cvtColor(cp_img, cv2.COLOR_GRAY2BGR)

		if key == ord("r"):
			func = process_extract(img, sigma = 1/float(inverse_sigma), K = K, min = _min, batch_size=32)
			for bb_batch in func:
					for bb in bb_batch:
						min_x, min_y, max_x, max_y = bb
						cv2.rectangle(cp_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), thickness=3)
						cv2.imshow('segmentation', cp_img)

	cv2.destroyAllWindows()
