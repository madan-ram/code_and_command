import matplotlib as mpl
mpl.use('Agg')
import os, sys
import numpy as np
import cv2
from glob  import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from get_boundingbox import process_extract
from utils import fit_image_into_frame
from network import build_filter_network

def test_create_dir(dir_path):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	return dir_path

if __name__ == '__main__':
	# filepaths = glob('data/FrontBW/*')
	filepaths = glob('data/*')

	batch_size = 32

	# create placeholder for variables
	input_t = tf.placeholder(tf.float32, [None, 128, 128, 1])
	labels_t = tf.placeholder(tf.int64, shape=[None, ])
	dropout_prob_t = tf.placeholder(tf.float32)

	predict, _ = build_filter_network(input_t, labels_t, dropout_prob_t)

	saver = tf.train.Saver()
	sess = tf.Session()
	tf.initialize_all_variables()
	saver.restore(sess, '/home/arya_01/AxisProject/AxisFilter/model/00239_FILTER/model')

	with sess.as_default():
		for _id, fp in enumerate(filepaths):
			_id += 100
			img = cv2.imread(fp, 0)
			img = fit_image_into_frame(img, frame_size=(736/2, 1600/2, 1), random_fill=False, fill_color=255, mode='fit')
			sig_area = img
			# sig_area = fit_image_into_frame(img, frame_size=(736, 1600, 1), random_fill=False, fill_color=255, mode='fit')
		#     sig_area  = img[350:350+260, 800:1600]
			
			# DEL_CODE
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			
			select = []
			for bb_batch in process_extract(sig_area, batch_size=batch_size):
				new_bb_batch = []
				for bb in bb_batch:
					min_x, min_y, max_x, max_y = bb
					extract_data = sig_area[min_y:max_y, min_x: max_x]
					extract_data = fit_image_into_frame(extract_data, frame_size=(128, 128, 1), random_fill=False, fill_color=255, mode='fit')
					new_bb_batch.append(extract_data)
				predict_data = predict.eval(feed_dict={
				input_t: np.asarray(new_bb_batch),
				dropout_prob_t: 1.0})
				# for p, bb in zip( np.argmax(predict_data, axis=1), bb_batch):
				#     if p == 1:
				#         select.append(bb)

				for p, bb in zip( np.argmax(predict_data, axis=1), bb_batch):
					min_x, min_y, max_x, max_y = bb
					if p == 1:
						cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), thickness=3)
						select.append(bb)
					else:
						cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), thickness=3)
		
					# if p == 1:
					# 	cv2.rectangle(img, (800+min_x, 350+min_y), (800+max_x, 350+max_y), (0, 255, 0), thickness=3)
					# 	select.append(bb)
					# else:
					# 	cv2.rectangle(img, (800+min_x, 350+min_y), (800+max_x, 350+max_y), (0, 0, 255), thickness=3)
			cv2.imwrite(os.path.join('tmp/output', str(_id)+'.png'), img)


			# for bb in select:
			#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			#     img = fit_image_into_frame(img, frame_size=(736, 1600, 3), random_fill=False, fill_color=255, mode='fit')
			#     cv2.rectangle(img, (800+min_x, 350+min_y), (800+max_x, 350+max_y), (0, 255, 0), thickness=1)
			#     cv2.imwrite(os.path.join('tmp/output', str(_id)+'.png'), img)