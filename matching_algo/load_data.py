import os, sys
import numpy as np
import cv2
import glob
import threading
from Queue import Queue
from utils import get_files, fit_image_into_frame
import random
import noimg
import imghdr
import math

def fixed_signature_extract(img):
	fixed_pos = (360, 1120, 600, 1530)
	img = fit_image_into_frame(img, frame_size=(720, 1600, 1), random_fill=False, fill_color=255, mode='fit')
	extracted_data = img[fixed_pos[0]:fixed_pos[2], fixed_pos[1]:fixed_pos[3]]
	return extracted_data

def load_binary_img_from_path(batch_of_path):
    anchore_img_list = []
    test_img_list = []
    labels = []
    for anchore_img_path, test_img_path, label in batch_of_path:
        anchore_img = cv2.imread(anchore_img_path, 0)
        # SINCE IMAGE IS CHEQUE EXTRACT SIGNATURE
        anchore_img = fixed_signature_extract(anchore_img)
        anchore_img = fit_image_into_frame(anchore_img, frame_size=(224, 224, 1), random_fill=False, fill_color=255, mode='fit')

        test_img = cv2.imread(test_img_path, 0)
        test_img = fit_image_into_frame(test_img, frame_size=(224, 224, 1), random_fill=False, fill_color=255, mode='fit')

        anchore_img_list.append(anchore_img)
        test_img_list.append(test_img)
        labels.append(label)
    return anchore_img_list, test_img_list, labels

def load_triplet_img_from_path(batch_of_path):
    anchore_img_list = []
    positive_img_list = []
    negative_img_list = []
    for anchore_img_path, positive_img_path, negative_img_path in batch_of_path:
        anchore_img = cv2.imread(anchore_img_path, 0)
        # SINCE IMAGE IS CHEQUE EXTRACT SIGNATURE
        anchore_img = fixed_signature_extract(anchore_img)
        anchore_img = fit_image_into_frame(anchore_img, frame_size=(224, 224, 1), random_fill=False, fill_color=255, mode='fit')

        positive_img = cv2.imread(positive_img_path, 0)
        positive_img = fit_image_into_frame(positive_img, frame_size=(224, 224, 1), random_fill=False, fill_color=255, mode='fit')

        negative_img = cv2.imread(negative_img_path, 0)
        negative_img = fit_image_into_frame(negative_img, frame_size=(224, 224, 1), random_fill=False, fill_color=255, mode='fit')

        anchore_img_list.append(anchore_img)
        positive_img_list.append(positive_img)
        negative_img_list.append(negative_img)
    return anchore_img_list, positive_img_list, negative_img_list

class TripletGenerator:

	def __init__(self, cheque_path, finicale_path, train_valid_split_percent=0.75):
		print 'creating generator ...'
		self.selected_path = []
		cheque_img_paths = glob.glob(os.path.join(cheque_path, '*', 'valid', 'FrontBW', '*'))
		for cheque_img_path in cheque_img_paths:
			session_date = cheque_img_path.split('/')[-4]
			fname = cheque_img_path.split('/')[-1]
			_id = fname.split('.')[0]
			finicale_img_paths = glob.glob(os.path.join(finicale_path, session_date, 'valid', _id, '*'))

			new_finicale_img_paths = []
			for p in finicale_img_paths:
				if imghdr.what(p) is not None:
					new_finicale_img_paths.append(p)
				else:
					print 'FAILED path:', p

			finicale_img_paths = new_finicale_img_paths
			if len(finicale_img_paths) > 0:
				self.selected_path.append([cheque_img_path, finicale_img_paths])

		train_split = int(math.ceil(len(self.selected_path)*train_valid_split_percent))
		self.selected_path_train = self.selected_path[:train_split]
		self.selected_path_valid = self.selected_path[train_split:]

		print "done loading file paths ..."
		print "total number of file path", len(self.selected_path)
		print "total number of file path for training", len(self.selected_path_train)
		print "total number of file path for validation", len(self.selected_path_valid)

	def train_data(self, batch_size=32):
		number_of_path = len(self.selected_path_train)
		num_batches =  int(math.ceil(number_of_path/float(batch_size)))
		counter = 0
		while True:
			temp = self.load_signature_triplet_path(self.selected_path_train, batch_size=batch_size)
			counter += 1
			if counter%num_batches == 0:
				break
			yield(load_triplet_img_from_path(temp))

	def valid_data(self, batch_size=32):
		number_of_path = len(self.selected_path_valid)
		num_batches =  int(math.ceil(number_of_path/float(batch_size)))
		counter = 0
		while True:
			temp = self.load_signature_triplet_path(self.selected_path_valid, batch_size=batch_size)
			counter += 1
			if counter%num_batches == 0:
				break
			yield(load_triplet_img_from_path(temp))

	def load_signature_triplet_path(self, selected_path, batch_size=32):
		sample_size = batch_size*2
		sample_ids = random.sample(xrange(len(selected_path)), sample_size)
		# get anchor path
		temp = []
		for index_id, neg_index_id in zip(sample_ids[:batch_size], sample_ids[batch_size:]):
			anchore_img_path = selected_path[index_id][0]
			positive_img_path = random.choice(selected_path[index_id][1])
			negative_img_path = random.choice(selected_path[neg_index_id][1])
			temp.append((anchore_img_path, positive_img_path, negative_img_path))
		return temp

class BinaryGenerator:

	def __init__(self, cheque_path, finicale_path, train_valid_split_percent=0.75):
		print 'creating generator ...'
		self.selected_path = []
		cheque_img_paths = glob.glob(os.path.join(cheque_path, '*', 'valid', 'FrontBW', '*'))
		for cheque_img_path in cheque_img_paths:
			session_date = cheque_img_path.split('/')[-4]
			fname = cheque_img_path.split('/')[-1]
			_id = fname.split('.')[0]
			finicale_img_paths = glob.glob(os.path.join(finicale_path, session_date, 'valid', _id, '*'))

			new_finicale_img_paths = []
			for p in finicale_img_paths:
				if imghdr.what(p) is not None:
					new_finicale_img_paths.append(p)
				else:
					print 'FAILED path:', p

			finicale_img_paths = new_finicale_img_paths
			if len(finicale_img_paths) > 0:
				self.selected_path.append([cheque_img_path, finicale_img_paths])

		train_split = int(math.ceil(len(self.selected_path)*train_valid_split_percent))
		self.selected_path_train = self.selected_path[:train_split]
		self.selected_path_valid = self.selected_path[train_split:]

		print "done loading file paths ..."
		print "total number of file path", len(self.selected_path)
		print "total number of file path for training", len(self.selected_path_train)
		print "total number of file path for validation", len(self.selected_path_valid)

	def train_data(self, batch_size=32):
		number_of_path = len(self.selected_path_train)
		num_batches =  int(math.ceil(number_of_path/float(batch_size)))
		counter = 0
		while True:
			temp = self.load_signature_binary_path(self.selected_path_train, batch_size=batch_size)
			counter += 1
			if counter%num_batches == 0:
				break
			yield(load_binary_img_from_path(temp))

	def valid_data(self, batch_size=32):
		number_of_path = len(self.selected_path_valid)
		num_batches =  int(math.ceil(number_of_path/float(batch_size)))
		counter = 0
		while True:
			temp = self.load_signature_binary_path(self.selected_path_valid, batch_size=batch_size)
			counter += 1
			if counter%num_batches == 0:
				break
			yield(load_binary_img_from_path(temp))

	def load_signature_binary_path(self, selected_path, batch_size=32):
		sample_size = batch_size*2
		sample_ids = random.sample(xrange(len(selected_path)), sample_size)
		# get anchor path
		temp = []
		for index_id, neg_index_id in zip(sample_ids[:batch_size], sample_ids[batch_size:]):
			anchore_img_path = selected_path[index_id][0]
			positive_img_path = random.choice(selected_path[index_id][1])
			negative_img_path = random.choice(selected_path[neg_index_id][1])
			temp.append((anchore_img_path, positive_img_path, 0))
			temp.append((anchore_img_path, negative_img_path, 1))

		# random.shuffle(temp)
		return temp[:batch_size]

if __name__ == '__main__':
	cheque_path = '/home/aipocuser/Axis_Project/code/signature_training/output/cheque_t'
	finicale_path = '/home/aipocuser/Axis_Project/code/signature_training/output/finicale_t'
	triplet_gen = TripletGenerator(cheque_path, finicale_path, train_valid_split_percent=0.8)
	for x in triplet_gen.train_data(batch_size=32):
		anchore_img_list, positive_img_list, negative_img_list = x
		counter = 0
		for a, p, n in zip(anchore_img_list, positive_img_list, negative_img_list):
			cv2.imwrite('tmp/train/anchore/'+str(counter)+'.png', a)
			cv2.imwrite('tmp/train/positive/'+str(counter)+'.png', p)
			cv2.imwrite('tmp/train/negative/'+str(counter)+'.png', n)
			counter += 1
		break

	for x in triplet_gen.valid_data(batch_size=32):
		anchore_img_list, positive_img_list, negative_img_list = x
		counter = 0
		for a, p, n in zip(anchore_img_list, positive_img_list, negative_img_list):
			cv2.imwrite('tmp/valid/anchore/'+str(counter)+'.png', a)
			cv2.imwrite('tmp/valid/positive/'+str(counter)+'.png', p)
			cv2.imwrite('tmp/valid/negative/'+str(counter)+'.png', n)
			counter += 1
		break

	binary_gen = BinaryGenerator(cheque_path, finicale_path, train_valid_split_percent=0.8)
	for x in binary_gen.train_data(batch_size=32):
		anchore_img_list, test_img_list, labels = x
		counter = 0
		for a, t, l in zip(anchore_img_list, test_img_list, labels):
			if l == 0:
				cv2.imwrite('tmp/train/anchore/'+str(counter)+'.png', a)
				cv2.imwrite('tmp/train/positive/'+str(counter)+'.png', t)
			else:
				cv2.imwrite('tmp/train/anchore/'+str(counter)+'.png', a)
				cv2.imwrite('tmp/train/negative/'+str(counter)+'.png', t)
			counter += 1
		break

	for x in binary_gen.valid_data(batch_size=32):
		anchore_img_list, test_img_list, labels = x
		counter = 0
		for a, t, l in zip(anchore_img_list, test_img_list, labels):
			if l == 0:
				cv2.imwrite('tmp/valid/anchore/'+str(counter)+'.png', a)
				cv2.imwrite('tmp/valid/positive/'+str(counter)+'.png', t)
			else:
				cv2.imwrite('tmp/valid/anchore/'+str(counter)+'.png', a)
				cv2.imwrite('tmp/valid/negative/'+str(counter)+'.png', t)
			counter += 1
		break