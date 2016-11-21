import os, sys
from glob import glob
import random
import shutil

def split_path(_dir):
	head, tail = os.path.split(_dir)
	if head == '' and tail != '':
		return [tail]
	elif head == '' and tail == '':
		return []
	return split_path(head) + [tail]

if __name__ == '__main__':
	split_dir = sys.argv[1]
	train_split_dir = sys.argv[2]
	valid_split_dir = sys.argv[3]
	train_percent = int(sys.argv[4])
	classes_dir = glob(os.path.join(split_dir, '*'))

	#create class director in train and valid
	for class_dir in classes_dir:
		class_name = split_path(class_dir)[-1]

		new_path = os.path.join(train_split_dir, class_name)
		if not os.path.exists(new_path):
			os.mkdir(new_path)

		new_path = os.path.join(valid_split_dir, class_name)
		if not os.path.exists(new_path):
			os.mkdir(new_path)

	# sample dataset based on split
	for class_dir in classes_dir:
		class_name = split_path(class_dir)[-1]
		for file_path in glob(os.path.join(class_dir, '*')):

			if random.randint(0, 100) <= train_percent:
				shutil.copy2(file_path, os.path.join(train_split_dir, class_name))
			else:
				shutil.copy2(file_path, os.path.join(valid_split_dir, class_name))