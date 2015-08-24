from utils import *
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# python accuracy_test.py [path to culster_image_dir] [path_to_metadata] 

list_category = []
label_map = {}
dir_list = getImmediateSubdirectories(sys.argv[1])
for sub_dir in dir_list:
	path_sub_dir = sys.argv[1]+'/'+sub_dir
	category_map = {}
	for f in getFiles(path_sub_dir):
		path_file = path_sub_dir + '/'+f
		_id = f.split('.')[0]
		try:	
			fr_metadata = open(sys.argv[2] +_id+ '.json')
			data = json.loads(fr_metadata.read())
		except Exception as e:
			print e, 'id not found', _id
			continue

		try:
			category_map[data['category']] += 1
		except Exception as e:
			category_map[data['category']] = 1
			list_category.append(data['category'])

	label_map[sub_dir] = category_map

set_of_category = list(set(list_category))

category_map = {}
i = 0
for cat in set_of_category:
	category_map[cat] = i
	i += 1

number_of_label = len(dir_list)
number_of_category = len(set_of_category)

confusion_matrix = np.zeros((number_of_category, number_of_label))

for label, data in label_map.items():
	for cat, count in data.items():
		confusion_matrix[category_map[cat]][int(label)] = count

print np.argmax(confusion_matrix, axis=0)
print category_map

# print confusion_matrix.tolist()
# plt.matshow(confusion_matrix)
# plt.savefig('cm.png')