# run python create_data.py <source-dir> <number_of_sample_from_same_dir> <number_of_sample_from_other_dir> <destination-dir> <type of db (lmdb|image) default->image> <batch size-per-iteration reduce it if you have less RAM>
# eg python create_data.py ./data/ 2 5 new_data/ lmdb 1000
from utils import *
import sys, os
import random
import cv2
import PIL
import hashlib
import multiprocessing
import threading
from multiprocessing import Pool, Queue
from multiprocessing.pool import ThreadPool
import lmdb
import caffe
import time

def print_log(st, se, message):
    print se -st , ':', message

def remove_by_index(a, index):
    return a[:index] + a[index+1 :]

def flatten(l):
    return reduce(lambda x,y: x + y,l)

def get_files_path(dir_path):
    image_files_name = getFiles(os.path.join(dir_path, 'face'))
    image_files_path = [os.path.join(dir_path, 'face', fname) for fname in image_files_name]
    return image_files_path

# define custom exception
class AnchorImReadFailed(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class OtherImReadFailed(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

def _write_batch_lmdb(db, batch):
    """
    Write a batch to an LMDB database
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for i, temp in enumerate(batch):
                datum, _id = temp
                key = str(_id)
                lmdb_txn.put(key, datum.SerializeToString())

    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit*2
        print('Doubling LMDB map size to %sMB ...' % (new_limit>>20,))
        try:
            db.set_mapsize(new_limit) # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0,87):
                raise Error('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
            else:
                raise e
        # try again
        _write_batch_lmdb(db, batch)

def create_data(args):

    anchor_image_path, other_image_path, label, _id, anchor_image_class, image_list, label_list = args

    try:
        anchor_image = np.asarray(PIL.Image.open(anchor_image_path), dtype='uint8')
        # anchor_image = caffe.io.load_image(anchor_image_path)
    except Exception as e:
        raise AnchorImReadFailed({"message": "anchor image read failed"})

    try:
        other_image = np.asarray(PIL.Image.open(other_image_path), dtype='uint8')
        # other_image = caffe.io.load_image(other_image_path)
    except Exception as e:
        raise OtherImReadFailed({"message": "other image read failed"})

    anchor_image = create_fixed_image_shape(anchor_image,  frame_size=(256, 256, 3), random_fill=False, mode='fit')
    other_image = create_fixed_image_shape(other_image,  frame_size=(256, 256, 3), random_fill=False, mode='fit')
    result = np.hstack((anchor_image, other_image))

    try:
        DB_TYPE = sys.argv[5]
    except Exception as e:
        DB_TYPE = "image"
        print "default creating image files as db"

    if DB_TYPE == "image":
        # convert from rgb to bgr to write into file
        result = result[:, :, ::-1]
        random_string = anchor_image_path + other_image_path
        cv2.imwrite(os.path.join(sys.argv[4], str(label), hashlib.md5(random_string).hexdigest()+'.jpg'), result)
    elif DB_TYPE == "lmdb":
        image_datum = caffe.proto.caffe_pb2.Datum()
        image_datum.height = result.shape[0]
        image_datum.width = result.shape[1]
        image_datum.channels = result.shape[2]
        image_datum.data = result.tostring()
        image_datum.label = int(anchor_image_class)
        image_list.append((image_datum, _id))
        
        label_datum = caffe.proto.caffe_pb2.Datum()
        label_datum.float_data.extend(np.array([label]).flat)
        label_datum.channels, label_datum.height, label_datum.width = 1, 1, 1
        label_list.append((label_datum, _id))


try:
    DB_TYPE = sys.argv[5]
except Exception as e:
    DB_TYPE = "image"
    print "default creating image files as db"

if DB_TYPE == "lmdb":
    image_db = lmdb.open(sys.argv[4]+'_image', map_size=1e+12)
    label_db = lmdb.open(sys.argv[4]+'_label', map_size=1e+12)

# read all the directory{class directory}
dirs = getImmediateSubdirectories(sys.argv[1])

num_of_batch = len(dirs)

print 'number of cpu used', multiprocessing.cpu_count()

try:
    batch_size = int(sys.argv[6])
except Exception as e:
    batch_size = 1000
    print "default batch_size is set to", batch_size

# reset for new batch of data
data_combination_with_label = []
manager = multiprocessing.Manager()
image_list = manager.list()
label_list = manager.list()

_id = 0
for class_label in xrange(num_of_batch):
    _dir = dirs[class_label]
    _dir_other = remove_by_index(dirs, class_label)
    dir_path = os.path.join(sys.argv[1], _dir)
    dir_path_other = [os.path.join(sys.argv[1], _dir_name) for _dir_name in _dir_other]
    dir_image_files_path = get_files_path(dir_path)
    dir_image_files_path_other = flatten([get_files_path(dir_path) for dir_path in dir_path_other])

    st = time.time()
    for y in xrange(len(dir_image_files_path)):
        try:
            anchor_image_path = dir_image_files_path[y]
            select_random_image_same_dir = random.sample(remove_by_index(dir_image_files_path, y), int(sys.argv[2]))
            select_random_image_other_dir = random.sample(dir_image_files_path_other, int(sys.argv[3]))
            [data_combination_with_label.append((anchor_image_path, path, 1, _id, class_label, image_list, label_list)) for path in select_random_image_same_dir]
            _id += 1
            [data_combination_with_label.append((anchor_image_path, path, 0, _id, class_label, image_list, label_list)) for path in select_random_image_other_dir]
            _id += 1
        except AnchorImReadFailed as e:
            print e
            continue
        except OtherImReadFailed as e:
            print e
            pass
        except Exception as e:
            print e
            pass

    if len(data_combination_with_label) >= batch_size:
        se = time.time()
        message = "created list of file to be read", len(data_combination_with_label)
        print_log(st, se, message)

        st = time.time()
        try:
            # create threading
            p = ThreadPool(multiprocessing.cpu_count())
            p.map(create_data, data_combination_with_label)
            p.close()
            p.join()
        except Exception as e:
            print e

        se = time.time()
        message = "completed image reading and proccessing", len(data_combination_with_label)
        print_log(st, se, message)

        st = time.time()
        if DB_TYPE == "lmdb":
            _write_batch_lmdb(image_db, image_list)
            _write_batch_lmdb(label_db, label_list)
        se = time.time()
        message = "completed writing batch data to "+ DB_TYPE
        print_log(st, se, message)

        percentage_completed = (float(class_label+1)/num_of_batch * 100)
        print 'percentage of completed', percentage_completed

        # reset for new batch of data
        del data_combination_with_label[:]
        manager = multiprocessing.Manager()
        image_list = manager.list()
        label_list = manager.list()

if DB_TYPE == "lmdb":
    label_db.close()
    image_db.close()