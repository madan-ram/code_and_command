import os, sys
import numpy as np
import cv2
from glob import glob
from utils import get_files, fit_image_into_frame
import random
import math
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from num2words import num2words

def compute_skew(image):
    image = cv2.bitwise_not(image)
    height, width = image.shape

    edges = cv2.Canny(image, 150, 200, 3, 5)
    lines = cv2.HoughLinesP(edges, 1, cv2.cv.CV_PI/180, 100, minLineLength=width / 2.0, maxLineGap=20)
    angle = 0.0
    if lines is None:
        return 0
    nlines = lines.size
    for x1, y1, x2, y2 in lines[0]:
        angle += np.arctan2(y2 - y1, x2 - x1)
    return angle / nlines


def deskew(image, angle):
    angle = np.math.degrees(angle)
    # image = cv2.bitwise_not(image)
    non_zero_pixels = cv2.findNonZero(image)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = image.shape[:2]
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    return cv2.getRectSubPix(rotated, (cols, rows), center)

class AmountInWordGenerator:

    def __init__(self, n_chars, num_classes=10, frame_shape = (128, 512, 1), num_samples=10000, train_valid_split_percent=0.75, inverted_amount=True, include_decimal=False):

        self.train_valid_split_percent = train_valid_split_percent
        self.n_chars = n_chars
        self.num_samples = num_samples
        self.num_train_batch = None
        self.num_valid_batch = None
        self.num_classes = num_classes
        self.frame_shape = frame_shape
        self.class_distribution = {}
        self.font_path_list = glob('/home/arya_01/AxisProject/word_to_num_match/generate_text_img/handwriten_fonts/*/*.ttf')

    def preprocess(self, img):
        angle = compute_skew(img)
        img = deskew(img.copy(), angle)
        blur = cv2.GaussianBlur(img,(3,3),0)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th = np.expand_dims(th, axis=2)
        return th

    def generate_random_img(self, num, font_size):
        font = ImageFont.truetype(random.choice(self.font_path_list), font_size)
        text_data = num2words(num, lang='en_IN')
        text_data = text_data.replace("-", " ")
        text_data = text_data.replace(",", "")
        width, height = font.getsize(text_data)
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text_data, (0,0,0), font=font)
        img = cv2.cvtColor( np.asarray(img), cv2.COLOR_RGB2GRAY)
        return img

    def random_with_N_digits(self, n):
        range_start = 10**(n-1)
        range_end = (10**n)-1
        return random.randint(range_start, range_end)

    def get_amount_in_word(self, preprocess=True):
        random_size = random.randint(3, self.n_chars)
        rand_num = self.random_with_N_digits(random_size)

        extract_data = self.generate_random_img(int(rand_num), 50)
        extract_data = fit_image_into_frame(extract_data, frame_size=self.frame_shape, random_fill=False, fill_color=255, mode='fit')
        if preprocess:
            extract_data = self.preprocess(extract_data)
        return extract_data, str(rand_num)

    def train_data(self, batch_size=32):
        # inputs, length_labels, number_labels, mask_matrixs
        num_batches = int(math.floor((self.num_samples*self.train_valid_split_percent)/float(batch_size)))
        self.num_train_batch = num_batches
        for batch_id in xrange(num_batches):
            input_list = []
            length_label_list = []
            number_labels = np.zeros((self.n_chars, batch_size, self.num_classes), dtype=np.float32)
            mask_matrix = np.zeros((self.n_chars, batch_size), dtype=np.bool)

            for index in xrange(batch_size):
                input_img, instrument_amount = self.get_amount_in_word()
                length_label = len(instrument_amount)
                input_list.append(input_img)
                length_label_list.append(length_label)
                for nid, n in enumerate(instrument_amount):
                    number_labels[nid][index][int(n)] = 1.0
                    mask_matrix[nid][index] = True

            yield(input_list, length_label_list, number_labels, mask_matrix)

    def valid_data(self, batch_size=32):
        # inputs, length_labels, number_labels, mask_matrixs
        num_batches = int(math.floor((self.num_samples*(1.-self.train_valid_split_percent))/float(batch_size)))
        self.num_valid_batch = num_batches
        for batch_id in xrange(num_batches):
            input_list = []
            length_label_list = []
            number_labels = np.zeros((self.n_chars, batch_size, self.num_classes), dtype=np.float32)
            mask_matrix = np.zeros((self.n_chars, batch_size), dtype=np.bool)

            for index in xrange(batch_size):
                input_img, instrument_amount = self.get_amount_in_word()
                length_label = len(instrument_amount)
                input_list.append(input_img)
                length_label_list.append(length_label)
                for nid, n in enumerate(instrument_amount):
                    number_labels[nid][index][int(n)] = 1.0
                    mask_matrix[nid][index] = True

            yield(input_list, length_label_list, number_labels, mask_matrix)


class AmountInWordMultiGenerator:

    def __init__(self, n_chars, multi_labels, frame_shape = (128, 512, 1), num_samples=10000, train_valid_split_percent=0.75, inverted_amount=True, include_decimal=False):

        self.multi_labels = multi_labels
        self.train_valid_split_percent = train_valid_split_percent
        self.n_chars = n_chars
        self.num_samples = num_samples
        self.num_train_batch = None
        self.num_valid_batch = None
        self.frame_shape = frame_shape
        self.class_distribution = {}
        self.font_path_list = glob('/home/arya_01/AxisProject/word_to_num_match/generate_text_img/handwriten_fonts/*/*.ttf')

    def preprocess(self, img):
        angle = compute_skew(img)
        img = deskew(img.copy(), angle)
        blur = cv2.GaussianBlur(img,(3,3),0)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th = np.expand_dims(th, axis=2)
        return th

    def generate_num2word(self, num):
        text_data = num2words(num, lang='en_IN')
        text_data = text_data.replace("-", " ")
        text_data = text_data.replace(",", "")
        return text_data

    def generate_random_img(self, num, font_size):
        font = ImageFont.truetype(random.choice(self.font_path_list), font_size)
        text_data = self.generate_num2word(num)
        width, height = font.getsize(text_data)
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text_data, (0,0,0), font=font)
        img = cv2.cvtColor( np.asarray(img), cv2.COLOR_RGB2GRAY)
        return img

    def random_with_N_digits(self, n):
        range_start = 10**(n-1)
        range_end = (10**n)-1
        return random.randint(range_start, range_end)

    def get_amount_in_word(self, preprocess=True):
        random_size = random.randint(3, self.n_chars)
        rand_num = self.random_with_N_digits(random_size)

        extract_data = self.generate_random_img(int(rand_num), 50)
        extract_data = fit_image_into_frame(extract_data, frame_size=self.frame_shape, random_fill=False, fill_color=255, mode='fit')
        if preprocess:
            extract_data = self.preprocess(extract_data)
        return extract_data, str(rand_num)

    def train_data(self, batch_size=32):
        # inputs, length_labels, number_labels, mask_matrixs
        num_batches = int(math.floor((self.num_samples*self.train_valid_split_percent)/float(batch_size)))
        self.num_train_batch = num_batches
        for batch_id in xrange(num_batches):
            input_list = []
            length_label_list = []
            multi_labels = np.zeros((batch_size, len(self.multi_labels), 2), dtype=np.float32)

            for index in xrange(batch_size):
                input_img, instrument_amount = self.get_amount_in_word()
                input_list.append(input_img)

                length_label = len(instrument_amount)
                length_label_list.append(length_label)

                text_data = self.generate_num2word(int(instrument_amount))
                text_data_list = text_data.split()

                for _id, w in enumerate(self.multi_labels):
                    if w in text_data_list:
                        multi_labels[index, _id, 1] = 1.0
                    else:
                        multi_labels[index, _id, 0] = 1.0

            yield(input_list, length_label_list, multi_labels)

    def valid_data(self, batch_size=32):
        # inputs, length_labels, number_labels, mask_matrixs
        num_batches = int(math.floor((self.num_samples*(1.-self.train_valid_split_percent))/float(batch_size)))
        self.num_valid_batch = num_batches
        for batch_id in xrange(num_batches):
            input_list = []
            length_label_list = []
            multi_labels = np.zeros((batch_size, len(self.multi_labels), 2), dtype=np.float32)

            for index in xrange(batch_size):
                input_img, instrument_amount = self.get_amount_in_word()
                input_list.append(input_img)

                length_label = len(instrument_amount)
                length_label_list.append(length_label)

                text_data = self.generate_num2word(int(instrument_amount))
                text_data_list = text_data.split()
                for _id, w in enumerate(self.multi_labels):
                    if w in text_data_list:
                        multi_labels[index, _id, 1] = 1.0
                    else:
                        multi_labels[index, _id, 0] = 1.0

            yield(input_list, length_label_list, multi_labels)

if __name__ == '__main__':
    # amount_in_word_gen = AmountInWordGenerator(7, 10, frame_shape = (128, 896, 1), train_valid_split_percent=0.75)
    # for iter_id, data in enumerate(amount_in_word_gen.valid_data(batch_size=32)):
    #     input_list, length_label_list, number_labels, mask_matrix = data
    #     for counter, img in enumerate(input_list):
    #         cv2.imwrite('tmp/img/'+str(iter_id)+'_'+str(counter)+'.png', img)
    #         print length_label_list[counter],

    multi_lables = ["crore", "lakh", "thousand", "hundred", "ninety", "eighty", "seventy", "sixty", 
        "fifty", "forty", "thirty", "twenty", "nineteen", "eighteen", "seventeen", "sixteen", "fifteen", 
        "fourteen", "thirteen", "twelve", "eleven", "ten", "nine", "eight" , "seven", "six", "five", "four", 
        "three", "two", "one", "zero"]
        
    amount_in_word_gen = AmountInWordMultiGenerator(7, multi_lables, frame_shape = (128, 896, 1), train_valid_split_percent=0.75)
    for iter_id, data in enumerate(amount_in_word_gen.train_data(batch_size=32)):
        input_list, length_label_list, multi_labels = data
        # for counter, img in enumerate(input_list):
        #     cv2.imwrite('tmp/img/'+str(iter_id)+'_'+str(counter)+'.png', img)
        #     print length_label_list[counter], 


        # np.set_printoptions(suppress=False)
        # with open('tmp/txt/number_labels.txt', 'w') as fw:
        #     fw.write(np.array_str(np.transpose(number_labels, (1, 0, 2))))

        # with open('tmp/txt/mask_matrix.txt', 'w') as fw:
        #     fw.write(np.array_str(np.transpose(mask_matrix, (1, 0))))

        # if iter_id == 0:
        #     break
