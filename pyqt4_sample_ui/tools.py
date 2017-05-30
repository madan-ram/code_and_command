import os, sys
import errno
import json
from PyQt4.QtGui import QApplication, QPixmap, QMainWindow, QImage, QLabel, QCheckBox, QRadioButton
from PyQt4 import uic
import cv2
import numpy as np

def make_sure_path_exists(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise
			
def numpytoQimage(img, sizeWidth=1000):

	if len(img.shape)==2:
		img = np.uint8(img)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	height, width, bytesPerComponent = img.shape
	if (width>=sizeWidth):
		img=cv2.resize(img,(sizeWidth,int((height*sizeWidth)/width)))

	height, width, bytesPerComponent = img.shape
	bytesPerLine = bytesPerComponent * width
	img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	image = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
	return image

def setImageLabel(widget, img):
	Qimg = numpytoQimage(img)
	# widget.resize(img.shape[1], img.shape[0])
	widget.setPixmap(QPixmap.fromImage(Qimg))

class main_window(QMainWindow, uic.loadUiType("labeltool.ui")[0]):

	def __init__(self, img_list):
		QMainWindow.__init__(self, None)
		self.setupUi(self)

		self.img = None
		
		img = cv2.imread('25thOctfilename470.jpg')
		
		self.output_dir = sys.argv[2]

		self.img_list = img_list
		self.index = 0

		# first time img load
		img = cv2.imread(self.img_list[self.index])
		width, height = img.shape[0], img.shape[1]
		set_max_size=512
		factor = 1.0
		if height>=width:
			factor = set_max_size/float(height)
		else:
			factor = set_max_size/float(width)
		img=cv2.resize(img, tuple(map(int, map(round, [img.shape[1]*factor, img.shape[0]*factor]))) )
		setImageLabel(self.ImageLabel, img)

		# select option read and write
		self.back_img.clicked.connect(self.go_back)
		self.front_img.clicked.connect(self.go_front)
		self.save_file.clicked.connect(self.save_file_func)

	def save_file_func(self):

		result = {}
		file_full_name = os.path.basename(self.img_list[self.index])
		fname_name, file_ext = file_full_name.split('.')

		output_path = os.path.join(self.output_dir, fname_name+'.txt')
		with open(output_path, 'w') as fw:
			result['road_cond'] = []
			for checkbox in self.road_cond.findChildren(QCheckBox):
				if checkbox.isChecked():
					result['road_cond'].append(str(checkbox.text()))

			result['climate'] = []
			for radio in self.climate.findChildren(QRadioButton):
				if radio.isChecked():
					result['climate'].append(str(radio.text()))

			result['when'] = []
			for radio in self.when.findChildren(QRadioButton):
				if radio.isChecked():
					result['when'].append(str(radio.text()))

			print result, fname_name
			fw.write(json.dumps(result))

	def go_front(self):
		if (self.index+1) < len(self.img_list):
			self.index += 1
			img = cv2.imread(self.img_list[self.index])
			width, height = img.shape[0], img.shape[1]
			set_max_size=512
			factor = 1.0
			if height>=width:
				factor = set_max_size/float(height)
			else:
				factor = set_max_size/float(width)
			img=cv2.resize(img, tuple(map(int, map(round, [img.shape[1]*factor, img.shape[0]*factor]))) )

			setImageLabel(self.ImageLabel, img)

	def go_back(self):
		if (self.index-1) > -1:
			self.index -= 1
			img = cv2.imread(self.img_list[self.index])
			width, height = img.shape[0], img.shape[1]
			set_max_size=512
			factor = 1.0
			if height>=width:
				factor = set_max_size/float(height)
			else:
				factor = set_max_size/float(width)
			img=cv2.resize(img, tuple(map(int, map(round, [img.shape[1]*factor, img.shape[0]*factor]))) )

			setImageLabel(self.ImageLabel, img)

if __name__ == '__main__':

	if len(sys.argv) < 3:
		print 'COMMAND: python '+sys.argv[0]+' [PATH_JSON] [OUT_DIR]' 
		sys.exit(-1)

	img_list = []
	data = json.loads(open(sys.argv[1]).read())
	for d in data:
		img_list.append(d['filename'])
	# create output dir
	make_sure_path_exists(sys.argv[2])

	app = QApplication(sys.argv)
	mw = main_window(img_list)
	mw.show()
	app.exec_()