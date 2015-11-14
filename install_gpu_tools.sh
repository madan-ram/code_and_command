#!/usr/sh

sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt-get install build-essential

#Install many dependencies, such as support for reading and writing image files, drawing on the screen, some needed tools, other libraries, etc…
sudo apt-get install build-essential libgtk2.0-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev cmake python-dev python-numpy python-tk libtbb-dev libeigen3-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev libqt4-dev libqt4-opengl-dev sphinx-common texlive-latex-extra libv4l-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev default-jdk ant libvtk5-qt4-dev -y

sudo apt-get install zip git -y

# Time to get the OpenCV 2.4.9 source code:
cd ~
wget http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.9/opencv-2.4.9.zip
unzip opencv-2.4.9.zip
rm opencv-2.4.9.zip*
mv opencv-2.4.9 OpenCV
cd OpenCV

mkdir build
cd build
cmake -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_VTK=ON ..
make -j4
sudo make install
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig
echo "export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig" >> ~/.bashrc
echo "export OPENCV_HOME=~/OpenCV" >> ~/.bashrc
. ~/.bashrc
cd ~
# install cuda and toolkit
wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
sudo chmod 775 cuda_6.5.14_linux_64.run

sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev -y
sudo apt-get install linux-generic
sudo init 6

sudo ./cuda_6.5.14_linux_64.run -override -driver -toolkit -samples -silent -verbose
sudo sh -c 'echo "/usr/local/cuda-6.5/lib64" >> /etc/ld.so.conf'
sudo ldconfig
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
. ~/.bashrc
rm cuda_6.5.14_linux_64.run

cd ~

# install caffe
git clone --branch caffe-0.11 https://github.com/NVIDIA/caffe.git
echo "export CAFFE_HOME=~/caffe" >> ~/.bashrc
. ~/.bashrc

sudo apt-get install \
    libgflags-dev libgoogle-glog-dev \
    libopencv-dev \
    libleveldb-dev libsnappy-dev liblmdb-dev libhdf5-serial-dev \
    libprotobuf-dev protobuf-compiler \
    libatlas-base-dev \
    python-dev python-pip python-numpy gfortran -y
sudo apt-get install --no-install-recommends libboost-all-dev -y
cd $CAFFE_HOME
for req in $(cat python/requirements.txt); do sudo pip install $req; done

sudo apt-get install libblas-dev checkinstall -y
sudo apt-get install libblas-doc checkinstall -y 
sudo apt-get install liblapacke-dev checkinstall -y 
sudo apt-get install liblapack-doc checkinstall -y

sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git -y
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ git libatlas3gf-base libatlas-dev -y
sudo apt-get install libboost-all-dev -y

cd $CAFFE_HOME
mkdir build
cd build
cmake ..
make --jobs=4
cd ~

# setup digit
git clone https://github.com/NVIDIA/DIGITS.git digits
echo "export DIGITS_HOME=~/digits" >> ~/.bashrc
. ~/.bashrc

cd $DIGITS_HOME
sudo pip install -r requirements.txt

sudo apt-get install graphviz -y

cd ~