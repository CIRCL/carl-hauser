# Issues not fixed yet ! sudo apt-get install python-opencv

# From sources : 
sudo apt-get install cmake
sudo apt-get install python-devel numpy
sudo apt-get install gcc gcc-c++
sudo apt-get install gtk2-devel
sudo apt-get install libv4l-devel
sudo apt-get install ffmpeg-devel
sudo apt-get install gstreamer-plugins-base-devel
sudo apt-get install libpng-devel
sudo apt-get install libjpeg-turbo-devel
sudo apt-get install jasper-devel
sudo apt-get install openexr-devel
sudo apt-get install libtiff-devel
sudo apt-get install libwebp-devel
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build
cmake ../
make
sudo make install

# Misc 
pip3 install pandas --user
pip3 install opencv-python --user
pip3 install matplotlib --user