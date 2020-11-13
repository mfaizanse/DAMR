rm -r build
sudo rm /home/faizi/programs/anaconda3/envs/plarr/lib/python3.7/site-packages/kinfu_cv.cpython-37m-x86_64-linux-gnu.so
mkdir build
cd build
cmake ..
make
sudo cp kinfu_cv.cpython-37m-x86_64-linux-gnu.so /home/faizi/programs/anaconda3/envs/plarr/lib/python3.7/site-packages/
cd ..