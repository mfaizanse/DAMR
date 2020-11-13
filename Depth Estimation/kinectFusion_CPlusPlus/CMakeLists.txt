# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
cmake_minimum_required(VERSION 3.1.0)

project(kinfu_cv)
#
## Find librealsense2 installed package
find_package(realsense2 REQUIRED)

find_package(OpenCV REQUIRED) # Transfer images (no need to be 3.3.1)
set (EXTRA_LIBS ${EXTRA_LIBS} ${OpenCV_LIBS})


# Enable C++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add the application sources to the target
add_executable(${PROJECT_NAME} lib2.cpp)

# Link librealsense2 to the target
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${realsense2_LIBRARY})