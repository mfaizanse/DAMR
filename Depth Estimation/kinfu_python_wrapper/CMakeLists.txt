# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2019 Intel Corporation. All Rights Reserved.
cmake_minimum_required(VERSION 3.1.0)

project(kinfu_cv)

find_package(OpenCV REQUIRED) # Transfer images (no need to be 3.3.1)
set (EXTRA_LIBS ${EXTRA_LIBS} ${OpenCV_LIBS})

# Enable C++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


# pybind
add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} module.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module ${OpenCV_LIBS})