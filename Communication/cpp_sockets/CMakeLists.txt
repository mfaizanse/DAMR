cmake_minimum_required(VERSION 3.5)
project(cpp_sockets)

# Cpp standard
set (CMAKE_CXX_STANDARD 14) # c++ 14

# add OpenCV library?
# Building options (before configure files)
find_package(OpenCV REQUIRED) # Transfer images (no need to be 3.3.1)
set (EXTRA_LIBS ${EXTRA_LIBS} ${OpenCV_LIBS})

include_directories (include)

# Create library
add_library(cpp_sockets
        src/client.cpp
        # src/server.cpp  # TODO(oleguer): Write the server
        )
target_link_libraries(cpp_sockets ${EXTRA_LIBS})

# Library installation:
install (TARGETS cpp_sockets DESTINATION bin)
install (FILES include/client.hpp DESTINATION include)

add_executable(${PROJECT_NAME}_01 main.cpp)
target_link_libraries(${PROJECT_NAME}_01 cpp_sockets ${OpenCV_LIBS})