cmake_minimum_required(VERSION 3.7)
project(sparse_gp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O2")

set(EXECUTABLE_OUTPUT_PATH .)

include_directories(../external/Eigen)

add_executable(sgp
    sgp.cpp
    kernel.h
)