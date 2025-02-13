cmake_minimum_required(VERSION 3.22)
project(detection)

set(CMAKE_CXX_STANDARD 20)

set(DET_DIR ${CMAKE_CURRENT_LIST_DIR}/../../detection)
add_subdirectory(${DET_DIR} DET)
include_directories(${DET_DIR})

add_executable(${PROJECT_NAME} main.cpp)

## Link the libraries
target_link_libraries(${PROJECT_NAME}
        Detection
        ${OpenCV_LIBS}
        )
