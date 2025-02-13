cmake_minimum_required(VERSION 3.16)
project(Detection)
set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE SOURCE *.cpp)
file(GLOB_RECURSE HEADER *.hpp *.h)

add_library(
        ${PROJECT_NAME}
        STATIC
        ${SOURCE} 
        ${HEADER}
        )

find_package(OpenCV REQUIRED)

find_package(ncnn REQUIRED)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# Link the libraries
target_link_libraries(${PROJECT_NAME}
        ncnn
        ${OpenCV_LIBS}
        )
