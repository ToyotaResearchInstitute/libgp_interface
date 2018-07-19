cmake_minimum_required(VERSION 2.8)
project(libgp_interface CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(include ${libgp_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED src/libgp_interface.cc)

target_Link_libraries(${PROJECT_NAME} gp)

# Expose libgp_interface's public includes to other subprojects through cache variable.
set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)
