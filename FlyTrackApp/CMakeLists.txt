cmake_minimum_required (VERSION 2.8)
project(flight_tracker)

include(ExternalProject)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations")
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Adding customized cmake module for building boost
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Find default python libraries and interpreter
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(PNG REQUIRED)
include(BuildBoost) # Custom module

message(STATUS "Boost_INCLUDE_DIR = " ${Boost_INCLUDE_DIR})

include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS} ${PNG_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Build and link the pylib module
add_library(FlightTracker_lib SHARED FlightTracker_lib.cpp flight_tracker_class.cpp frame_loader.cpp focal_grid.cpp model_class.cpp multi_body_icp.cpp image_segmentation.cpp find_initial_state.cpp)
target_link_libraries(FlightTracker_lib Threads::Threads)
target_link_libraries(FlightTracker_lib ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${PCL_LIBRARIES} ${PNG_LIBRARY})
add_dependencies(FlightTracker_lib Boost)

# Tweaks the name of the library to match what Python expects
set_target_properties(FlightTracker_lib PROPERTIES SUFFIX .so)
set_target_properties(FlightTracker_lib PROPERTIES PREFIX "")

#add_executable(flight_tracker ImageProcessClass.cpp)
