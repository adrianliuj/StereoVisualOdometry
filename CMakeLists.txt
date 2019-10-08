cmake_minimum_required( VERSION 2.8 )
project ( myslam )

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "DEBUG" )
set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O0" )

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

############### dependencies ######################
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
# Eigen
include_directories( "/usr/include/eigen3" )
# OpenCV
find_package( OpenCV 3.4.2 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
#Pangolin
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
#Ceres
find_package(Ceres REQUIRED)
message("CERES_LIBRARIES = ${CERES_LIBRARIES}")
include_directories(${CERES_INCLUDE_DIRS})

set( THIRD_PARTY_LIBS 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${CERES_LIBRARIES}
#    ${CSPARSE_LIBRARY}
    opencv_cudabgsegm
    opencv_cudaobjdetect
    opencv_cudastereo
    opencv_cudafeatures2d
    opencv_cudacodec
    opencv_cudaoptflow
    opencv_cudalegacy
    opencv_cudawarping
    opencv_cudaimgproc
    opencv_cudafilters
)

############### source and test ######################
include_directories( ${PROJECT_SOURCE_DIR}/include )
add_subdirectory( src )
add_subdirectory( test )
