cmake_minimum_required(VERSION 2.8)
project( IDcardRecognize )

file( GLOB LIB_SRCS "modelpredict/*.h" )
file( GLOB TEST_SRCS ${LIB_SRCS} "./test/*.cpp" )
file( GLOB JNI_SRCS  ${LIB_SRCS} "./jni/*.cpp" )

SET( OPENCV_PATH "/home/hzqiaohan/.local" ) 
SET( OPENCV_INCLUDE "${OPENCV_PATH}/include" "${OPENCV_PATH}/include/opencv" "${OPENCV_PATH}/include/opencv2" ) 
SET( OPENCV_LIB "${OPENCV_PATH}/lib" ) 

SET( JPEG_PATH "/home/hzqiaohan/turbojpeg" ) 
SET( JPEG_INCLUDE "${JPEG_PATH}/include" ) 
SET( JPEG_LIB "${JPEG_PATH}/lib" ) 

SET( cJSON_INCLUDE "/home/hzqiaohan/.local/include/cjson" ) 
SET( cJSON_LIB "/home/hzqiaohan/.local/lib" ) 

#SET( CAFFE_PATH "/home/hzqiaohan/caffe/caffe-master")
#SET( CAFFE_INC "${CAFFE_PATH}/include")
#SET( CAFFE_LIB "${CAFFE_PATH}/build/lib")

SET( BLAS_PATH "/home/hzqiaohan/.local")
SET( BLAS_INC "${BLAS_PATH}/include")
SET( BLAS_LIB "${BLAS_PATH}/lib")

INCLUDE_DIRECTORIES( ${cJSON_INCLUDE} ${JPEG_INCLUDE} ${OPENCV_INCLUDE} ${BLAS_INC} "modelpredict" )
LINK_DIRECTORIES( ${JPEG_LIB} ${cJSON_LIB} ${OPENCV_LIB} ${BLAS_LIB} "/usr/local/cuda/lib64" )

SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -D__STDC_CONSTANT_MACROS" )

add_executable( test ${TEST_SRCS} )
target_link_libraries( test -lopencv_core -lopencv_highgui -lopencv_imgproc -lopenblas -lopencv_features2d -lopencv_legacy -lcudnn -lcudart -lcublas -lcjson -lturbojpeg -lpthread )
#target_link_libraries( IDrec -lglog -lcaffe -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_legacy -lboost_filesystem -lboost_system)
#add_library( deskew SHARED ${LIB_SRCS}  )

#add_library( textline SHARED ${JNI_SRCS}  )
#target_link_libraries( textline -lcudnn )
