# dependences
* opencv(or other libs to decode image to float32, 3-channel tensor)
* cudnn 8.0
* cudart 8.0
* cublas 8.0

# compile
cd textline-server
mkdir build
cd build
cmake ..
make

# output
test: ./test test.png
libtextline.so(contains jni interface, see jni/com_netease_mm_image_jni_OCR.h)

# files description
* jni/ : files about jni interface, built to offer web server
* modelpredict/ : files to infer by crnn
* test/ : files to test the crnn network
