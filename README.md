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
* bin/: the model weights, see textline-traincrnn-tf repo which train the model and export the weights

# improvement
modelpredict/crnn.cpp Line 271~303 -> these codes implement the split and concat feature map, where we first copy feature map from gpu to cpu, and tune the address in cpu, and finally cpoy back to gpu.

This operation cost too much, a way to improve performance is altering the CNN to let the last feature map with width=1 and height=1; then you don't need to copy and alter address.
