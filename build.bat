:: rmdir /S build
mkdir build
cd build
cmake .. -DOptiX_INSTALL_DIR=$OptiX_INSTALL_DIR -DCUDA_SDK_INSTALL_DIR=$CUDA_PATH_V12_1 -DOptiX_INCLUDE="C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.0.0\include" 
cmake --build . --config Release --verbose
cd ..