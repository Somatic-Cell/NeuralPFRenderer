@echo off
rmdir /S build
mkdir build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
:: cmake --build . --config Debug --verbose 
cmake --build build --config Release --verbose 