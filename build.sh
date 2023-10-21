#!/bin/bash

echo "Setting env variables"
export PATH="$PATH:C:/opencv/build/x64/vc15/bin"

rm -rf build/

echo "Creating build directory"
mkdir build && cd build

cmake ../CMakeLists.txt
cmake --build . --config Release
