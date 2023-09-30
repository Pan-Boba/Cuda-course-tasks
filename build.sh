#!/bin/bash

rm -rf build/

echo "Creating build directory"
mkdir build && cd build
cmake ../CMakeLists.txt

cmake --build .
