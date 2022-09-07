# About
The denoiser submitted at [Raytracing Camp 8](https://sites.google.com/view/raytracingcamp8/).

# How to build and run
$ git clone --recursive https://github.com/ykozw/rtcamp8_denoiser.git
$ cd rtcamp8_denoiser
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build . --config Release
$ cmake --install .
$ cd ../bin
$ denoiser.exe ../images/color.hdr ../images/albedo.hdr ../images/normal.hdr out.hdr
