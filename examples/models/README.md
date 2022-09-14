# Nvidia backend implementation notes

The Nvidia backend for DNNL can be exposed to the user via the `dnnl::engine::kind::gpu` engine kind. Currently, for the case when user's system has both Intel and Nvidia GPUs, 
`DNNL_GPU_VENDOR=NVIDIA` Flag is used in CMake, since the devices are clustered based on the device vendor ID and index pattern can not be used to distinguish between Intel GPU and Nvidia GPU.

## Requirements for Building from source

- oneDNN Nvidia backend requires Intel oneAPI DPC++ Compiler, details of aquiring and installing it is available https://github.com/muhammad-tanvir-1211/oneDNN/tree/vgg_sample/README.md 
- git-lfs is required to checkout the build

```
$ git clone https://github.com/muhammad-tanvir-1211/oneDNN.git -b vgg_sample
$ cd oneDNN
```

The Nvidia backend requires 
- Nvidia CUDA driver version  418.87.01 or 440.33.01
- cuBLAS library version  10.1 or 1.2,  
- cuDNN library version 7.6.5.
- `LD_LIBRARY_PATH` should be set to `/Path/to/dpcpp/lib`

To obtain the weights needed for running the models see ```trained_weights/README.md```. These must be obtained prior to building the project as CMake will move them to the correct location when running.

## Commands to run on Nvidia GPU

### Setup

```
$ export CXX=/Path/to/dpcpp/bin/clang++
$ export CC=/Path/to/dpcpp/bin/clang
$ export LD_LIBRARY=/Path/to/dpcpp/lib
```

### Building and Running

```
$ mkdir build
$ cd build
$ cmake -DDNNL_GPU_RUNTIME=DPCPP -DDNNL_GPU_VENDOR=NVIDIA -G Ninja .. -DOPENCLROOT=/Path/to/OpenCL-ICD-Loader -DOPENCLHEADERS=/Path/to/OpenCL-Headers
```

To run VGG-16
```
$ ninja run_vgg
```

## Commands to run on Intel GPU

### OpenCL

```
$ mkdir ocl-build
$ cd ocl-build
$ cmake -DDNNL_GPU_RUNTIME=OCL -G Ninja .. -DOPENCLROOT=/Path/to/OpenCL-ICD-Loader

$ ninja run_vgg
```

### DPCPP / SYCL

```
$ export CXX=/Path/to/dpcpp/bin/clang++
$ export CC=/Path/to/dpcpp/bin/clang
$ export LD_LIBRARY=/Path/to/dpcpp/lib

$ mkdir sycl-build
$ cd sycl-build
$ cmake -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP -G Ninja .. -DOPENCLROOT=/Path/to/OpenCL-ICD-Loader -DOPENCLHEADERS=/Path/to/OpenCL-Headers

$ ninja run_vgg
```