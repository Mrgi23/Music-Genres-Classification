# Music Genres Classification - CUDA Support (C++ Only)

## Overview
**CUDA (Compute Unified Device Architecture)** is **NVIDIA**’s platform for general-purpose GPU computing.
It provides libraries like `libcudart`, `libcublas`, and `libcudnn` that enable fast matrix operations and deep learning primitives.

**CUDA** is only useful on systems with an **NVIDIA** GPU. On CPU-only systems, a build compiled with **CUDA** will still run, but it will automatically fall back to CPU execution. If using GPU, `torch` and `cuda` versions must match with the included one (`libtorch-2.8.0`, `cuda-12.8`).

## Why It’s Needed
- Required for training or inference using **NVIDIA** GPUs.
- Accelerates linear algebra, convolution, and neural network operations.
- Not available on **macOS**, **Raspberry Pi** or any `aarch64` architecture.

## Installation (Linux Ubuntu/Debian)
```sh
# Add NVIDIA’s package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | sed -e 's/\.//')/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA runtime + cuBLAS + cuDNN
sudo apt-get install -y cuda-runtime-12-8 cuda-toolkit-12-8 libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# Make sure CUDA is included in the PATH
echo "export PATH=/usr/local/cuda-12.8/bin:$PATH" > ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH" > ~/.bashrc
source ~/.bashrc

# Match Python version
pip install torch==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```
## Next Steps
For details on testing and validation methods used in this system, see [Testing & Validation](TESTING.md).

For instructions on building and running the project, see the [Building Guide](BUILD.md).