# Music Genres Classification - Building Guide

## Overview
This project provides **two equivalent implementations**:
- **C++** – for high-performance native builds
- **Python** – for scripting and cross-platform workflows

Both implementations expose the same functionality: audio preprocessing, dataset handling, model definition, and training. You can build and run either independently.

## General Requirements
- **Git** – version control
- **CMake ≥ 4.1** – build configuration
- **GCC == 12** – C++20 and `libtorch` compatible compiler
- **Make** or **Ninja** – build system
- **Python ≥ 3.12** – required for Python implementation
- **CUDA == 12.8** – **NVIDIA** GPU acceleration runtime (**CUDA Runtime + cuBLAS/cuDNN**). For setup instructions, see [Cuda Installation](CUDA.md).

## Install Dependencies

To build and run the project, ensure you have the required tools installed:

### Linux (Debian/Ubuntu)
```sh
sudo apt update && sudo apt install -y build-essential cmake curl gcc-12 g++-12 git libaubio-dev libcurl4-openssl-dev libomp-dev libzip-dev make ninja-build nlohmann-json3-dev python3.12 python3.12-dev python3.12-venv
```

### Windows
- Install [Git for Windows](https://git-scm.com/downloads)
- Install [Python3.12+](https://www.python.org/downloads/)
    - Ensure `python` and `pip` are added to the system `PATH`

## Build
### **Clone the Repository**
Clone the project using SSH:
```sh
git clone git@gitlab.com:mrgi23/music-genres-classification.git
cd music-genres-classification
```

### C++
Create a `build` directory, generate build files with **CMake**, and compile:
```sh
mkdir -p build && cd build
cmake -G Ninja ..
ninja -j$(nproc)
```

#### Python Bindings (Optional)

The C++ code can also be wrapped into Python bindings to produce a `.so` extension module.
This is disabled by default. To enable it, pass the `-DBUILD_PYTHON=ON` flag:
```sh
mkdir -p build && cd build
cmake -G Ninja .. -DBUILD_PYTHON=ON
ninja -j$(nproc)
```

This produces the binaries in the `bin` folder:

- C++ Executable: `bin/musicnet`
- Python Module: `bin/musicnet.cpython-<version>-x86_64-linux-gnu.so` (importable in Python with **torch** `2.8.0(+cu128)`)

### Python
Create and activate a virtual environment (optional), and install dependencies:

#### 1. Linux
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. Windows
```sh
python -m venv .venv
source .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The project provides three different entry points, depending on which implementation you want to run:

- **C++ executable** → `bin/musicnet`
- **C++ wrapped in Python (bindings)** → `app/musicnet_cpp.py`
- **Pure Python implementation** → `app/musicnet_py.py`

All three support command-line arguments for controlling training, prediction, and evaluation.
|   **Flag**  |        **Long Form**       | **Description** |
|-------------|----------------------------|-----------------|
|    `-h`     |          `--help`          | Prints usage help and exits. |
|    `-wd`    | `--working-dir` (C++ Only) | Absolute or relative path to the root folder of the project. Must be provided if running program outside of root folder. |
|    `-p`     |        `--predict`         | Path to a `.wav` file to classify into one of the available genres. If not provided, the entire test dataset will be evaluated. |
|    `-f`     |         `--force`          | Forces training of the model. Without this flag, a pretrained model is downloaded automatically from the GitLab package registry. |
|    `-s`     |          `--save`          | Save the model after training (only works if training is performed with `--force`). |

### C++ Executable
```sh
# From root folder
./bin/musicnet --force --save
cd ..
./music-genres-classification/bin/musicnet -wd ./music-genres-classification -p <path_to_wav_file>
```

### C++ via Python bindings
```sh
# From root folder
PYTHONPATH=./bin python app/musicnet_cpp.py --force -p <path_to_wav_file>
```

### Python
```sh
# From root folder
PYTHONPATH=./src/python app/musicnet_py.py
```

## Troubleshooting

|        **Issue**        |    **Possible Fix**   |
|-------------------------|-----------------------|
| `cmake` not found       | Install using `sudo apt install cmake` |
| Compiler errors         | Ensure `g++` version is 12 (`g++ --version`). |
| Python version mismatch | Run `python3.12` explicitly if needed. |
|  Cuda version mismatch  | Ensure `cuda` version is 12.8 |


## Next Steps
For general project information, see the [README](../README.md).

For details on testing and validation methods used in this system, see [Testing & Validation](TESTING.md).