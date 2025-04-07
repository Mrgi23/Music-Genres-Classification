# Music Genres Classification - Testing & Validation Guide

## Overview
Testing strategies ensure the accuracy, reliability, and performance of the Music Genres Classification. Testing is divided into unit tests, integration tests, and coverage analysis, ensuring every component functions correctly in isolation and as part of the overall system.

## Testing Frameworks
### C++ Testing
- **Google Test (`gtest`), Google Mock (`gmock`)** – Testing framework for C++
- **LLVM (`llvm-cov`)** – Code coverage analysis

### Python Testing
- **PyTest (`pytest`, `pytest-mock`)** – Testing framework for Python
- **PyTest(`pytest-cov`)** – Code coverage analysis

### Compatibility Notice
For compatibility details, see the [Testing Framework Compatibility](COMPATIBILITY.md)

## Installing Dependencies
Ensure that all required dependencies are installed.

### C++ Dependencies
#### Application Dependencies
- `aubio` – Used for audio analysis and feature extraction.
- `curl` – Used for making HTTP requests, e.g., for downloading datasets or files.
- `zip` – Required for extracting compressed files, such as datasets.

#### Testing Dependencies
- **CMake(`cmake`)** – Build system for compiling tests
- **LLVM (`llvm-cov`)** – Required for C++ code coverage (**only works with `clang`**)

**C++ code coverage is only supported with Clang**
- If using **GCC or MSVC**, code coverage will not be available.
- Windows users must run tests inside **WSL with Clang/LLVM**.

### Python Dependencies
The Python dependencies required for both application and testing are listed in the `requirements.txt` file.

### Installation
#### 1. Linux/WSL
```sh
sudo apt update && sudo apt install -y cmake clang curl make lcov libaubio-dev libcurl4-openssl-dev libzip-dev llvm python3.10 python3.10-dev python3.10-venv

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 2. macOS
```sh
brew update && brew install aubio cmake clang curl make lcov libzip llvm python@3.10

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Unit Testing
Unit testing ensures that individual components of the system function correctly in isolation. Each module is tested independently to verify expected behavior under different conditions.

### Scope of Unit Testing
- Validate **Downloader** functionality to ensure the **GTZAN** dataset is correctly downloaded.
- Test **Preprocessor** for correct feature extraction, including **MFCCs** computations.
- Verify **AudioDataset** handles data correctly, ensuring features are processed and returned properly.
- Confirm that **Model** structure and layers are initialized properly.
- Verify **Trainer**'s functionality by ensuring it correctly initiates training, handles forward/backward passes, and updates weights appropriately.

## Integration Testing
Integration tests verify that the various components of the Music Genres Classification system work together correctly. These tests ensure that data flows properly between modules, that transformations are performed as expected, and that the overall system produces the correct output when given a known input.

### Scope of Integration Testing
- Ensure **Downloader**, **Preprocessor** and **AudioDataset** interact correctly, with data flowing seamlessly through the pipeline.
- Verify the full data processing flow from raw audio input to feature extraction and data loading.
- Test **Model** and **Trainer** interaction, confirming that data from the loader reaches the model and that training occurs as expected.
- Assess **overall performance** by training the model on the full dataset and confirming correct behavior during both training and evaluation.

##  Running Tests
### C++ Tests
```sh
mkdir -p tests/build && cd tests/build
cmake ..
make -j${nproc}
make unit # Unit Tests
make integration # Integration Tests
```

### Python Tests
```sh
source .venv/bin/activate
PYTHONPATH=./src/python pytest tests/python/unit # Unit Tests
PYTHONPATH=./src/python pytest tests/python/integration # Integration Tests
```

### Test File Location
| **Language**              | **Directory**              |
|---------------------------|----------------------------|
| C++ Unit Tests            | `./tests/cpp/unit`           |
| C++ Integration Tests     | `./tests/cpp/integration`    |
| Python Unit Tests         | `./tests/python/unit`        |
| Python Integration Tests  | `./tests/python/integration` |

For detailed test cases, see the corresponding test files in `./tests/<language>/<type>`.

## **Code Coverage**
Code coverage ensures tests sufficiently exercise the codebase, identifying tested portions.

### Generating Coverage Reports
#### C++ Code Coverage
C++ code coverage is generated using **LLVM's `llvm-cov`**, which only works with **Clang**.
**MSVC and MinGW are not supported for code coverage.**
- Linux/macOS: **Native support with Clang**
- Windows: **Must use WSL with Clang/LLVM**
```sh
mkdir -p tests/build && cd tests/build
cmake ..
make -j${nproc}
make coverage
```
#### Python Code Coverage
```sh
source .venv/bin/activate
PYTHONPATH=./src/python pytest --cov=./ --cov-report=html:reports/htmlcov/python tests/python
```

### Coverage Report Locations
| **Language**       | **Directory**            |
|--------------------|--------------------------|
| C++ Unit Tests     | `./reports/htmlcov/cpp`    |
| Python Unit Tests  | `./reports/htmlcov/python` |

To view coverage report:
```sh
firefox ./<directory>/index.html
```

## CI/CD Integration
The testing framework is integrated with **GitLab CI/CD**, ensuring automated testing and coverage reporting on every merge request.

### CI/CD Pipeline Overview
The pipeline is structured into the following stages:
| **Stage**    | **Purpose** |
|--------------|-------------|
| **Setup**    | Builds and pushes a custom Docker image with all dependencies pre-installed (`cmake`, `clang`, `llvm`, `python3.10`, `googletest`, `googlemock`, `pytest`, etc.) |
| **Update**   | Update custom Docker image's Python virtual environment with all dependencies from the `requirements.txt` |
| **Build**    | Compiles C++ tests |
| **Test**     | Runs both C++ (Google Test) and Python (pytest) tests, and generates coverage reports using `llvm-cov` and `pytest-cov` |
| **Deploy**   | Creates new tag based on the release version and publishes reports via **GitLab Pages** (only `main`) |

### When Does CI/CD Run?
The CI/CD pipeline is triggered in the following cases:
- On every merge request to `develop` branch
- On every merge commit to `main` branch
- On manual pipeline execution from **GitLab UI**

**The pipeline is blocked if tests fail**, ensuring only validated code is merged.

### GitLab CI/CD Coverage Badge
![Coverage](https://gitlab.com/mrgi23/music-genres-classification/badges/main/coverage.svg)

### GitLab Pages (Published Reports)
Deployed via GitLab Pages, coverage reports are accessible at:
- [Music Genres Classification - Metrics and Evaluation](https://mrgi23.gitlab.io/music-genres-classification/index.html)
- [C++ Code Coverage Report](https://mrgi23.gitlab.io/music-genres-classification/cpp/index.html)
- [Python Code Coverage Report](https://mrgi23.gitlab.io/music-genres-classification/python/index.html)


## Next Steps
For instructions on building and running the project, see the [Building Guide](BUILD.md).

For a more in-depth look into the **CNN** architecture, see [CNN Architecture](CNN.md).
