cmake_minimum_required(VERSION 3.10)

# Set compiler
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Declare the project.
project(Tests)

# Set ROOT_DIR
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Includes
include(FetchContent)

# Fetch Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
)
FetchContent_MakeAvailable(googletest)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Install Torch
find_package(Torch QUIET)
if (Torch_FOUND)
  message(STATUS "Torch found")
else()
  message(STATUS "Torch not found, setting up...")
  include("${ROOT_DIR}/cmake/SetupTorch.cmake")
  message(STATUS "Torch setup done")
  find_package(Torch REQUIRED)
endif()

# Include Google Test and Google Mock
include_directories(SYSTEM "${gtest_SOURCE_DIR}/include" "${gmock_SOURCE_DIR}/include")

# Include Torch
include_directories(SYSTEM "${TORCH_INCLUDE_DIRS}")

# Include project headers
include_directories("${ROOT_DIR}/inc")

# Collect all source files from src/ directory, excluding main.
file(GLOB CXX_SOURCES "${ROOT_DIR}/src/cpp/*.cpp")
list(REMOVE_ITEM CXX_SOURCES "${ROOT_DIR}/src/cpp/main.cpp")

# Collect all test files from tests/cpp/unit
file(GLOB UNIT_TEST_SOURCES "${ROOT_DIR}/tests/cpp/unit/*.cpp")
file(GLOB INTEGRATION_TEST_SOURCES "${ROOT_DIR}/tests/cpp/integration/*.cpp")

# Create the test executables
add_executable(unitTests "${UNIT_TEST_SOURCES}" "${CXX_SOURCES}")
add_executable(integrationTests "${INTEGRATION_TEST_SOURCES}" "${CXX_SOURCES}")

# Link with Google Test, Torch, etc.
target_link_libraries(unitTests aubio curl gtest gtest_main OpenMP::OpenMP_CXX gmock "${TORCH_LIBRARIES}" pthread zip)
target_link_libraries(integrationTests aubio curl gtest gtest_main gmock OpenMP::OpenMP_CXX "${TORCH_LIBRARIES}" pthread zip)

# Enable code coverage flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping -O0 -g0")

# Create report folder
file(MAKE_DIRECTORY "${ROOT_DIR}/reports/htmlcov/cpp")

# Make unit tests
add_custom_target(
  unit
  COMMAND "./unitTests"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Make integration tests
add_custom_target(
  integration
  COMMAND "./integrationTests"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Make coverage
add_custom_target(
  coverage
  COMMAND LLVM_PROFILE_FILE=coverageUnit.profraw "./unitTests"
  COMMAND LLVM_PROFILE_FILE=coverageIntegration.profraw "./integrationTests"
  COMMAND llvm-profdata merge -sparse coverageUnit.profraw coverageIntegration.profraw -o coverage.profdata
  COMMAND llvm-cov export --object="./unitTests" --object="./integrationTests"  -instr-profile=coverage.profdata -format=lcov --ignore-filename-regex="tests/" > "${ROOT_DIR}/reports/htmlcov/cpp/coverage.lcov"
  COMMAND genhtml "${ROOT_DIR}/reports/htmlcov/cpp/coverage.lcov" --output-directory "${ROOT_DIR}/reports/htmlcov/cpp/"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
