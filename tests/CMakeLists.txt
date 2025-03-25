cmake_minimum_required(VERSION 3.10)

# Set compiler
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Declare the project.
project(Tests)

# Includes
include(FetchContent)

# Fetch Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
)
FetchContent_MakeAvailable(googletest)

# Include Google Test and Google Mock
include_directories(SYSTEM "${gtest_SOURCE_DIR}/include" "${gmock_SOURCE_DIR}/include")

# Set ROOT_DIR
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Include project headers
include_directories("${ROOT_DIR}/inc")

# Collect all source files from src/ directory, excluding main.
file(GLOB CXX_SOURCES "${ROOT_DIR}/src/cpp/*.cpp")
list(REMOVE_ITEM CXX_SOURCES "${ROOT_DIR}/src/cpp/main.cpp")

# Collect all test files from tests/cpp/unit
file(GLOB UNIT_TEST_SOURCES "${ROOT_DIR}/tests/cpp/unit/*.cpp")

# Create the test executables
add_executable(unitTests "${UNIT_TEST_SOURCES}" "${CXX_SOURCES}")

# Link with Armadillo, Google Test, LiquidDSP and pthread
target_link_libraries(unitTests aubio curl gtest gtest_main gmock zip pthread)

# Enable code coverage flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping -O0")

# Create report folder
file(MAKE_DIRECTORY "${ROOT_DIR}/reports/htmlcov/cpp")

# Make unit tests
add_custom_target(
  unit
  COMMAND "./unitTests"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Make coverage
add_custom_target(
  coverage
  COMMAND LLVM_PROFILE_FILE=coverageUnit.profraw "./unitTests"
  COMMAND llvm-profdata merge -sparse coverageUnit.profraw -o coverage.profdata
  COMMAND llvm-cov export "./unitTests"  -instr-profile=coverage.profdata -format=lcov > "${ROOT_DIR}/reports/htmlcov/cpp/coverage.lcov"
  COMMAND genhtml "${ROOT_DIR}/reports/htmlcov/cpp/coverage.lcov" --output-directory "${ROOT_DIR}/reports/htmlcov/cpp/"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
