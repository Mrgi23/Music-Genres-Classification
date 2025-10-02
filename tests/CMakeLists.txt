# ===== Setup =====
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== Projecct =====
project(Tests)

# ===== Root Directory =====
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(TESTS_DIR "${ROOT_DIR}/tests/cpp")

# ===== Dependencies =====
# ----- Google Test & Google Mock -----
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
)
FetchContent_MakeAvailable(googletest)

# ----- OpenMP -----
find_package(OpenMP REQUIRED)

# ----- Torch -----
find_package(Torch QUIET)
if (Torch_FOUND)
  message(STATUS "Torch found")
else()
  message(STATUS "Torch not found, setting up...")
  include("${ROOT_DIR}/cmake/SetupTorch.cmake")
  message(STATUS "Torch setup done")
  find_package(Torch REQUIRED)
endif()

# ===== Sources =====
file(GLOB CXX_SOURCES "${ROOT_DIR}/src/cpp/*.cpp")

# ===== Targets =====
# ----- Unit Tests -----
file(GLOB UNIT_TEST_SOURCES "${TESTS_DIR}/unit/*.cpp")
add_executable(unitTests "${CXX_SOURCES}" "${UNIT_TEST_SOURCES}")

target_include_directories(
  unitTests PRIVATE
  "${ROOT_DIR}/inc"
  "${gtest_SOURCE_DIR}/include"
  "${gmock_SOURCE_DIR}/include"
  "${TORCH_INCLUDE_DIRS}"
)

target_link_libraries(
  unitTests PRIVATE
  aubio
  curl
  gtest
  gtest_main
  gmock
  OpenMP::OpenMP_CXX
  pthread
  "${TORCH_LIBRARIES}"
  zip
)

add_custom_target(
  unit
  COMMAND "./unitTests"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ----- Integration Tests -----
file(GLOB INTEGRATION_TEST_SOURCES "${TESTS_DIR}/integration/*.cpp")
add_executable(integrationTests "${CXX_SOURCES}" "${INTEGRATION_TEST_SOURCES}")

target_include_directories(
  integrationTests PRIVATE
  "${ROOT_DIR}/inc"
  "${gtest_SOURCE_DIR}/include"
  "${gmock_SOURCE_DIR}/include"
  "${TORCH_INCLUDE_DIRS}"
)

target_link_libraries(
  integrationTests PRIVATE
  aubio
  curl
  gtest
  gtest_main
  gmock
  OpenMP::OpenMP_CXX
  pthread
  "${TORCH_LIBRARIES}"
  zip
)

add_custom_target(
  integration
  COMMAND "./integrationTests"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ===== Coverage =====
target_compile_options(unitTests PRIVATE --coverage -O0 -g)
target_link_options(unitTests PRIVATE --coverage)

target_compile_options(integrationTests PRIVATE --coverage -O0 -g)
target_link_options(integrationTests PRIVATE --coverage)

set(REPORT_DIR "${ROOT_DIR}/reports/htmlcov/cpp")
file(MAKE_DIRECTORY "${REPORT_DIR}")
set(REPORT_FILE "${REPORT_DIR}/coverage.info")

# ----- Make Coverage -----
add_custom_target(
  coverage
  COMMAND ./unitTests
  COMMAND ./integrationTests
  COMMAND geninfo . -o raw.info --ignore-errors inconsistent,inconsistent --ignore-errors mismatch,mismatch --rc geninfo_unexecuted_blocks=1 > /dev/null
  COMMAND lcov -e raw.info "${ROOT_DIR}/inc*" "${ROOT_DIR}/src/cpp/*" -o "${REPORT_FILE}" --ignore-errors unused,unused > /dev/null
  COMMAND sed -i "/D0Ev/d" "${REPORT_FILE}"
  COMMAND genhtml "${REPORT_FILE}" -o "${REPORT_DIR}"
)
