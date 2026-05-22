# ===== Project Setup =====
cmake_minimum_required(VERSION 3.28)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== Project =====
project(Tests)

# ===== Root Directory =====
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(CMAKE_CONFIG_DIR "${ROOT_DIR}/cmake" CACHE PATH "CMake Configurations")

# ===== Setup =====
include("${CMAKE_CONFIG_DIR}/Utils.cmake")

# ===== Additional Dependencies =====
# ----- Google Test & Google Mock -----
find_package(GTest REQUIRED)

# ===== Targets =====
set(TESTS_DIR "${ROOT_DIR}/tests/cpp")

# ----- Unit Tests -----
file(GLOB UNIT_TEST_SOURCES "${TESTS_DIR}/unit/*.cpp")
add_executable(unitTests "${CXX_SOURCES}" "${UNIT_TEST_SOURCES}")

target_include_directories(unitTests PRIVATE "${INCLUDE_DIR}")

target_link_libraries(
  unitTests PRIVATE
  Aubio::Aubio
  CURL::libcurl
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
  LibArchive::LibArchive
  OpenMP::OpenMP_CXX
  pthread
  "${TORCH_LIBRARIES}"
)

add_custom_target(
  unit
  COMMAND "./unitTests"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ----- Integration Tests -----
file(GLOB INTEGRATION_TEST_SOURCES "${TESTS_DIR}/integration/*.cpp")
add_executable(integrationTests "${CXX_SOURCES}" "${INTEGRATION_TEST_SOURCES}")

target_include_directories(integrationTests PRIVATE "${INCLUDE_DIR}")

target_link_libraries(
  integrationTests PRIVATE
  Aubio::Aubio
  CURL::libcurl
  GTest::gtest
  GTest::gtest_main
  GTest::gmock
  LibArchive::LibArchive
  OpenMP::OpenMP_CXX
  pthread
  "${TORCH_LIBRARIES}"
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

add_custom_target(
  coverage
  COMMAND ./unitTests
  COMMAND ./integrationTests
  COMMAND lcov --capture --directory . -o raw.info --ignore-errors inconsistent,inconsistent --ignore-errors mismatch,mismatch --rc geninfo_unexecuted_blocks=1 > /dev/null
  COMMAND lcov -e raw.info "${INCLUDE_DIR}/*" "${SOURCES_DIR}/*" -o coverage.info --ignore-errors unused,unused > /dev/null
  COMMAND sed -i "/D0Ev/d" coverage.info
  COMMAND genhtml coverage.info -o "${REPORT_DIR}"
)
