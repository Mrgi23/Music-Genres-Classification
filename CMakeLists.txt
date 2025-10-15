cmake_minimum_required(VERSION 3.10)

# ===== Project =====
project(MusicGenresClassification)

# ===== Options =====
option(BUILD_PYTHON "Build Python bindings" OFF)

# ===== Root Directory =====
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
set(CMAKE_CONFIG_DIR "${ROOT_DIR}/cmake" CACHE PATH "CMake Configurations")

# ===== Setup =====
include("${CMAKE_CONFIG_DIR}/Utils.cmake")

# ===== Set Variables =====
set(APP_DIR "${ROOT_DIR}/app")
set(OUTPUT_DIR "${ROOT_DIR}/bin" CACHE PATH "Output Directory")
set(MODULE_NAME "musicnet" CACHE STRING "Output Module Name")

# ===== Setup Python Module =====
if (BUILD_PYTHON)
    include("${CMAKE_CONFIG_DIR}/SetupPythonModule.cmake")
endif()

# ===== Additional Dependencies =====
# ----- Nlohmann-Json -----
find_package(nlohmann_json REQUIRED)

# ----- toml++ -----
include(FetchContent)
FetchContent_Declare(
  tomlplusplus
  GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
  GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

# ===== Add Additional Sources =====
list(APPEND CXX_SOURCES "${APP_DIR}/main.cpp")

# ===== Executable =====
add_executable(musicnet_exe "${CXX_SOURCES}")

target_include_directories(
  musicnet_exe PRIVATE
  "${INCLUDE_DIR}"
  "${APP_DIR}"
  "${AUBIO_INCLUDE_DIRS}"
  "${LIBZIP_INCLUDE_DIRS}"
  "${TORCH_INCLUDE_DIRS}"
)

target_link_libraries(
  musicnet_exe PRIVATE
  "${AUBIO_LIBRARIES}"
  CURL::libcurl
  "${LIBZIP_LIBRARIES}"
  nlohmann_json::nlohmann_json
  OpenMP::OpenMP_CXX
  pthread
  tomlplusplus::tomlplusplus
  "${TORCH_LIBRARIES}"
)

set_target_properties(
  musicnet_exe PROPERTIES
  OUTPUT_NAME "${MODULE_NAME}"
  RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)
