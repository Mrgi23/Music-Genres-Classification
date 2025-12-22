# ===== Utils.cmake =====

# ===== Setup =====
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===== Build =====
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type")

# ===== Set Variables =====
set(INCLUDE_DIR "${ROOT_DIR}/inc" CACHE PATH "C++ Headers")
set(SOURCES_DIR "${ROOT_DIR}/src/cpp" CACHE PATH "C++ Sources")

# ===== Dependencies =====
find_package(PkgConfig REQUIRED)

# ----- Aubio -----
pkg_check_modules(AUBIO REQUIRED aubio)

# ----- Curl -----
find_package(CURL REQUIRED)

# ----- OpenMP -----
find_package(OpenMP REQUIRED)

# ----- Torch -----
include("${CMAKE_CONFIG_DIR}/SetupTorch.cmake")

# ----- Tar.Zstd -----
pkg_check_modules(LIBARCHIVE REQUIRED libarchive)

# ===== Sources =====
file(GLOB CXX_SOURCES "${SOURCES_DIR}/*.cpp")
set(CXX_SOURCES "${CXX_SOURCES}" CACHE STRING "Project Sources")
