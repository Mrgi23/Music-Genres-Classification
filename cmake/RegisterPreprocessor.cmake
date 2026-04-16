# ----- RegisterPreprocessor.cmake -----

# ===== Core C++ library =====
add_library(preprocessor_core STATIC "${SOURCES_DIR}/Preprocessor.cpp")

target_include_directories(preprocessor_core PUBLIC "${INCLUDE_DIR}")

target_link_libraries(preprocessor_core PRIVATE Aubio::Aubio LibTorch::LibTorch)

set_target_properties(preprocessor_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(Preprocessor::core ALIAS preprocessor_core)
