# ----- RegisterPreprocessor.cmake -----

# ===== Core C++ library =====
add_library(
    preprocessor_core STATIC
    "${SOURCES_DIR}/Preprocessor.cpp"
)

target_include_directories(
    preprocessor_core PUBLIC
    "${INCLUDE_DIR}"
    "${AUBIO_INCLUDE_DIRS}"
)

target_link_libraries(
    preprocessor_core PRIVATE
    "${AUBIO_LIBRARIES}"
    "${TORCH_LIBRARIES}"
)

set_target_properties(preprocessor_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(Preprocessor::core ALIAS preprocessor_core)
