# ----- RegisterDataset.cmake -----

# ===== Core C++ library =====
add_library(dataset_core STATIC "${SOURCES_DIR}/dataset.cpp")
target_include_directories(dataset_core PUBLIC "${INCLUDE_DIR}")
target_link_libraries(dataset_core PRIVATE Aubio::Aubio OpenMP::OpenMP_CXX "${TORCH_LIBRARIES}")
set_target_properties(dataset_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
