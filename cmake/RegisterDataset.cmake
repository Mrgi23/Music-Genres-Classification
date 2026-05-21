# ----- RegisterDataset.cmake -----

# ===== Core C++ library =====
add_library(dataset_core STATIC "${SOURCES_DIR}/Dataset.cpp")

target_include_directories(dataset_core PUBLIC "${INCLUDE_DIR}")

target_link_libraries(dataset_core PRIVATE OpenMP::OpenMP_CXX "${TORCH_LIBRARIES}")

set_target_properties(dataset_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(Dataset::core ALIAS dataset_core)
