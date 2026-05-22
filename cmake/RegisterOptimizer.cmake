# ----- RegisterOptimizer.cmake -----

# ===== Core C++ library =====
add_library(optimizer_core INTERFACE)
target_include_directories(optimizer_core INTERFACE "${INCLUDE_DIR}")
target_link_libraries(optimizer_core INTERFACE "${TORCH_LIBRARIES}")
