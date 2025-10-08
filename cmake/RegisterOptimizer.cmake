# ----- RegisterOptimizer.cmake -----

# ===== Core C++ library =====
add_library(
    optimizer_core INTERFACE
)

# Public includes (since header-only)
target_include_directories(
    optimizer_core INTERFACE
    "${INCLUDE_DIR}"
)

# Torch is header+lib, so we still propagate link
target_link_libraries(
    optimizer_core INTERFACE
    "${TORCH_LIBRARIES}"
)

add_library(Optimizer::core ALIAS optimizer_core)
