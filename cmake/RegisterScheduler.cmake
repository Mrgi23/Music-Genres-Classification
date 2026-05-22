# ----- RegisterScheduler.cmake -----

# ===== Core C++ library =====
add_library(scheduler_core STATIC "${SOURCES_DIR}/scheduler.cpp")
target_include_directories(scheduler_core PUBLIC "${INCLUDE_DIR}")
target_link_libraries(scheduler_core PRIVATE "${TORCH_LIBRARIES}")
set_target_properties(scheduler_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
