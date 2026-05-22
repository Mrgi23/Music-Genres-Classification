# ----- RegisterTrainer.cmake -----

# ===== Core C++ library =====
add_library(trainer_core STATIC "${SOURCES_DIR}/trainer.cpp")
target_include_directories(trainer_core PUBLIC "${INCLUDE_DIR}")
target_link_libraries(trainer_core PRIVATE "${TORCH_LIBRARIES}")
set_target_properties(trainer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
