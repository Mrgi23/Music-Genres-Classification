# ----- RegisterModel.cmake -----

# ===== Core C++ library =====
add_library(model_core STATIC "${SOURCES_DIR}/model.cpp")
target_include_directories(model_core PUBLIC "${INCLUDE_DIR}")
target_link_libraries(model_core PRIVATE "${TORCH_LIBRARIES}")
set_target_properties(model_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
