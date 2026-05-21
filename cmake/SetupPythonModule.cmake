# ----- SetupPythonModule.cmake -----

# ===== Set Python Module Variables =====
set(BINDINGS_DIR "${ROOT_DIR}/bindings" CACHE PATH "Pybind11 binding sources")

# ===== Additional Dependencies =====
# ----- PyBind11 -----
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED CONFIG)

# ===== Register Bindings =====
include("${CMAKE_CONFIG_DIR}/RegisterDownloader.cmake")
include("${CMAKE_CONFIG_DIR}/RegisterPreprocessor.cmake")
include("${CMAKE_CONFIG_DIR}/RegisterDataset.cmake")
include("${CMAKE_CONFIG_DIR}/RegisterModel.cmake")
include("${CMAKE_CONFIG_DIR}/RegisterScheduler.cmake")
include("${CMAKE_CONFIG_DIR}/RegisterOptimizer.cmake")
include("${CMAKE_CONFIG_DIR}/RegisterTrainer.cmake")

# ===== Python Module Sources =====
file(GLOB BINDING_SOURCES "${BINDINGS_DIR}/*.cpp")

# ===== Python extension module (pybind11) =====
pybind11_add_module(musicnet_module "${BINDING_SOURCES}")

target_link_libraries(
    musicnet_module PRIVATE
    Downloader::core
    Preprocessor::core
    Dataset::core
    Model::core
    Scheduler::core
    Optimizer::core
    Trainer::core
    "${TORCH_LIBRARIES}"
)

set_target_properties(
    musicnet_module PROPERTIES
    OUTPUT_NAME "${MODULE_NAME}"
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    POSITION_INDEPENDENT_CODE ON
)
