# ----- SetupTorch.cmake -----

# ===== Set Torch Variables =====
set(PACKAGE libtorch)
set(TORCH_INSTALL_DIR "${ROOT_DIR}/external")
set(TORCH_ROOT "${TORCH_INSTALL_DIR}/${PACKAGE}" CACHE PATH "Libtorch Root")

if(EXISTS "${TORCH_ROOT}")
    message(STATUS "Torch found")
else()
    message(STATUS "Torch not found, setting up...")

    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    # ===== Set Additional Torch Variables =====
    set(PACKAGE libtorch)
    set(VERSION 2.8.0-cu128)

    set(URL "ghcr.io/mrgi23/${PACKAGE}:${VERSION}")

    set(ARCHIVE_FILE "${PACKAGE}.tar.gz")
    set(SHA_FILE "${ARCHIVE_FILE}.sha256")

    set(ARCHIVE_DOWNLOAD_PATH "${TORCH_INSTALL_DIR}/${ARCHIVE_FILE}")
    set(SHA_DOWNLOAD_PATH "${TORCH_INSTALL_DIR}/${SHA_FILE}")

    # ===== Download Package =====
    file(MAKE_DIRECTORY "${TORCH_INSTALL_DIR}")

    find_program(ORAS_EXE oras)
    if (NOT ORAS_EXE)
        message(FATAL_ERROR "oras not found in PATH")
    endif()

    message(STATUS "Pulling package from: ${URL}")

    execute_process(
        COMMAND "${ORAS_EXE}" pull "${URL}"
        WORKING_DIRECTORY "${TORCH_INSTALL_DIR}"
        RESULT_VARIABLE ORAS_RC
        OUTPUT_QUIET
    )
    if (NOT ORAS_RC EQUAL 0)
        message(FATAL_ERROR "oras pull failed (rc=${ORAS_RC}) for ${URL}")
    endif()

    # ===== Verify SHA256 =====
    file(READ "${SHA_DOWNLOAD_PATH}" SHA_CONTENT)
    string(REGEX MATCH "^[0-9a-fA-F]+" SHA_EXPECTED "${SHA_CONTENT}")
    file(REMOVE "${SHA_DOWNLOAD_PATH}")

    file(SHA256 "${ARCHIVE_DOWNLOAD_PATH}" SHA_ACTUAL)
    if (NOT SHA_ACTUAL STREQUAL SHA_EXPECTED)
        message(FATAL_ERROR "SHA256 mismatch for ${ARCHIVE_DOWNLOAD_PATH}")
    endif()

    # ===== Extract Package ====
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E tar xzf "${ARCHIVE_DOWNLOAD_PATH}"
        WORKING_DIRECTORY "${TORCH_INSTALL_DIR}"
    )
    file(REMOVE "${ARCHIVE_DOWNLOAD_PATH}")

    message(STATUS "Torch setup done")
endif()

find_package(Torch REQUIRED PATHS "${TORCH_ROOT}" NO_DEFAULT_PATH)
