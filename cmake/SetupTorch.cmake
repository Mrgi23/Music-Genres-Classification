# ----- SetupTorch.cmake -----

# ===== Set Torch Variables =====
set(PACKAGE libtorch)
set(TORCH_INSTALL_DIR "${ROOT_DIR}/external")
set(TORCH_ROOT "${TORCH_INSTALL_DIR}/${PACKAGE}" CACHE PATH "Libtorch Root")

if(EXISTS "${TORCH_ROOT}")
    message(STATUS "Torch found")
else()
    message(STATUS "Torch not found, setting up...")

    # ===== Set Additional Torch Variables =====
    set(PROJECT_ID 68060502)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        set(ARCH "x86_64")
        set(VERSION 2.8.0+cu128)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(ARCH "aarch64")
        set(VERSION 2.8.0)
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(BASE_URL "https://gitlab.com/api/v4/projects/${PROJECT_ID}/packages/generic/${PACKAGE}/${VERSION}/${ARCH}")

    set(ARCHIVE_FILE "${PACKAGE}.tar.gz")

    set(SHA_FILE "${ARCHIVE_FILE}.sha256")

    set(ARCHIVE_DOWNLOAD_PATH "${TORCH_INSTALL_DIR}/${ARCHIVE_FILE}")

    set(SHA_DOWNLOAD_PATH "${TORCH_INSTALL_DIR}/${SHA_FILE}")

    # ==== Download Package ====
    file(MAKE_DIRECTORY "${TORCH_INSTALL_DIR}")

    file(
        DOWNLOAD "${BASE_URL}/${SHA_FILE}" "${SHA_DOWNLOAD_PATH}"
        TLS_VERIFY ON
    )

    file(READ "${SHA_DOWNLOAD_PATH}" SHA_CONTENTS)

    string(REGEX MATCH "^[0-9a-fA-F]+" SHA256 "${SHA_CONTENTS}")

    message(STATUS "Downloading ${BASE_URL}/${ARCHIVE_FILE}")
    file(
        DOWNLOAD "${BASE_URL}/${ARCHIVE_FILE}" "${ARCHIVE_DOWNLOAD_PATH}"
        EXPECTED_HASH SHA256=${SHA256}
        TLS_VERIFY ON
    )

    file(REMOVE "${SHA_DOWNLOAD_PATH}")

    # ===== Extract Package ====
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E tar xzf "${ARCHIVE_DOWNLOAD_PATH}"
        WORKING_DIRECTORY "${TORCH_INSTALL_DIR}"
    )

    file(REMOVE "${ARCHIVE_DOWNLOAD_PATH}")
    message(STATUS "Torch setup done")
endif()


find_package(Torch REQUIRED PATHS "${TORCH_ROOT}" NO_DEFAULT_PATH)
