# ----- SetupTorch.cmake -----

# ===== Setup Variables =====
set(PROJECT_ID 68060502)

set(PACKAGE libtorch)

if (NOT VERSION)
    set(VERSION 2.6.0)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(ARCH "x86_64")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(ARCH "aarch64")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(BASE_URL "https://gitlab.com/api/v4/projects/${PROJECT_ID}/packages/generic/${PACKAGE}/${VERSION}/${ARCH}")

set(ARCHIVE_FILE "${PACKAGE}-${VERSION}-${ARCH}.tar.gz")

set(SHA_FILE "${ARCHIVE_FILE}.sha256")

set(INSTALL_DIR "${ROOT_DIR}/external")

set(ARCHIVE_DOWNLOAD_PATH "${INSTALL_DIR}/${ARCHIVE_FILE}")

set(SHA_DOWNLOAD_PATH "${INSTALL_DIR}/${SHA_FILE}")
# ===========================

# ==== Download Packages ====
file(MAKE_DIRECTORY "${INSTALL_DIR}")

file(
    DOWNLOAD "${BASE_URL}/${SHA_FILE}" "${SHA_DOWNLOAD_PATH}"
    TLS_VERIFY ON
)

file(READ "${SHA_DOWNLOAD_PATH}" SHA_CONTENTS)

string(REGEX MATCH "^[0-9a-fA-F]+" SHA256 "${SHA_CONTENTS}")

file(
    DOWNLOAD "${BASE_URL}/${ARCHIVE_FILE}" "${ARCHIVE_DOWNLOAD_PATH}"
    EXPECTED_HASH SHA256=${SHA256}
    TLS_VERIFY ON
)

file(REMOVE "${SHA_DOWNLOAD_PATH}")
# ===========================

# ===== Extract Packages ====
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar xzf "${ARCHIVE_DOWNLOAD_PATH}"
    WORKING_DIRECTORY "${INSTALL_DIR}"
)

file(REMOVE "${ARCHIVE_DOWNLOAD_PATH}")
list(PREPEND CMAKE_PREFIX_PATH "${INSTALL_DIR}/${PACKAGE}-${ARCH}")
# ===========================