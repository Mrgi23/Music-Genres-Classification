# ----- RegisterDownloader.cmake -----

# ===== Core C++ library =====
add_library(
    downloader_core STATIC
    "${SOURCES_DIR}/Downloader.cpp"
)

target_include_directories(
    downloader_core PUBLIC
    "${INCLUDE_DIR}"
    "${LIBARCHIVE_INCLUDE_DIRS}"
)

target_link_libraries(
    downloader_core PRIVATE
    CURL::libcurl
    "${LIBARCHIVE_LIBRARIES}"
)

set_target_properties(downloader_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(Downloader::core ALIAS downloader_core)
