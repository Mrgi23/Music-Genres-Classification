#pragma once

#ifdef __cplusplus

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

/**
 * @class Downloader
 * @brief Utility class to download and extract the GTZAN dataset (or any dataset) from a URL.
 *
 * The Downloader handles downloading a dataset archive from Kaggle (or other sources),
 * saving it locally, and extracting it into the specified root directory.
 * It also provides helper methods for file existence checks and writing to disk.
 */
class Downloader
{
    public:
        /**
         * @brief Construct a new Downloader object.
         *
         * @param[in] rootPath Root path where the dataset will be stored.
         * @param[in] url URL from which the dataset should be downloaded. Defaults to the GTZAN dataset Kaggle API endpoint.
         */
        Downloader(
            const fs::path & rootPath,
            const std::string & url = "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification"
        );
        /**
         * @brief Destructor.
         *
         * Default cleanup.
         */
        ~Downloader();

        /**
         * @brief Download a file from a given URL and save it to the given path.
         *
         * @param[in] filePath Destination path where the file will be saved.
         * @param[in] url The URL to download the file from.
         */
        static void DownloadFromUrl(const fs::path & filePath, const std::string & url);

        /**
         * @brief Get the root path where the dataset will be extracted.
         *
         * @return fs::path The root path.
         */
        fs::path GetRootPath() const;

        /**
         * @brief Download the dataset and extract it into the root directory.
         *
         * This method checks if the dataset already exists, downloads it if missing,
         * and extracts the archive into the root path.
         */
        void DownloadAndExtract();
    private:
        /**
         * @brief libcurl write callback to stream data into a file.
         *
         * @param[in] ptr Pointer to the data buffer.
         * @param[in] size Size of each data element.
         * @param[in] nmemb Number of elements.
         * @param[out] userdata File handle provided by the caller.
         * @return size_t Number of bytes written.
         */
        static size_t writeToFile(void * ptr, size_t size, size_t nmemb, void * userdata);

        /**
         * @brief Check whether the dataset already exists at the root path.
         *
         * @return true If the dataset already exists.
         * @return false If it does not exist and needs to be downloaded.
         */
        bool IsExisting();
        /**
         * @brief Extract the downloaded archive into the root path.
         */
        void ExtractFile();

        fs::path m_rootPath;
        std::string m_url;
        fs::path m_zipFile;
};

#endif
