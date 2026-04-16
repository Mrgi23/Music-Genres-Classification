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
         * @param[in] url URL from which the dataset should be downloaded. Defaults to the Cloudflare R2 storage API endpoint.
         */
        Downloader(
            const fs::path & rootPath,
            const std::string & url = "https://s3.mrgi23.com/artifacts/Music-Genres-Classification/dataset/dataset.tar.zst"
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
         * @return true If download was successfull.
         * @return false If download has failed.
         *
         * @throws std::invalid_argument If the URL is invalid.
         */
        static bool DownloadFromUrl(const fs::path & filePath, const std::string & url);

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
        fs::path m_archiveFile;
};

#endif
