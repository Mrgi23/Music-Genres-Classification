#ifndef DOWNLOADER_H
#define DOWNLOADER_H

#ifdef __cplusplus

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

class Downloader
{
    public:
        Downloader(
            fs::path root = "../resources",
            std::string url = "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification"
        );
        ~Downloader();

        void run();
        fs::path root() const;
    private:
        fs::path m_root;
        std::string m_url;
        fs::path m_zipFile;

        bool datasetExists();
        void datasetDownload();
        void datasetExtract();
        static size_t writeToFile(void * ptr, size_t size, size_t nmemb, void * userdata);
};

#endif

#endif
