#ifndef DOWNLOADER_H
#define DOWNLOADER_H

#ifdef __cplusplus

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

class Downloader {
    private:
        fs::path dst;
        std::string url;
        fs::path zipFile;

        static size_t writeToFile(void * ptr, size_t size, size_t nmemb, void * userdata) {
            // Cast user data.
            std::ofstream * out = static_cast<std::ofstream*>(userdata);

            // Write incoming chunk.
            out->write(static_cast<char*>(ptr), size * nmemb);
            return size * nmemb;
        }

        bool datasetExists();
        void datasetDownload();
        void datasetExtract();
    public:
        Downloader(
            fs::path dst = "../dataset",
            std::string url = "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification"
        ) : dst(dst), url(url), zipFile(dst / "dataset.zip") {}
        ~Downloader() {}

        void run();
        inline fs::path const root() { return dst; }
};

#endif

#endif
