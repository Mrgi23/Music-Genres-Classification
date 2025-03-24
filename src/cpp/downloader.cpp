#include <curl/curl.h>
#include <zip.h>
#include "downloader.h"

using namespace std;

bool Downloader::datasetExists() {
    // Check if .wav files exist.
    for (auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.path().extension() == ".wav") { return true; }
    }
    return false;
}

void Downloader::datasetDownload() {
    // Create zip file.
    ofstream out(zipFile, ios::binary);

    // Setup curl for downloading dataset from the url.
    CURL * curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeToFile);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    // Download the dataset from the URL.
    curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    out.close();
}

void Downloader::datasetExtract() {
    // Open the zip file.
    int err = 0;
    zip * archive = zip_open(zipFile.c_str(), ZIP_RDONLY, &err);

    zip_int64_t num_entries = zip_get_num_entries(archive, 0);
    for (zip_uint64_t i = 0; i < num_entries; ++i) {
        const char * mem = zip_get_name(archive, i, 0);
        std::string member(mem);

        // Select only .wav files.
        if (member.starts_with("Data/genres_original/") && member.ends_with(".wav")) {
            // Define the .wav file relative path.
            fs::path relativePath = fs::path(member).lexically_relative("Data/genres_original/");

            // Compute the absoulte save path.
            fs::path savePath = root / fs::path(relativePath);
            fs::create_directories(savePath.parent_path());

            // Copy the .wav file and store it in the dataset folder, keeping the tree intact.
            zip_file * file = zip_fopen_index(archive, i, 0);
            ofstream out(savePath, ios::binary);
            char buffer[8192];
            zip_int64_t bytesRead;
            while ((bytesRead = zip_fread(file, buffer, sizeof(buffer))) > 0) { out.write(buffer, bytesRead); }
            out.close();
            zip_fclose(file);
        }
    }
    zip_close(archive);

    // Remove the zip file.
    fs::remove(zipFile);

    // Remove single corrupted file.
    fs::path corruptedFile = root / "jazz" / "jazz.00054.wav";
    fs::remove(corruptedFile);
}

void Downloader::run() {
    // Create dataset folder.
    fs::create_directories(root);

    // Download and extract dataset if it does not exists.
    if (!datasetExists()) {
        // Download dataset.
        datasetDownload();

        // Extract dataset.
        datasetExtract();
    }
}
