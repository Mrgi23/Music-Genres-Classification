#include "Downloader.h"

#include <algorithm>
#include <curl/curl.h>
#include <zip.h>

using namespace std;

Downloader::Downloader(const fs::path & rootPath, const string & url)
    : m_rootPath(rootPath), m_url(url), m_zipFile(rootPath / "dataset.zip")
{

}

Downloader::~Downloader() = default;

fs::path Downloader::GetRootPath() const
{
    return m_rootPath;
}

void Downloader::DownloadFromUrl(const fs::path & filePath, const string & url)
{
    fs::create_directories(filePath.parent_path());
    ofstream out(filePath, ios::binary);

    CURL * curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeToFile);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    out.close();
}

void Downloader::DownloadAndExtract()
{
    if (!IsExisting())
    {
        DownloadFromUrl(m_zipFile, m_url);

        ExtractFile();
    }
}

size_t Downloader::writeToFile(void * ptr, size_t size, size_t nmemb, void * userdata)
{
    std::ofstream * out = static_cast<std::ofstream*>(userdata);

    out->write(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

bool Downloader::IsExisting()
{
    return (
        fs::exists(m_rootPath) &&
        std::any_of
        (
            fs::recursive_directory_iterator(m_rootPath),
            fs::recursive_directory_iterator(),
            [](auto& entry)
            {
                return entry.path().extension() == ".wav";
            }
        )
    );
}

void Downloader::ExtractFile()
{
    int err = 0;
    zip * archive = zip_open(m_zipFile.c_str(), ZIP_RDONLY, &err);

    zip_int64_t num_entries = zip_get_num_entries(archive, 0);
    for (zip_uint64_t i = 0; i < num_entries; ++i)
    {
        const char * mem = zip_get_name(archive, i, 0);
        std::string member(mem);

        if (member.starts_with("Data/genres_original/") && member.ends_with(".wav"))
        {
            fs::path relativePath = fs::path(member).lexically_relative("Data/genres_original/");
            fs::path savePath = m_rootPath / fs::path(relativePath);
            fs::create_directories(savePath.parent_path());

            zip_file * file = zip_fopen_index(archive, i, 0);
            ofstream out(savePath, ios::binary);
            char buffer[8192];
            zip_int64_t bytesRead;
            while ((bytesRead = zip_fread(file, buffer, sizeof(buffer))) > 0)
            {
                out.write(buffer, bytesRead);
            }

            out.close();
            zip_fclose(file);
        }
    }
    zip_close(archive);
    fs::remove(m_zipFile);

    fs::path corruptedFile = m_rootPath / "jazz" / "jazz.00054.wav";
    fs::remove(corruptedFile);
}
