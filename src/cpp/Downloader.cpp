#include "Downloader.h"

#include <algorithm>
#include <archive.h>
#include <archive_entry.h>
#include <curl/curl.h>
#include <stdexcept>
#include <vector>

using namespace std;

Downloader::Downloader(const fs::path & rootPath, const string & url)
    : m_rootPath(rootPath), m_url(url), m_archiveFile(rootPath / "dataset.tar.zst")
{

}

Downloader::~Downloader() = default;

fs::path Downloader::GetRootPath() const
{
    return m_rootPath;
}

bool Downloader::DownloadFromUrl(const fs::path & filePath, const string & url)
{
    fs::create_directories(filePath.parent_path());
    FILE * out = std::fopen(filePath.string().c_str(), "wb");

    CURL * curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, nullptr);
    curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 1024L * 1024L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, nullptr);
    CURLcode rc = curl_easy_perform(curl);

    if (rc == CURLE_OK)
    {
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_NOBODY, 0L);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
        rc = curl_easy_perform(curl);
    }

    curl_easy_cleanup(curl);
    std::fclose(out);

    return rc == CURLE_OK;
}

void Downloader::DownloadAndExtract()
{
    if (!IsExisting())
    {
        if (DownloadFromUrl(m_archiveFile, m_url))
            ExtractFile();
        else
            throw invalid_argument("Downloader::DownloadAndExtract: URL: " + m_url + " is invalid.");
    }
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
    archive * a = archive_read_new();
    archive_read_support_filter_all(a); // zstd
    archive_read_support_format_all(a); // tar
    archive_read_open_filename(a, m_archiveFile.c_str(), 1024 * 1024);

    archive_entry * entry = nullptr;

    while (archive_read_next_header(a, &entry) == ARCHIVE_OK)
    {
        const char * mem = archive_entry_pathname(entry);
        std::string member(mem ? mem : "");

        fs::path savePath = m_rootPath / fs::path(member);
        fs::create_directories(savePath.parent_path());

        ofstream out(savePath, ios::binary);
        vector<char> buffer(1 << 20);
        la_ssize_t bytesRead;

        while ((bytesRead = archive_read_data(a, buffer.data(), buffer.size())) > 0)
        {
            out.write(buffer.data(), static_cast<std::streamsize>(bytesRead));
        }

        out.close();
    }

    archive_read_close(a);
    archive_read_free(a);

    fs::remove(m_archiveFile);
}
