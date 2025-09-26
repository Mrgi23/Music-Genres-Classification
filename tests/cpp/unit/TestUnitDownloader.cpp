#include "Downloader.h"

#include <gtest/gtest.h>

using namespace std;

TEST(TestDownloader, DownloadAndExtract)
{
    // Define the test object.
    fs::remove_all("../../resources");
    Downloader downloader("../../resources");

    // Define the expected result.
    uint nFoldersExpected = 10U;
    uint nFilesExpected = 999U;

    // Compute the result.
    downloader.DownloadAndExtract();
    downloader.DownloadAndExtract();

    // Test the result.
    uint nFolders = 0U;
    uint nFiles = 0U;
    for (auto& entry : fs::recursive_directory_iterator(downloader.GetRootPath()))
    {
        if (fs::is_directory(entry))
        {
            nFolders++;
        }
        else if (entry.path().extension() == ".wav")
        {
            nFiles++;
        }
    }
    ASSERT_EQ(nFolders, nFoldersExpected) << "Invalid number of the dataset classes.";
    ASSERT_EQ(nFiles, nFilesExpected) << "Invalid number of the dataset samples.";
}
