#include <gtest/gtest.h>
#include "downloader.h"

using namespace std;

TEST(TestDownloader, testDownloader) {
    // Define the test object.
    Downloader downloader("../../resources");

    // Define the expected result.
    uint nFoldersExpected = 10U;
    uint nFilesExpected = 999U;

    // Compute the result.
    fs::path root = downloader.getRoot();
    for (const auto& entry : fs::directory_iterator(root)) {
        if (fs::is_directory(entry.path())) { fs::remove_all(entry.path()); }
    }
    downloader.run();
    downloader.run();

    // Test the result.
    uint nFolders = 0U;
    uint nFiles = 0U;
    for (auto& entry : fs::recursive_directory_iterator(root)) {
        if (fs::is_directory(entry)) { nFolders++; }
        else if (entry.path().extension() == ".wav") { nFiles++; }
    }
    ASSERT_EQ(nFolders, nFoldersExpected) << "Invalid number of the dataset classes.";
    ASSERT_EQ(nFiles, nFilesExpected) << "Invalid number of the dataset samples.";
}
