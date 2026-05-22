#include "downloader.hpp"
#include <gtest/gtest.h>

TEST(TestDownloader, downloadAndExtract)
{
  fs::remove_all("../../../resources");
  Downloader downloader("../../../resources");

  uint nFoldersExpected = 10U;
  uint nFilesExpected = 999U;

  downloader.downloadAndExtract();
  downloader.downloadAndExtract();

  uint nFolders = 0U;
  uint nFiles = 0U;
  for (const auto& entry : fs::recursive_directory_iterator(downloader.rootPath()))
  {
    if (fs::is_directory(entry))
      nFolders++;
    else if (entry.path().extension() == ".wav")
      nFiles++;
  }
  ASSERT_EQ(nFolders, nFoldersExpected) << "Invalid number of the dataset classes.";
  ASSERT_EQ(nFiles, nFilesExpected) << "Invalid number of the dataset samples.";
}

TEST(TestDownloader, downloadAndExtractThrowError)
{
  Downloader downloader("./", "https://s3.mrgi23.com/artifacts/Music-Genres-Classification/dataset/invalid.tar.zst");
  EXPECT_THROW(downloader.downloadAndExtract(), std::invalid_argument);
}