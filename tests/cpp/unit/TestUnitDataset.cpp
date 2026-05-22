#include "Dataset.h"
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;
using ::testing::Throw;

struct MockDataset
{
  static void Mock(fs::path& rootPath, bool preload, size_t size, long H, long W)
  {
    rootPath = fs::temp_directory_path() / "resources";
    fs::create_directory(rootPath);

    if (!preload)
    {
      std::vector<std::string> genres = {"jazz", "rock"};
      for (const auto& genre : genres)
      {
        fs::path dirPath = rootPath / genre;
        fs::create_directory(dirPath);

        for (int i = 0; i < size / genres.size(); ++i)
        {
          fs::path filePath = dirPath / (genre + ".0000" + std::to_string(i) + ".wav");
          std::ofstream file(filePath);
          file << "Mock...Data";
        }
      }
    }
    else
    {
      std::vector<torch::Tensor> data(size, torch::rand({H, W}));
      std::vector<torch::Tensor> target(size, torch::rand(1));
      c10::Dict<int64_t, std::string> classes;
      classes.insert(0, "jazz");
      classes.insert(1, "rock");

      fs::path datasetPath = rootPath / "dataset_cpp.pt";
      torch::serialize::OutputArchive dataset;
      dataset.write("data", data);
      dataset.write("target", target);
      dataset.write("classes", classes);
      dataset.save_to(datasetPath);
    }
  }
};

class MockPreprocessor : public Preprocessor
{
  public:
    MockPreprocessor(PreprocessorConfig cfg = PreprocessorConfig()) : Preprocessor(cfg) {}
    MOCK_METHOD(torch::Tensor, processFile, (const fs::path& filePath), (override));
    MOCK_METHOD(torch::Tensor, normalizeData, (const torch::Tensor& x), (override));
};

class TestInitializationParam : public ::testing::TestWithParam<std::string>
{
  protected:
    fs::path rootPath;
    long H, W;
    size_t datasetSize;
    AudioDataset* audioDataset = nullptr;
    MockPreprocessor* mockPreprocessor = nullptr;

    void SetUp() override
    {
      std::string init = GetParam();
      datasetSize = 6;
      H = 1290;
      W = 13;
      MockDataset::Mock(rootPath, (init == "preload") ? true : false, datasetSize, H, W);
      mockPreprocessor = new MockPreprocessor();

      if (init == "corrupt")
      {
        EXPECT_CALL(*mockPreprocessor, processFile(_))
          .WillOnce(Throw(std::runtime_error("Preprocessor::loadAndCrop: Invalid or corrupted file.")))
          .WillRepeatedly(Return(torch::rand({1, H, W})));
        ASSERT_THROW(audioDataset = new AudioDataset(rootPath, mockPreprocessor), std::runtime_error);
      }
      else if (init == "init")
      {
        EXPECT_CALL(*mockPreprocessor, processFile(_))
          .WillRepeatedly(Return(torch::rand({1, H, W})));
      }

      audioDataset = new AudioDataset(rootPath, mockPreprocessor);
    }

    void TearDown() override
    {
      fs::remove_all(rootPath);
      delete mockPreprocessor;
      mockPreprocessor = nullptr;
      delete audioDataset;
      audioDataset = nullptr;
    }
};

TEST_P(TestInitializationParam, get)
{
  EXPECT_CALL(*mockPreprocessor, normalizeData(_)).WillOnce(Return(torch::rand({1, H, W})));
  torch::data::Example<> sample = audioDataset->get(0);
  ASSERT_EQ(sample.data.numel(), H * W) << "Invalid size of the sample data.";
  ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

INSTANTIATE_TEST_SUITE_P(
  TestDatasetWithParams,
  TestInitializationParam,
  ::testing::Values("init", "corrupt", "preload"),
  [](const testing::TestParamInfo<std::string>& info)
  {
    return info.param;
  }
);

class TestDataset : public ::testing::Test
{
  protected:
    fs::path rootPath;
    long H, W;
    size_t datasetSize;
    AudioDataset* audioDataset = nullptr;

    void SetUp() override
    {
      datasetSize = 6;
      H = 1290;
      W = 13;
      MockDataset::Mock(rootPath, true, datasetSize, H, W);
      audioDataset = new AudioDataset(rootPath, nullptr);
    }

    void TearDown() override
    {
      fs::remove_all(rootPath);
      delete audioDataset;
      audioDataset = nullptr;
    }
};

TEST_F(TestDataset, classes)
{
  c10::Dict<int64_t, std::string> classesExpected;
  classesExpected.insert(0, "jazz");
  classesExpected.insert(1, "rock");

  c10::Dict<int64_t, std::string> classes = audioDataset->classes();
  ASSERT_EQ(classes.size(), classesExpected.size()) << "Invalid number of classes.";
  for (const auto& pair : classes)
  {
    ASSERT_TRUE(classesExpected.contains(pair.key())) << "Invalid class.";
    ASSERT_EQ(pair.value(), classesExpected.at(pair.key())) << "Invalid class.";
  }
}

TEST_F(TestDataset, size)
{
  torch::optional<size_t> size = audioDataset->size();
  ASSERT_EQ(size.value(), datasetSize) << "Invalid size of the dataset.";
}
