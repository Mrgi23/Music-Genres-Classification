#include "dataset.hpp"
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Return;

struct MockDataset
{
  static void Mock(fs::path& rootPath, size_t size, long H, long W)
  {
    rootPath = fs::temp_directory_path() / "resources";
    fs::create_directory(rootPath);

    std::vector<torch::Tensor> data(size, torch::rand({H, W}));
    std::vector<torch::Tensor> target(size, torch::rand(1));
    c10::Dict<int64_t, std::string> classes;

    fs::path datasetPath = rootPath / "dataset_cpp.pt";
    torch::serialize::OutputArchive dataset;
    dataset.write("data", data);
    dataset.write("target", target);
    dataset.write("classes", classes);
    dataset.save_to(datasetPath);
  }
};

class MockAudioDataset : public AudioDataset
{
  public:
    MockAudioDataset(const fs::path& rootPath, Preprocessor* preprocessor) : AudioDataset(rootPath, preprocessor) {}
    MOCK_METHOD(torch::data::Example<>, get, (size_t index), (override));
};

class TestSubset : public ::testing::Test
{
  protected:
    fs::path rootPath;
    long H, W;
    size_t datasetSize, subsetSize;
    MockAudioDataset* mockAudiodataset = nullptr;
    AudioSubset* audioSubset = nullptr;

    void SetUp() override
    {
      datasetSize = 6;
      H = 1290;
      W = 13;
      MockDataset::Mock(rootPath, datasetSize, H, W);
      mockAudiodataset = new MockAudioDataset(rootPath, nullptr);

      subsetSize = 3;
      audioSubset = new AudioSubset(mockAudiodataset, std::vector<size_t>{0, 1, 2});
    }

    void TearDown() override
    {
      fs::remove_all(rootPath);
      delete mockAudiodataset;
      mockAudiodataset = nullptr;
      delete audioSubset;
      audioSubset = nullptr;
    }
};

TEST_F(TestSubset, get)
{
  torch::Tensor data = torch::rand({H, W});
  torch::Tensor target = torch::rand(1);
  EXPECT_CALL(*mockAudiodataset, get(_)).WillOnce(Return(torch::data::Example<>(data, target)));

  torch::data::Example<> sample = audioSubset->get(0);
  ASSERT_EQ(sample.data.numel(), H * W) << "Invalid size of the sample data.";
  ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestSubset, size)
{
  torch::optional<size_t> size = audioSubset->size();
  ASSERT_EQ(size.value(), subsetSize) << "Invalid size of the dataset.";
}

TEST_F(TestSubset, stackedData)
{
  torch::Tensor data = torch::rand({H, W});
  torch::Tensor target = torch::rand(1);
  EXPECT_CALL(*mockAudiodataset, get(_)).WillRepeatedly(Return(torch::data::Example<>(data, target)));

  data = audioSubset->stackedData();
  ASSERT_EQ(data.numel(), subsetSize * H * W) << "Invalid size of the data.";

  data = audioSubset->stackedData();
  ASSERT_EQ(data.numel(), subsetSize * H * W) << "Invalid size of the data.";
}
