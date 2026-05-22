#include "Dataset.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <random>

class TestSubset : public ::testing::Test
{
  protected:
    Preprocessor* preprocessor = nullptr;
    AudioDataset* audioDataset = nullptr;
    AudioSubset* audioSubset = nullptr;

    void SetUp() override
    {
      fs::path datasetPath = "../../../resources";
      preprocessor = new Preprocessor();
      audioDataset = new AudioDataset(datasetPath, preprocessor);

      std::vector<size_t> pool(999);
      std::iota(pool.begin(), pool.end(), 0);

      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(pool.begin(), pool.end(), g);

      std::vector<size_t> indices(pool.begin(), pool.begin() + 500);
      audioSubset = new AudioSubset(audioDataset, indices);
    }

    void TearDown() override
    {
      delete preprocessor;
      preprocessor = nullptr;
      delete audioDataset;
      audioDataset = nullptr;
      delete audioSubset;
      audioSubset = nullptr;
    }
};

TEST_F(TestSubset, get)
{
  uint size = audioSubset->dataset()->preprocessor()->config().size;
  uint hop = audioSubset->dataset()->preprocessor()->config().hop;
  uint nmfcc = audioSubset->dataset()->preprocessor()->config().nmfcc;

  size_t numFramesExpected = size / hop + 1;
  size_t mfccSizeExpected = nmfcc;
  torch::data::Example<> sample = audioSubset->get(0);
  ASSERT_EQ(sample.data.numel(), numFramesExpected * mfccSizeExpected) << "Invalid size of the sample data.";
  ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestSubset, size)
{
  size_t sizeExpected = 500;
  torch::optional<size_t> size = audioSubset->size();
  ASSERT_EQ(size.value(), sizeExpected) << "Invalid size of the dataset.";
}

TEST_F(TestSubset, GetStackedData)
{
  uint size = audioSubset->dataset()->preprocessor()->config().size;
  uint hop = audioSubset->dataset()->preprocessor()->config().hop;
  uint nmfcc = audioSubset->dataset()->preprocessor()->config().nmfcc;

  size_t numSamplesExpected = 500;
  size_t numFramesExpected = size / hop + 1;
  size_t mfccSizeExpected = nmfcc;
  torch::Tensor data = audioSubset->stackedData();
  ASSERT_EQ(data.numel(), numSamplesExpected * numFramesExpected * mfccSizeExpected) << "Invalid size of the data.";
}
