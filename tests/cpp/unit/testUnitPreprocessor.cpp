#include "preprocessor.hpp"
#include <gtest/gtest.h>

class TestPreprocessor : public ::testing::Test
{
  protected:
    Preprocessor preprocessor;
};

TEST_F(TestPreprocessor, processFile)
{
  fs::path filePath = "../../../resources/jazz/jazz.00001.wav";
  uint numFramesExpected = 1 + preprocessor.config().size / preprocessor.config().hop;
  uint mfccSizeExpected = preprocessor.config().nmfcc;

  torch::Tensor mfcc = preprocessor.processFile(filePath);
  ASSERT_EQ(mfcc.numel(), numFramesExpected * mfccSizeExpected) << "Invalid number of the MFCC frames.";
}

TEST_F(TestPreprocessor, processFileThrowError)
{
  fs::path filePath = "../../../resources/jazz/jazz.00054.txt";
  EXPECT_THROW(preprocessor.processFile(filePath), std::runtime_error);
}

TEST_F(TestPreprocessor, normalizeData)
{
  uint numFrames = 1 + preprocessor.config().size / preprocessor.config().hop;
  uint mfccSize = preprocessor.config().nmfcc;
  torch::Tensor x = 2.0f * torch::ones({numFrames, mfccSize});
  torch::Tensor meanExpected = torch::ones({numFrames, mfccSize});
  torch::Tensor stdExpected = 2.0f * torch::ones({numFrames, mfccSize});
  torch::Tensor yExpected = 0.5f * torch::ones({numFrames, mfccSize});

  preprocessor.setMean(meanExpected);
  preprocessor.setStd(stdExpected);
  torch::Tensor y = preprocessor.normalizeData(x);
  ASSERT_TRUE(torch::equal(preprocessor.mean(), meanExpected)) << "Invalid mean value.";
  ASSERT_TRUE(torch::equal(preprocessor.std(), stdExpected)) << "Invalid standard deviation value.";
  ASSERT_TRUE(torch::equal(y, yExpected)) << "Invalid normalized signal.";
}
