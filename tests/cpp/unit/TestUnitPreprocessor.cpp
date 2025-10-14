#include "Preprocessor.h"

#include <gtest/gtest.h>

using namespace std;

// Define the test object.
class TestPreprocessor : public ::testing::Test
{
    protected:
        Preprocessor preprocessor;
};

TEST_F(TestPreprocessor, ProcessFile)
{
    // Define the input and expected result.
    fs::path filePath = "../../resources/jazz/jazz.00001.wav";
    uint numFramesExpected = 1 + preprocessor.GetCfg().size / preprocessor.GetCfg().hop;
    uint mfccSizeExpected = preprocessor.GetCfg().nmfcc;

    // Compute the result
    torch::Tensor mfcc = preprocessor.ProcessFile(filePath);

    // Test the result.
    ASSERT_EQ(mfcc.numel(), numFramesExpected * mfccSizeExpected) << "Invalid number of the MFCC frames.";
}

TEST_F(TestPreprocessor, ProcessFileThrowError)
{
    fs::path filePath = "../../resources/jazz/jazz.00054.txt"; // Preprocessor::LoadAndCrop: Invalid or corrupted file.
    EXPECT_THROW(preprocessor.ProcessFile(filePath), runtime_error);
}

TEST_F(TestPreprocessor, NormalizeData)
{
    // Define the input and expected result.
    uint numFrames = 1 + preprocessor.GetCfg().size / preprocessor.GetCfg().hop;
    uint mfccSize = preprocessor.GetCfg().nmfcc;
    torch::Tensor x = 2.0f * torch::ones({numFrames, mfccSize});
    torch::Tensor meanExpected = torch::ones({numFrames, mfccSize});
    torch::Tensor stdExpected = 2.0f * torch::ones({numFrames, mfccSize});
    torch::Tensor yExpected = 0.5f * torch::ones({numFrames, mfccSize});

    // Compute the result.
    preprocessor.SetMean(meanExpected);
    preprocessor.SetStd(stdExpected);
    torch::Tensor y = preprocessor.NormalizeData(x);

    // Test the result.
    ASSERT_TRUE(torch::equal(preprocessor.GetMean(), meanExpected)) << "Invalid mean value.";
    ASSERT_TRUE(torch::equal(preprocessor.GetStd(), stdExpected)) << "Invalid standard deviation value.";
    ASSERT_TRUE(torch::equal(y, yExpected)) << "Invalid normalized signal.";
}
