#include <gtest/gtest.h>

#define private public
#include "preprocessor.h"
#undef private

using namespace std;

// Define the test object.
class TestPreprocessor : public ::testing::Test
{
    protected:
        Preprocessor preprocessor;
};

TEST_F(TestPreprocessor, runValidOutput)
{
    // Define the input and expected result.
    fs::path filePath = "../../resources/jazz/jazz.00001.wav";
    uint numFramesExpected = 1 + preprocessor.m_size / preprocessor.m_hop;
    uint mfccSizeExpected = preprocessor.m_nmfcc;

    // Compute the result
    torch::Tensor mfcc = preprocessor.run(filePath);

    // Test the result.
    ASSERT_EQ(mfcc.numel(), numFramesExpected * mfccSizeExpected) << "Invalid number of the MFCC frames.";
}

TEST_F(TestPreprocessor, runInvalidInput)
{
    fs::path filePath = "../../resources/jazz/jazz.00054.txt"; // Preprocessor::loadAndCrop: Invalid or corrupted file.
    EXPECT_THROW(preprocessor.run(filePath), runtime_error);
}

TEST_F(TestPreprocessor, normalizeValidOutput)
{
    // Define the input and expected result.
    uint numFrames = 1 + preprocessor.m_size / preprocessor.m_hop;
    uint mfccSize = preprocessor.m_nmfcc;
    torch::Tensor x = 2.0f * torch::ones({numFrames, mfccSize});
    torch::Tensor meanExpected = torch::ones({numFrames, mfccSize});
    torch::Tensor stdExpected = 2.0f * torch::ones({numFrames, mfccSize});
    torch::Tensor yExpected = 0.5f * torch::ones({numFrames, mfccSize});

    // Compute the result.
    preprocessor.setMean(meanExpected);
    preprocessor.setStd(stdExpected);
    torch::Tensor y = preprocessor.normalize(x);

    // Test the result.
    ASSERT_TRUE(torch::equal(preprocessor.mean(), meanExpected)) << "Invalid mean value.";
    ASSERT_TRUE(torch::equal(preprocessor.std(), stdExpected)) << "Invalid standard deviation value.";
    ASSERT_TRUE(torch::equal(y, yExpected)) << "Invalid normalized signal.";
}
