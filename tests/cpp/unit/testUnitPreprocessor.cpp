#include <gtest/gtest.h>
#include "downloader.h"

#define private public
#include "preprocessor.h"
#undef private

using namespace std;

// Define the test object.
class TestPreprocessor : public ::testing::Test {
    protected:
        Preprocessor preprocessor;
};

TEST_F(TestPreprocessor, validOutput) {
    // Define the input and expected result.
    fs::path filePath = "../../dataset/jazz/jazz.00001.wav";
    uint nFramesExpected = preprocessor.size / preprocessor.hop + 1;
    uint specSizeExpected = preprocessor.nfft / 2 + 1;
    uint mfccSizeExpected = preprocessor.nmfcc;

    // Compute the result;
    vector<vector<float>> spectrogram;
    vector<vector<float>> mfcc;
    Downloader("../../dataset").run();
    preprocessor.run(filePath, spectrogram, mfcc);

    // Test the result.
    ASSERT_EQ(spectrogram.size(), nFramesExpected) << "Invalid number of the spectrogram frames.";
    ASSERT_EQ(spectrogram[0].size(), specSizeExpected) << "Invalid size of the spectrogram.";
    ASSERT_EQ(mfcc.size(), nFramesExpected) << "Invalid number of the MFCC frames.";
    ASSERT_EQ(mfcc[0].size(), mfccSizeExpected) << "Invalid size of the MFCC.";
}

TEST_F(TestPreprocessor, invalidInput) {
    vector<vector<float>> spectrogram;
    vector<vector<float>> mfcc;
    fs::path filePath = "../../dataset/jazz/jazz.00054.txt"; // Preprocessor::loadAndCrop: Invalid or corrupted file.
    EXPECT_THROW(preprocessor.run(filePath, spectrogram, mfcc), runtime_error);
}
