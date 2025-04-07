#include <gtest/gtest.h>
#include "downloader.h"

#define private public
#include "preprocessor.h"
#include "dataset.h"
#undef private

using namespace std;

// Define the test object.
class TestAudioDataset : public ::testing::Test {
    protected:
        AudioDataset * audioDataset = nullptr;
        Preprocessor * preprocessor = nullptr;

        void SetUp() override {
            fs::path datasetPath = "../../dataset";
            Downloader(datasetPath).run();
            preprocessor = new Preprocessor();
            audioDataset = new AudioDataset(datasetPath, preprocessor);
        }

        void TearDown() override {
            delete preprocessor;
            preprocessor = nullptr;
            delete audioDataset;
            audioDataset = nullptr;
        }
};

TEST_F(TestAudioDataset, getValidOutput) {
    // Collect the parameters.
    uint size = audioDataset->preprocessor->size;
    uint hop = audioDataset->preprocessor->hop;
    uint nmfcc = audioDataset->preprocessor->nmfcc;

    // Define the expected result.
    size_t numFrames = size / hop + 1;
    size_t mfccSize = nmfcc;

    // Compute the result.
    torch::data::Example<> sample = audioDataset->get(0);

    // Test the result.
    ASSERT_EQ(sample.data.numel(), numFrames * mfccSize) << "Invalid size of the samle data.";
    ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestAudioDataset, sizeValidOutput) {
    // Define the expected result.
    size_t sizeExpected = 999;

    // Compute the result.
    torch::optional<size_t> size = audioDataset->size();

    // Test the result.
    ASSERT_EQ(size.value(), sizeExpected) << "Invalid size of the dataset.";
}

TEST_F(TestAudioDataset, classesValidOutput) {
    // Define the expected result.
    map<string, uint> classesExpected = {
        {"blues", 0},
        {"classical", 1},
        {"country", 2},
        {"disco", 3},
        {"hiphop", 4},
        {"jazz", 5},
        {"metal", 6},
        {"pop", 7},
        {"reggae", 8},
        {"rock", 9}
    };

    // Compute the result.
    map<string, uint> classes = audioDataset->getClasses();

    // Test the result.
    ASSERT_EQ(classes.size(), classesExpected.size()) << "Invalid number of classes.";
    for (const auto& pair : classes) {
        ASSERT_EQ(pair.second, classesExpected[pair.first]) << "Invalid class.";
    }
}
