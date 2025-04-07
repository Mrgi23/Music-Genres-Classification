#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>
#include "preprocessor.h"

#define private public
#include "dataset.h"
#undef private

using namespace std;
using ::testing::_;
using ::testing::Return;

// Helper function.
vector<vector<float>> randomMatrix(size_t rows, size_t cols) {
    // Get random seed and initialize PRN generator.
    random_device rd;
    mt19937 gen(rd());

    // Generate random numbers between 0 and 1.
    uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Create matrix of random numbers.
    vector<vector<float>> matrix(rows, vector<float>(cols));
    for (auto& row : matrix) {
        for (auto& elem : row) {
            elem = dist(gen);
        }
    }
    return matrix;
}

// Define the mock objects.
class MockPreprocessor : public Preprocessor {
    public:
        MockPreprocessor(
            uint size = 660000,
            uint nfft = 1024,
            uint hop = 512,
            uint nmels = 128,
            uint nmfcc = 13
        ) : Preprocessor(size, nfft, hop, nmels, nmfcc) {}
        MOCK_METHOD(vector<vector<float>>, run, (const fs::path& filePath), (override));
};

// Define the test object.
class TestAudioDataset : public ::testing::Test {
    protected:
        fs::path rootPath;
        AudioDataset * audioDataset = nullptr;
        MockPreprocessor * mockPreprocessor = nullptr;

        void mockDataset(void) {
            // Create temporary root dir.
            rootPath = fs::temp_directory_path() / "dataset";
            fs::create_directory(rootPath);

            // Create temporary genres dirs.
            vector<string> genres = {"jazz", "rock"};
            for (const auto& genre : genres) {
                fs::path dirPath = rootPath / genre;
                fs::create_directory(dirPath);

                // Create 3 temporary .wav files
                for (int i = 0; i < 3; ++i) {
                    fs::path filePath = dirPath / (genre + ".0000" + to_string(i) + ".wav");
                    ofstream file(filePath);
                    file << "Mock...Data";
                }
            }
        }

        void SetUp() override {
            mockDataset();
            mockPreprocessor = new MockPreprocessor();
            audioDataset = new AudioDataset(rootPath, mockPreprocessor);
        }

        void TearDown() override {
            fs::remove_all(rootPath);
            delete mockPreprocessor;
            mockPreprocessor = nullptr;
            delete audioDataset;
            audioDataset = nullptr;
        }
};

TEST_F(TestAudioDataset, getValidOutput) {
    // Mock the Preprocessor.
    size_t numFrames = 1290;
    size_t mfccSize = 13;
    vector<vector<float>> mockMfcc = randomMatrix(numFrames, mfccSize);
    EXPECT_CALL(*mockPreprocessor, run(_)).WillOnce(Return(mockMfcc));

    // Compute the result.
    torch::data::Example<> sample = audioDataset->get(0);

    // Test the result.
    ASSERT_EQ(sample.data.numel(), numFrames * mfccSize) << "Invalid size of the samle data.";
    ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestAudioDataset, sizeValidOutput) {
    // Define the expected result.
    size_t sizeExpected = 6;

    // Compute the result.
    torch::optional<size_t> size = audioDataset->size();

    // Test the result.
    ASSERT_EQ(size.value(), sizeExpected) << "Invalid size of the dataset.";
}

TEST_F(TestAudioDataset, classesValidOutput) {
    // Define the expected result.
    map<string, uint> classesExpected = {
        {"jazz", 0},
        {"rock", 1}
    };

    // Compute the result.
    map<string, uint> classes = audioDataset->getClasses();

    // Test the result.
    ASSERT_EQ(classes.size(), classesExpected.size()) << "Invalid number of classes.";
    for (const auto& pair : classes) {
        ASSERT_EQ(pair.second, classesExpected[pair.first]) << "Invalid class.";
    }
}
