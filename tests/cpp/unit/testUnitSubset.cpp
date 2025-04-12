#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "preprocessor.h"
#include "dataset.h"

using namespace std;
using ::testing::_;
using ::testing::Return;

// Define the mock objects.
class MockAudioDataset : public AudioDataset {
    public:
        MockAudioDataset(const fs::path& rootPath, Preprocessor * preprocessor) : AudioDataset(rootPath, preprocessor) {}

        MOCK_METHOD(torch::data::Example<>, get, (size_t index), (override));
};

// Define the test objects.
class TestSubset : public ::testing::Test {
    protected:
        fs::path rootPath;
        MockAudioDataset * mockAudiodataset = nullptr;
        AudioSubset * audioSubset = nullptr;

        void mockDataset() {
            // Create temporary root dir.
            rootPath = fs::temp_directory_path() / "resources";
            fs::create_directory(rootPath);

            // Create temporary preloaded dataset.
            vector<torch::Tensor> data(6, torch::rand({1290, 13}));
            vector<torch::Tensor> target(6, torch::rand(1));
            c10::Dict<std::string, torch::Tensor> classes;
            classes.insert("jazz", torch::tensor(0, torch::kLong));
            classes.insert("rock", torch::tensor(1, torch::kLong));

            // Save dataset.
            fs::path datasetPath = rootPath / "dataset_cpp.pt";
            torch::serialize::OutputArchive dataset;
            dataset.write("data", data);
            dataset.write("target", target);
            dataset.write("classes", classes);
            dataset.save_to(datasetPath);
        }

        void SetUp() override {
            // Mock temporary dataset.
            mockDataset();

            // Mock the AudioDataset.
            mockAudiodataset = new MockAudioDataset(rootPath, nullptr);

            // Initialize AudioSubset.
            audioSubset = new AudioSubset(mockAudiodataset, vector<size_t>{0, 1, 2});
        }

        void TearDown() override {
            // Cleanup.
            fs::remove_all(rootPath);
            delete mockAudiodataset;
            mockAudiodataset = nullptr;
            delete audioSubset;
            audioSubset = nullptr;
        }
};

TEST_F(TestSubset, getValidOutput) {
    // Define the expected result.
    long numFramesExpected = 1290;
    long mfccSizeExpected = 13;

    // Mock the AudioDataset.
    torch::Tensor data = torch::rand({numFramesExpected, mfccSizeExpected});
    torch::Tensor target = torch::rand(1);
    EXPECT_CALL(*mockAudiodataset, get(_)).WillOnce(Return(torch::data::Example<>(data, target)));

    // Compute the result.
    torch::data::Example<> sample = audioSubset->get(0);

    // Test the result.
    ASSERT_EQ(sample.data.numel(), numFramesExpected * mfccSizeExpected) << "Invalid size of the sample data.";
    ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestSubset, sizeValidOutput) {
    // Define the expected result.
    size_t sizeExpected = 3;

    // Compute the result.
    torch::optional<size_t> size = audioSubset->size();

    // Test the result.
    ASSERT_EQ(size.value(), sizeExpected) << "Invalid size of the dataset.";
}

TEST_F(TestSubset, dataValidOutput) {
    // Define the expected result.
    size_t numSamplesExpected = 3;
    long numFramesExpected = 1290;
    long mfccSizeExpected = 13;

    // Mock the AudioDataset.
    torch::Tensor data = torch::rand({numFramesExpected, mfccSizeExpected});
    torch::Tensor target = torch::rand(1);
    EXPECT_CALL(*mockAudiodataset, get(_)).Times(3).WillRepeatedly(Return(torch::data::Example<>(data, target)));

    // Compute the result.
    data = audioSubset->data();

    // Test the result.
    ASSERT_EQ(data.numel(), numSamplesExpected * numFramesExpected * mfccSizeExpected) << "Invalid size of the data.";
}
