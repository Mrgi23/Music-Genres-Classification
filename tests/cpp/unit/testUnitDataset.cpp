#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "preprocessor.h"
#include "dataset.h"

using namespace std;
using ::testing::_;
using ::testing::Return;

// Define the mock objects.
class MockPreprocessor : public Preprocessor
{
    public:
        MockPreprocessor(
            uint size = 660000,
            uint nfft = 1024,
            uint hop = 512,
            uint nmels = 128,
            uint nmfcc = 13
        ) : Preprocessor(size, nfft, hop, nmels, nmfcc) {}
        MOCK_METHOD(torch::Tensor, run, (const fs::path & filePath), (override));
        MOCK_METHOD(torch::Tensor, normalize, (const torch::Tensor & x), (override));
};

// Define the test object.
class TestDataset : public ::testing::Test, public ::testing::WithParamInterface<bool>
{
    protected:
        fs::path rootPath;
        AudioDataset * audioDataset = nullptr;
        MockPreprocessor * mockPreprocessor = nullptr;

        void mockDataset(bool preload)
        {
            // Create temporary root dir.
            rootPath = fs::temp_directory_path() / "resources";
            fs::create_directory(rootPath);

            if (!preload)
            {
                // Create temporary genres dirs.
                vector<string> genres = {"jazz", "rock"};
                for (const auto& genre : genres)
                {
                    fs::path dirPath = rootPath / genre;
                    fs::create_directory(dirPath);

                    // Create 3 temporary .wav files
                    for (int i = 0; i < 3; ++i)
                    {
                        fs::path filePath = dirPath / (genre + ".0000" + to_string(i) + ".wav");
                        ofstream file(filePath);
                        file << "Mock...Data";
                    }
                }
            }
            else
            {
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
        }

        void SetUp() override
        {
            // Mock temporary dataset.
            bool preload = GetParam();
            mockDataset(preload);

            // Mock the Preprocessor.
            mockPreprocessor = new MockPreprocessor();
            if (!preload)
            {
                EXPECT_CALL(*mockPreprocessor, run(_)).Times(6).WillRepeatedly(Return(torch::rand({1290, 13})));
            }

            // Initialize the AudioDataset.
            audioDataset = new AudioDataset(rootPath, mockPreprocessor);
        }

        void TearDown() override
        {
            // Cleanup.
            fs::remove_all(rootPath);
            delete mockPreprocessor;
            mockPreprocessor = nullptr;
            delete audioDataset;
            audioDataset = nullptr;
        }
};

TEST_P(TestDataset, getValidOutput)
{
    // Define the expected result.
    long numFramesExpected = 1290;
    long mfccSizeExpected = 13;

    // Mock the Preprocessor.
    EXPECT_CALL(*mockPreprocessor, normalize(_)).WillOnce(Return(torch::rand({numFramesExpected, mfccSizeExpected})));

    // Compute the result.
    torch::data::Example<> sample = audioDataset->get(0);

    // Test the result.
    ASSERT_EQ(sample.data.numel(), numFramesExpected * mfccSizeExpected) << "Invalid size of the sample data.";
    ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_P(TestDataset, sizeValidOutput)
{
    // Define the expected result.
    size_t sizeExpected = 6;

    // Compute the result.
    torch::optional<size_t> size = audioDataset->size();

    // Test the result.
    ASSERT_EQ(size.value(), sizeExpected) << "Invalid size of the dataset.";
}

TEST_P(TestDataset, classesValidOutput)
{
    // Define the expected result.
    c10::Dict<std::string, torch::Tensor> classesExpected;
    classesExpected.insert("jazz", torch::tensor(0, torch::kLong));
    classesExpected.insert("rock", torch::tensor(1, torch::kLong));

    // Compute the result.
    c10::Dict<std::string, torch::Tensor> classes = audioDataset->classes();

    // Test the result.
    ASSERT_EQ(classes.size(), classesExpected.size()) << "Invalid number of classes.";
    for (const auto& pair : classes) {
        ASSERT_TRUE(classesExpected.contains(pair.key())) << "Invalid class.";
        ASSERT_TRUE(torch::equal(pair.value(), classesExpected.at(pair.key()))) << "Invalid class.";
    }
}

INSTANTIATE_TEST_SUITE_P(_, TestDataset, ::testing::Values(true, false));
