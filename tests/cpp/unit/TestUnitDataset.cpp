#include "Dataset.h"

#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace std;
using ::testing::_;
using ::testing::Return;
using ::testing::Throw;

// Define the mock objects.
struct MockDataset
{
    static void Mock(fs::path & rootPath, bool preload, size_t size, long H, long W)
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
                for (int i = 0; i < size / genres.size(); ++i)
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
            vector<torch::Tensor> data(size, torch::rand({H, W}));
            vector<torch::Tensor> target(size, torch::rand(1));
            c10::Dict<int64_t, string> classes;
            classes.insert(0, "jazz");
            classes.insert(1, "rock");

            // Save dataset.
            fs::path datasetPath = rootPath / "dataset_cpp.pt";
            torch::serialize::OutputArchive dataset;
            dataset.write("data", data);
            dataset.write("target", target);
            dataset.write("classes", classes);
            dataset.save_to(datasetPath);
        }
    }
};

class MockPreprocessor : public Preprocessor
{
    public:
        MockPreprocessor(PreprocessorConfig cfg = PreprocessorConfig()) : Preprocessor(cfg) {}
        MOCK_METHOD(torch::Tensor, ProcessFile, (const fs::path & filePath), (override));
        MOCK_METHOD(torch::Tensor, NormalizeData, (const torch::Tensor & x), (override));
};

// Define the test object.
class TestInitializationParam : public ::testing::TestWithParam<string>
{
    protected:
        fs::path rootPath;
        long H, W;
        size_t datasetSize;
        AudioDataset * audioDataset = nullptr;
        MockPreprocessor * mockPreprocessor = nullptr;

        void SetUp() override
        {
            // Mock temporary dataset.
            string init = GetParam();
            datasetSize = 6;
            H = 1290;
            W = 13;
            MockDataset::Mock(rootPath, (init == "preload") ? true : false, datasetSize, H, W);

            // Mock the Preprocessor.
            mockPreprocessor = new MockPreprocessor();
            if (init == "corrupt")
            {
                EXPECT_CALL(*mockPreprocessor, ProcessFile(_))
                .WillOnce(Throw(runtime_error("Preprocessor::LoadAndCrop: Invalid or corrupted file.")))
                .WillRepeatedly(Return(torch::rand({1, H, W})));
                ASSERT_THROW(audioDataset = new AudioDataset(rootPath, mockPreprocessor), runtime_error);
            }
            else if (init == "init")
            {
               EXPECT_CALL(*mockPreprocessor, ProcessFile(_))
               .WillRepeatedly(Return(torch::rand({1, H, W})));
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

TEST_P(TestInitializationParam, get)
{
    // Mock the Preprocessor.
    EXPECT_CALL(*mockPreprocessor, NormalizeData(_)).WillOnce(Return(torch::rand({1, H, W})));

    // Compute the result.
    torch::data::Example<> sample = audioDataset->get(0);

    // Test the result.
    ASSERT_EQ(sample.data.numel(), H * W) << "Invalid size of the sample data.";
    ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

INSTANTIATE_TEST_SUITE_P(
    TestDatasetWithParams,
    TestInitializationParam,
    ::testing::Values("init", "corrupt", "preload"),
    [](const testing::TestParamInfo<string> & info)
    {
        return info.param;
    }
);

// Define the test object.
class TestDataset : public ::testing::Test
{
    protected:
        fs::path rootPath;
        long H, W;
        size_t datasetSize;
        AudioDataset * audioDataset = nullptr;

        void SetUp() override
        {
            // Mock temporary dataset.
            datasetSize = 6;
            H = 1290;
            W = 13;
            MockDataset::Mock(rootPath, true, datasetSize, H, W);

            // Initialize the AudioDataset.
            audioDataset = new AudioDataset(rootPath, nullptr);
        }

        void TearDown() override
        {
            // Cleanup.
            fs::remove_all(rootPath);
            delete audioDataset;
            audioDataset = nullptr;
        }
};

TEST_F(TestDataset, GetClasses)
{
    // Define the expected result.
    c10::Dict<int64_t, string> classesExpected;
    classesExpected.insert(0, "jazz");
    classesExpected.insert(1, "rock");

    // Compute the result.
    c10::Dict<int64_t, string> classes = audioDataset->GetClasses();

    // Test the result.
    ASSERT_EQ(classes.size(), classesExpected.size()) << "Invalid number of classes.";
    for (const auto& pair : classes) {
        ASSERT_TRUE(classesExpected.contains(pair.key())) << "Invalid class.";
        ASSERT_EQ(pair.value(), classesExpected.at(pair.key())) << "Invalid class.";
    }
}

TEST_F(TestDataset, size)
{
    // Compute the result.
    torch::optional<size_t> size = audioDataset->size();

    // Test the result.
    ASSERT_EQ(size.value(), datasetSize) << "Invalid size of the dataset.";
}
