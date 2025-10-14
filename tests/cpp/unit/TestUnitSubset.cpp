#include "Dataset.h"

#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace std;
using ::testing::_;
using ::testing::Return;

// Define the mock objects.
struct MockDataset
{
    static void Mock(fs::path & rootPath, size_t size, long H, long W)
    {
        // Create temporary root dir.
        rootPath = fs::temp_directory_path() / "resources";
        fs::create_directory(rootPath);

        // Create temporary preloaded dataset.
        vector<torch::Tensor> data(size, torch::rand({H, W}));
        vector<torch::Tensor> target(size, torch::rand(1));
        c10::Dict<int64_t, string> classes;;

        // Save dataset.
        fs::path datasetPath = rootPath / "dataset_cpp.pt";
        torch::serialize::OutputArchive dataset;
        dataset.write("data", data);
        dataset.write("target", target);
        dataset.write("classes", classes);
        dataset.save_to(datasetPath);
    }
};

class MockAudioDataset : public AudioDataset
{
    public:
        MockAudioDataset(const fs::path & rootPath, Preprocessor * preprocessor) : AudioDataset(rootPath, preprocessor) {}
        MOCK_METHOD(torch::data::Example<>, get, (size_t index), (override));
};

// Define the test objects.
class TestSubset : public ::testing::Test
{
    protected:
        fs::path rootPath;
        long H, W;
        size_t datasetSize, subsetSize;
        MockAudioDataset * mockAudiodataset = nullptr;
        AudioSubset * audioSubset = nullptr;

        void SetUp() override
        {
            // Mock temporary dataset.
            datasetSize = 6;
            H = 1290;
            W = 13;
            MockDataset::Mock(rootPath, datasetSize, H, W);

            // Mock the AudioDataset.
            mockAudiodataset = new MockAudioDataset(rootPath, nullptr);

            // Initialize AudioSubset.
            subsetSize = 3;
            audioSubset = new AudioSubset(mockAudiodataset, vector<size_t>{0, 1, 2});
        }

        void TearDown() override
        {
            // Cleanup.
            fs::remove_all(rootPath);
            delete mockAudiodataset;
            mockAudiodataset = nullptr;
            delete audioSubset;
            audioSubset = nullptr;
        }
};

TEST_F(TestSubset, get)
{
    // Mock the AudioDataset.
    torch::Tensor data = torch::rand({H, W});
    torch::Tensor target = torch::rand(1);
    EXPECT_CALL(*mockAudiodataset, get(_)).WillOnce(Return(torch::data::Example<>(data, target)));

    // Compute the result.
    torch::data::Example<> sample = audioSubset->get(0);

    // Test the result.
    ASSERT_EQ(sample.data.numel(), H * W) << "Invalid size of the sample data.";
    ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestSubset, size)
{
    // Compute the result.
    torch::optional<size_t> size = audioSubset->size();

    // Test the result.
    ASSERT_EQ(size.value(), subsetSize) << "Invalid size of the dataset.";
}

TEST_F(TestSubset, GetStackedData)
{
    // Mock the AudioDataset.
    torch::Tensor data = torch::rand({H, W});
    torch::Tensor target = torch::rand(1);
    EXPECT_CALL(*mockAudiodataset, get(_)).WillRepeatedly(Return(torch::data::Example<>(data, target)));

    // Compute the result.
    data = audioSubset->GetStackedData();

    // Test the result.
    ASSERT_EQ(data.numel(), subsetSize * H * W) << "Invalid size of the data.";

    // Compute the result, again.
    data = audioSubset->GetStackedData();

    // Test the result.
    ASSERT_EQ(data.numel(), subsetSize * H * W) << "Invalid size of the data.";
}
