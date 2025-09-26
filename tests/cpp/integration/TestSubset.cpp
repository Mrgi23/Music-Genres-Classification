#include "Dataset.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <random>

using namespace std;

// Define the test object.
class TestSubset : public ::testing::Test
{
    protected:
        Preprocessor * preprocessor = nullptr;
        AudioDataset * audioDataset = nullptr;
        AudioSubset * audioSubset = nullptr;

        void SetUp() override
        {
            // Create dataset.
            fs::path datasetPath = "../../resources";
            preprocessor = new Preprocessor();
            audioDataset = new AudioDataset(datasetPath, preprocessor);

            // Create pool of indices.
            vector<size_t> pool(999);
            iota(pool.begin(), pool.end(), 0);

            // Set up a random number generator and shuffle indices.
            random_device rd;
            mt19937 g(rd());
            shuffle(pool.begin(), pool.end(), g);

            // Select random indices.
            vector<size_t> indices(pool.begin(), pool.begin() + 500);

            // Create subset.
            audioSubset = new AudioSubset(audioDataset, indices);
        }

        void TearDown() override
        {
            delete preprocessor;
            preprocessor = nullptr;
            delete audioDataset;
            audioDataset = nullptr;
            delete audioSubset;
            audioSubset = nullptr;
        }
};

TEST_F(TestSubset, get)
{
    // Collect the parameters.
    uint size = audioSubset->GetDataset()->GetPreprocessor()->GetCfg().size;
    uint hop = audioSubset->GetDataset()->GetPreprocessor()->GetCfg().hop;
    uint nmfcc = audioSubset->GetDataset()->GetPreprocessor()->GetCfg().nmfcc;

    // Define the expected result.
    size_t numFramesExpected = size / hop + 1;
    size_t mfccSizeExpected = nmfcc;

    // Compute the result.
    torch::data::Example<> sample = audioSubset->get(0);

    // Test the result.
    ASSERT_EQ(sample.data.numel(), numFramesExpected * mfccSizeExpected) << "Invalid size of the sample data.";
    ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestSubset, size)
{
    // Define the expected result.
    size_t sizeExpected = 500;

    // Compute the result.
    torch::optional<size_t> size = audioSubset->size();

    // Test the result.
    ASSERT_EQ(size.value(), sizeExpected) << "Invalid size of the dataset.";
}

TEST_F(TestSubset, GetStackedData)
{
    // Collect the parameters.
    uint size = audioSubset->GetDataset()->GetPreprocessor()->GetCfg().size;
    uint hop = audioSubset->GetDataset()->GetPreprocessor()->GetCfg().hop;
    uint nmfcc = audioSubset->GetDataset()->GetPreprocessor()->GetCfg().nmfcc;

    // Define the expected result.
    size_t numSamplesExpected = 500;
    size_t numFramesExpected = size / hop + 1;
    size_t mfccSizeExpected = nmfcc;

    // Compute the result.
    torch::Tensor data = audioSubset->GetStackedData();

    // Test the result.
    ASSERT_EQ(data.numel(), numSamplesExpected * numFramesExpected * mfccSizeExpected) << "Invalid size of the data.";
}
