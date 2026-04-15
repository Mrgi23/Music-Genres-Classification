#include "Dataset.h"

#include <gtest/gtest.h>

using namespace std;

// Define the test object.
class TestDataset : public ::testing::Test
{
    protected:
        AudioDataset * audioDataset = nullptr;
        Preprocessor * preprocessor = nullptr;

        void SetUp() override
        {
            // Create dataset.
            fs::path datasetPath = "../../../resources";
            preprocessor = new Preprocessor();
            audioDataset = new AudioDataset(datasetPath, preprocessor);
        }

        void TearDown() override
        {
            delete preprocessor;
            preprocessor = nullptr;
            delete audioDataset;
            audioDataset = nullptr;
        }
};

TEST_F(TestDataset, GetClasses)
{
    // Define the expected result.
    c10::Dict<int64_t, string> classesExpected;
    classesExpected.insert(0, "blues");
    classesExpected.insert(1, "classical");
    classesExpected.insert(2, "country");
    classesExpected.insert(3, "disco");
    classesExpected.insert(4, "hiphop");
    classesExpected.insert(5, "jazz");
    classesExpected.insert(6, "metal");
    classesExpected.insert(7, "pop");
    classesExpected.insert(8, "reggae");
    classesExpected.insert(9, "rock");

    // Compute the result.
    c10::Dict<int64_t, string> classes = audioDataset->GetClasses();

    // Test the result.
    ASSERT_EQ(classes.size(), classesExpected.size()) << "Invalid number of classes.";
    for (const auto &pair : classes)
    {
        ASSERT_TRUE(classesExpected.contains(pair.key())) << "Invalid class.";
        ASSERT_EQ(pair.value(), classesExpected.at(pair.key())) << "Invalid class.";
    }
}

TEST_F(TestDataset, get)
{
    // Collect the parameters.
    uint size = audioDataset->GetPreprocessor()->GetCfg().size;
    uint hop = audioDataset->GetPreprocessor()->GetCfg().hop;
    uint nmfcc = audioDataset->GetPreprocessor()->GetCfg().nmfcc;

    // Define the expected result.
    size_t numFramesExpected = size / hop + 1;
    size_t mfccSizeExpected = nmfcc;

    // Compute the result.
    torch::data::Example<> sample = audioDataset->get(0);

    // Test the result.
    ASSERT_EQ(sample.data.numel(), numFramesExpected * mfccSizeExpected) << "Invalid size of the sample data.";
    ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestDataset, size)
{
    // Define the expected result.
    size_t sizeExpected = 999;

    // Compute the result.
    torch::optional<size_t> size = audioDataset->size();

    // Test the result.
    ASSERT_EQ(size.value(), sizeExpected) << "Invalid size of the dataset.";
}
