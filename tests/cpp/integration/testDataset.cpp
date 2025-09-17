#include <gtest/gtest.h>

#define private public
#include "preprocessor.h"
#undef private

#include "dataset.h"

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
            fs::path datasetPath = "../../resources";
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

TEST_F(TestDataset, getValidOutput)
{
    // Collect the parameters.
    uint size = audioDataset->preprocessor()->m_size;
    uint hop = audioDataset->preprocessor()->m_hop;
    uint nmfcc = audioDataset->preprocessor()->m_nmfcc;

    // Define the expected result.
    size_t numFramesExpected = size / hop + 1;
    size_t mfccSizeExpected = nmfcc;

    // Compute the result.
    torch::data::Example<> sample = audioDataset->get(0);

    // Test the result.
    ASSERT_EQ(sample.data.numel(), numFramesExpected * mfccSizeExpected) << "Invalid size of the sample data.";
    ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestDataset, sizeValidOutput)
{
    // Define the expected result.
    size_t sizeExpected = 999;

    // Compute the result.
    torch::optional<size_t> size = audioDataset->size();

    // Test the result.
    ASSERT_EQ(size.value(), sizeExpected) << "Invalid size of the dataset.";
}

TEST_F(TestDataset, classesValidOutput)
{
    // Define the expected result.
    c10::Dict<std::string, torch::Tensor> classesExpected;
    classesExpected.insert("blues", torch::tensor(0, torch::kLong));
    classesExpected.insert("classical", torch::tensor(1, torch::kLong));
    classesExpected.insert("country", torch::tensor(2, torch::kLong));
    classesExpected.insert("disco", torch::tensor(3, torch::kLong));
    classesExpected.insert("hiphop", torch::tensor(4, torch::kLong));
    classesExpected.insert("jazz", torch::tensor(5, torch::kLong));
    classesExpected.insert("metal", torch::tensor(6, torch::kLong));
    classesExpected.insert("pop", torch::tensor(7, torch::kLong));
    classesExpected.insert("reggae", torch::tensor(8, torch::kLong));
    classesExpected.insert("rock", torch::tensor(9, torch::kLong));

    // Compute the result.
    c10::Dict<std::string, torch::Tensor> classes = audioDataset->classes();

    // Test the result.
    ASSERT_EQ(classes.size(), classesExpected.size()) << "Invalid number of classes.";
    for (const auto &pair : classes)
    {
        ASSERT_TRUE(classesExpected.contains(pair.key())) << "Invalid class.";
        ASSERT_TRUE(torch::equal(pair.value(), classesExpected.at(pair.key()))) << "Invalid class.";
    }
}
