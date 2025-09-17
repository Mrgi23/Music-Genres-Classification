#include <algorithm>
#include <gtest/gtest.h>
#include "preprocessor.h"
#include "dataset.h"
#include "model.h"
#include "trainer.h"

using namespace std;

// Define the test object.
class TestTrainer : public ::testing::Test
{
    protected:
        Preprocessor * preprocessor = nullptr;
        AudioDataset * audioDataset = nullptr;
        AudioSubset * audioSubset = nullptr;
        MusicModel * musicModel = nullptr;
        Trainer * trainer = nullptr;
        int batch_size;

        void SetUp() override
        {
            // Create dataset.
            fs::path datasetPath = "../../resources";
            preprocessor = new Preprocessor();
            audioDataset = new AudioDataset(datasetPath, preprocessor);

            // Create subset of dataset.
            batch_size = 2;
            std::vector<size_t> indices(batch_size);
            iota(indices.begin(), indices.end(), 0);
            audioSubset = new AudioSubset(audioDataset, indices);

            // Initialize the MusicModel.
            musicModel = new MusicModel();

            // Initialize the Trainer.
            torch::optim::Optimizer * adamOpt = new torch::optim::Adam(
                musicModel->get()->parameters(),
                torch::optim::AdamOptions(1e-3)
            );
            trainer = new Trainer(*musicModel, *adamOpt);
        }

        void TearDown() override
        {
            // Cleanup.
            delete preprocessor;
            preprocessor = nullptr;
            delete audioDataset;
            audioDataset = nullptr;
            delete audioSubset;
            audioSubset = nullptr;
            delete musicModel;
            musicModel = nullptr;
            delete trainer;
            trainer = nullptr;
        }
};

TEST_F(TestTrainer, fitValidOutput)
{
    // Create the DataLoader.
    std::unique_ptr<AudioDataloader<RandomSampler>> dataloader = torch::data::make_data_loader<RandomSampler>(
        audioSubset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );

    // Compute the result.
    float avgLoss;
    float acc;
    trainer->fit(*dataloader, avgLoss, acc);

    // Test the result.
    ASSERT_GT(avgLoss, 0) << "Invalid train loss value.";
}

TEST_F(TestTrainer, sizeValidOutput)
{
    // Create the DataLoader.
    std::unique_ptr<AudioDataloader<SequentialSampler>> dataloader = torch::data::make_data_loader<SequentialSampler>(
        audioSubset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );

    // Compute the result.
    float avgLoss;
    float acc;
    trainer->eval(*dataloader, avgLoss, acc);

    // Test the result.
    ASSERT_GT(avgLoss, 0) << "Invalid validation loss value.";
}
