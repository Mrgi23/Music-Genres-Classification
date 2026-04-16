#include "Trainer.h"

#include <algorithm>
#include <gtest/gtest.h>

using namespace std;

// Define the test object.
class TestTrainer : public ::testing::Test
{
    protected:
        Preprocessor * preprocessor = nullptr;
        AudioDataset * audioDataset = nullptr;
        AudioSubset * audioSubset = nullptr;
        MusicModel * musicModel = nullptr;
        ReduceLROnPlateau * scheduler = nullptr;
        Trainer * trainer = nullptr;
        int batch_size;

        void SetUp() override
        {
            // Create dataset.
            fs::path datasetPath = "../../../resources";
            preprocessor = new Preprocessor();
            audioDataset = new AudioDataset(datasetPath, preprocessor);

            // Create subset of dataset.
            batch_size = 2;
            vector<size_t> indices(batch_size);
            iota(indices.begin(), indices.end(), 0);
            audioSubset = new AudioSubset(audioDataset, indices);

            // Initialize the MusicModel.
            musicModel = new MusicModel();

            // Initialize the ReduceLROnPlateau optimizer.
            scheduler = new ReduceLROnPlateau("max", 0.5, 10);

            // Initialize the Trainer.
            OptimizerType type = OptimizerType::Adam;
            OptimizerConfig cfg(1e-3);
            trainer = new Trainer(*musicModel, type, cfg);
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
            delete scheduler;
            scheduler = nullptr;
            delete trainer;
            trainer = nullptr;
        }
};

TEST_F(TestTrainer, AttachScheduler)
{
    trainer->AttachScheduler(scheduler);
    EXPECT_NO_THROW(scheduler->UpdateLR(0.5));
}

TEST_F(TestTrainer, TrainModel)
{
    // Create the DataLoader.
    unique_ptr<AudioDataloader<RandomSampler>> dataloader = torch::data::make_data_loader<RandomSampler>(
        audioSubset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );

    // Compute the result.
    float avgLoss;
    float acc;
    trainer->TrainModel(*dataloader, avgLoss, acc);

    // Test the result.
    ASSERT_GT(avgLoss, 0) << "Invalid train loss value.";
}

TEST_F(TestTrainer, EvalModel)
{
    // Create the DataLoader.
    unique_ptr<AudioDataloader<SequentialSampler>> dataloader = torch::data::make_data_loader<SequentialSampler>(
        audioSubset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );

    // Compute the result.
    float avgLoss;
    float acc;
    trainer->EvalModel(*dataloader, avgLoss, acc);

    // Test the result.
    ASSERT_GT(avgLoss, 0) << "Invalid validation loss value.";
}
