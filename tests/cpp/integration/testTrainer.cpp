#include "trainer.hpp"
#include <algorithm>
#include <gtest/gtest.h>

class TestTrainer : public ::testing::Test
{
  protected:
    Preprocessor* preprocessor = nullptr;
    AudioDataset* audioDataset = nullptr;
    AudioSubset* audioSubset = nullptr;
    MusicModel* musicModel = nullptr;
    ReduceLROnPlateau* scheduler = nullptr;
    Trainer* trainer = nullptr;
    int batch_size;

    void SetUp() override
    {
      fs::path datasetPath = "../../../resources";
      preprocessor = new Preprocessor();
      audioDataset = new AudioDataset(datasetPath, preprocessor);

      batch_size = 2;
      std::vector<size_t> indices(batch_size);
      std::iota(indices.begin(), indices.end(), 0);
      audioSubset = new AudioSubset(audioDataset, indices);

      musicModel = new MusicModel();

      scheduler = new ReduceLROnPlateau("max", 0.5, 10);
      OptimizerType type = OptimizerType::Adam;
      OptimizerConfig cfg(1e-3);
      trainer = new Trainer(*musicModel, type, cfg);
    }

    void TearDown() override
    {
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

TEST_F(TestTrainer, attach)
{
  trainer->attach(scheduler);
  EXPECT_NO_THROW(scheduler->update(0.5));
}

TEST_F(TestTrainer, train)
{
  std::unique_ptr<AudioDataloader<RandomSampler>> dataloader = torch::data::make_data_loader<RandomSampler>(
    audioSubset->map(Stack<>()),
    torch::data::DataLoaderOptions().batch_size(batch_size));

  float avgLoss;
  float acc;
  trainer->train(*dataloader, avgLoss, acc);
  ASSERT_GT(avgLoss, 0) << "Invalid train loss value.";
}

TEST_F(TestTrainer, eval)
{
  std::unique_ptr<AudioDataloader<SequentialSampler>> dataloader = torch::data::make_data_loader<SequentialSampler>(
      audioSubset->map(Stack<>()),
      torch::data::DataLoaderOptions().batch_size(batch_size));

  float avgLoss;
  float acc;
  trainer->eval(*dataloader, avgLoss, acc);
  ASSERT_GT(avgLoss, 0) << "Invalid validation loss value.";
}
