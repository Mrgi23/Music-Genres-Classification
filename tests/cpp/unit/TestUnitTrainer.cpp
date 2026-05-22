#include "Trainer.h"
#include <algorithm>
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

struct MockDataset
{
  static void Mock(fs::path& rootPath, size_t size, long H, long W)
  {
    rootPath = fs::temp_directory_path() / "resources";
    fs::create_directory(rootPath);

    std::vector<torch::Tensor> data(size, torch::rand({H, W}));
    std::vector<torch::Tensor> target(size, torch::rand(1));
    c10::Dict<int64_t, std::string> classes;

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
    MockAudioDataset(const fs::path& rootPath, Preprocessor* preprocessor) : AudioDataset(rootPath, preprocessor) {}
    MOCK_METHOD(torch::data::Example<>, get, (size_t index), (override));
};

class MockMusicModelImpl : public MusicModelImpl
{
  public:
    MockMusicModelImpl() : MusicModelImpl() {}
    MOCK_METHOD(torch::Tensor, forward, (torch::Tensor x), (override));
};

class TestTrainer : public ::testing::Test
{
  protected:
    fs::path rootPath;
    long H, W, datasetSize, outputSize;
    MockAudioDataset* mockAudioDataset = nullptr;
    AudioSubset* audioSubset = nullptr;
    std::shared_ptr<MockMusicModelImpl> mockMusicModelImpl;
    MusicModel* mockMusicModel = nullptr;
    ReduceLROnPlateau* scheduler = nullptr;
    Trainer* trainer = nullptr;

    void SetUp() override
    {
      datasetSize = 6;
      outputSize = 10;
      H = 1290;
      W = 13;
      MockDataset::Mock(rootPath, datasetSize, H, W);
      mockAudioDataset = new MockAudioDataset(rootPath, nullptr);

      std::vector<size_t> indices(datasetSize);
      std::iota(indices.begin(), indices.end(), 0);
      audioSubset = new AudioSubset(mockAudioDataset, indices);

      mockMusicModelImpl = std::make_shared<MockMusicModelImpl>();
      mockMusicModel = new MusicModel(std::static_pointer_cast<MusicModelImpl>(mockMusicModelImpl));

      scheduler = new ReduceLROnPlateau("max", 0.5, 10);
      OptimizerType type = OptimizerType::Adam;
      OptimizerConfig cfg(1e-3);
      trainer = new Trainer(*mockMusicModel, type, cfg);
    }

    void TearDown() override
    {
      fs::remove_all(rootPath);
      delete mockAudioDataset;
      mockAudioDataset = nullptr;
      delete audioSubset;
      audioSubset = nullptr;
      delete mockMusicModel;
      mockMusicModel = nullptr;
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
  float avgLossExpected = logf(10);

  std::unique_ptr<AudioDataloader<RandomSampler>> mockDataloader = torch::data::make_data_loader<RandomSampler>(
    audioSubset->map(Stack<>()),
    torch::data::DataLoaderOptions().batch_size(datasetSize)
  );
  torch::Tensor data = torch::zeros({1, H, W});
  torch::Tensor target = torch::randint(outputSize, {}, torch::kLong);
  EXPECT_CALL(*mockAudioDataset, get(_)).WillRepeatedly(Return(torch::data::Example<>(data, target)));
  EXPECT_CALL(*mockMusicModelImpl, forward(_))
    .WillOnce(Return(
      torch::zeros(
        {datasetSize, outputSize},
        torch::TensorOptions().device(DeviceManager::get()).requires_grad(true)
      )
    )
  );

  float avgLoss;
  float acc;
  trainer->train(*mockDataloader, avgLoss, acc);
  ASSERT_NEAR(avgLoss, avgLossExpected, 1e-6) << "Invalid train loss value.";
}

TEST_F(TestTrainer, eval)
{
  float avgLossExpected = logf(10);
  std::unique_ptr<AudioDataloader<SequentialSampler>> mockDataloader = torch::data::make_data_loader<SequentialSampler>(
    audioSubset->map(Stack<>()),
    torch::data::DataLoaderOptions().batch_size(datasetSize)
  );
  torch::Tensor data = torch::zeros({1, H, W});
  torch::Tensor target = torch::randint(outputSize, {}, torch::kLong);
  EXPECT_CALL(*mockAudioDataset, get(_)).WillRepeatedly(Return(torch::data::Example<>(data, target)));
  EXPECT_CALL(*mockMusicModelImpl, forward(_))
    .WillOnce(Return(
      torch::zeros(
        {datasetSize, outputSize},
        torch::TensorOptions().device(DeviceManager::get()).requires_grad(false)
      )
    )
  );

  float avgLoss;
  float acc;
  trainer->eval(*mockDataloader, avgLoss, acc);
  ASSERT_NEAR(avgLoss, avgLossExpected, 1e-6) << "Invalid validation loss value.";
}
