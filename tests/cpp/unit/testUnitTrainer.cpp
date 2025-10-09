#include <algorithm>
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "preprocessor.h"
#include "dataset.h"
#include "model.h"
#include "optimizer.h"
#include "scheduler.h"
#include "trainer.h"

using namespace std;
using ::testing::_;
using ::testing::Return;
using ::testing::Invoke;

#include <iostream>

// Define the mock objects.
class MockAudioDataset : public AudioDataset
{
    public:
        MockAudioDataset(const fs::path & rootPath, Preprocessor * preprocessor) : AudioDataset(rootPath, preprocessor) {}
        MOCK_METHOD(torch::data::Example<>, get, (size_t index), (override));
};
class MockMusicModelImpl : public MusicModelImpl
{
    public:
        MockMusicModelImpl() : MusicModelImpl() {}
        MOCK_METHOD(torch::Tensor, forward, (torch::Tensor x), (override));
};

// Define the test objects.
class TestTrainer : public ::testing::Test
{
    protected:
        fs::path rootPath;
        MockAudioDataset * mockAudioDataset = nullptr;
        AudioSubset * audioSubset = nullptr;
        std::shared_ptr<MockMusicModelImpl> mockMusicModelImpl;
        MusicModel * mockMusicModel = nullptr;
        ReduceLROnPlateau * scheduler = nullptr;
        Trainer * trainer = nullptr;
        int batch_size;

        void mockDataset()
        {
            // Create temporary root dir.
            rootPath = fs::temp_directory_path() / "resources";
            fs::create_directory(rootPath);

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

        void SetUp() override
        {
            // Mock temporary dataset.
            mockDataset();

            // Mock the AudioDataset.
            mockAudioDataset = new MockAudioDataset(rootPath, nullptr);

            // Create actuall subset.
            batch_size = 6;
            std::vector<size_t> indices(batch_size);
            std::iota(indices.begin(), indices.end(), 0);
            audioSubset = new AudioSubset(mockAudioDataset, indices);

            // Mock the MusicModel
            mockMusicModelImpl = std::make_shared<MockMusicModelImpl>();
            mockMusicModel = new MusicModel(std::static_pointer_cast<MusicModelImpl>(mockMusicModelImpl));

            scheduler = new ReduceLROnPlateau("max", 0.5, 10);

            // Initialize the Trainer.
            OptimizerType type = OptimizerType::Adam;
            OptimizerConfig cfg(1e-3);
            trainer = new Trainer(*mockMusicModel, type, cfg);
        }

        void TearDown() override
        {
            // Cleanup.
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

TEST_F(TestTrainer, attachSchedulerOutput)
{
    trainer->attachScheduler(scheduler);
    EXPECT_NO_THROW(scheduler->step(0.5));
}

TEST_F(TestTrainer, fitValidOutput)
{
    // Define the expected result.
    float avgLossExpected = logf(10);

    // Mock the DataLoader.
    std::unique_ptr<AudioDataloader<RandomSampler>> mockDataloader = torch::data::make_data_loader<RandomSampler>(
        audioSubset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );
    torch::Tensor data = torch::zeros({1, 1290, 13});
    torch::Tensor target = torch::randint(10, {}, torch::kLong);
    EXPECT_CALL(*mockAudioDataset, get(_)).Times(batch_size).WillRepeatedly(Return(torch::data::Example<>(data, target)));

    // Mock the MusicModel.
    EXPECT_CALL(*mockMusicModelImpl, forward(_)).WillOnce(Return(
        torch::zeros({batch_size, 10}, torch::TensorOptions().device(DeviceManager::get()).requires_grad(true))
    ));

    // Compute the result.
    float avgLoss;
    float acc;
    trainer->fit(*mockDataloader, avgLoss, acc);

    // Test the result.
    ASSERT_NEAR(avgLoss, avgLossExpected, 1e-6) << "Invalid train loss value.";
}

TEST_F(TestTrainer, evalValidOutput)
{
    // Define the expected result.
    float avgLossExpected = logf(10);

    // Mock the DataLoader.
    std::unique_ptr<AudioDataloader<SequentialSampler>> mockDataloader = torch::data::make_data_loader<SequentialSampler>(
        audioSubset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );
    torch::Tensor data = torch::zeros({1, 1290, 13});
    torch::Tensor target = torch::randint(10, {}, torch::kLong);
    EXPECT_CALL(*mockAudioDataset, get(_)).Times(batch_size).WillRepeatedly(Return(torch::data::Example<>(data, target)));

    // Mock the MusicModel.
    EXPECT_CALL(*mockMusicModelImpl, forward(_)).WillOnce(Return(
        torch::zeros({batch_size, 10}, torch::TensorOptions().device(DeviceManager::get()).requires_grad(false))
    ));

    // Compute the result.
    float avgLoss;
    float acc;
    trainer->eval(*mockDataloader, avgLoss, acc);

    // Test the result.
    ASSERT_NEAR(avgLoss, avgLossExpected, 1e-6) << "Invalid validation loss value.";
}
