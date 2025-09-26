#include "Trainer.h"

#include <algorithm>
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace std;
using ::testing::_;
using ::testing::Return;
using ::testing::Invoke;

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
        long H, W, datasetSize, outputSize;
        MockAudioDataset * mockAudioDataset = nullptr;
        AudioSubset * audioSubset = nullptr;
        shared_ptr<MockMusicModelImpl> mockMusicModelImpl;
        MusicModel * mockMusicModel = nullptr;
        ReduceLROnPlateau * scheduler = nullptr;
        Trainer * trainer = nullptr;

        void SetUp() override
        {
            // Mock temporary dataset.
            datasetSize = 6;
            outputSize = 10;
            H = 1290;
            W = 13;
            MockDataset::Mock(rootPath, datasetSize, H, W);

            // Mock the AudioDataset.
            mockAudioDataset = new MockAudioDataset(rootPath, nullptr);

            // Create actuall subset.
            vector<size_t> indices(datasetSize);
            iota(indices.begin(), indices.end(), 0);
            audioSubset = new AudioSubset(mockAudioDataset, indices);

            // Mock the MusicModel
            mockMusicModelImpl = make_shared<MockMusicModelImpl>();
            mockMusicModel = new MusicModel(static_pointer_cast<MusicModelImpl>(mockMusicModelImpl));

            // Initialize the ReduceLROnPlateau scheduler.
            scheduler = new ReduceLROnPlateau("max", 0.5, 10);

            // Initialize the Trainer.
            OptimizerType type = OptimizerType::Adam;
            OptimizerConfig cfg(1e-3);
            trainer = new Trainer(*mockMusicModel, type, cfg);
        }

        void TearDown() override
        {
            // Cleanup.
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

TEST_F(TestTrainer, AttachScheduler)
{
    trainer->AttachScheduler(scheduler);
    EXPECT_NO_THROW(scheduler->UpdateLR(0.5));
}

TEST_F(TestTrainer, fitValidOutput)
{
    // Define the expected result.
    float avgLossExpected = logf(10);

    // Mock the DataLoader.
    unique_ptr<AudioDataloader<RandomSampler>> mockDataloader = torch::data::make_data_loader<RandomSampler>(
        audioSubset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(datasetSize)
    );
    torch::Tensor data = torch::zeros({1, H, W});
    torch::Tensor target = torch::randint(outputSize, {}, torch::kLong);
    EXPECT_CALL(*mockAudioDataset, get(_)).WillRepeatedly(Return(torch::data::Example<>(data, target)));

    // Mock the MusicModel.
    EXPECT_CALL(*mockMusicModelImpl, forward(_)).WillOnce(Return(
        torch::zeros({datasetSize, outputSize}, torch::TensorOptions().device(DeviceManager::Get()).requires_grad(true))
    ));

    // Compute the result.
    float avgLoss;
    float acc;
    trainer->TrainModel(*mockDataloader, avgLoss, acc);

    // Test the result.
    ASSERT_NEAR(avgLoss, avgLossExpected, 1e-6) << "Invalid train loss value.";
}

TEST_F(TestTrainer, EvalModel)
{
    // Define the expected result.
    float avgLossExpected = logf(10);

    // Mock the DataLoader.
    unique_ptr<AudioDataloader<SequentialSampler>> mockDataloader = torch::data::make_data_loader<SequentialSampler>(
        audioSubset->map(Stack<>()),
        torch::data::DataLoaderOptions().batch_size(datasetSize)
    );
    torch::Tensor data = torch::zeros({1, H, W});
    torch::Tensor target = torch::randint(outputSize, {}, torch::kLong);
    EXPECT_CALL(*mockAudioDataset, get(_)).WillRepeatedly(Return(torch::data::Example<>(data, target)));

    // Mock the MusicModel.
    EXPECT_CALL(*mockMusicModelImpl, forward(_)).WillOnce(Return(
        torch::zeros({datasetSize, outputSize}, torch::TensorOptions().device(DeviceManager::Get()).requires_grad(false))
    ));

    // Compute the result.
    float avgLoss;
    float acc;
    trainer->EvalModel(*mockDataloader, avgLoss, acc);

    // Test the result.
    ASSERT_NEAR(avgLoss, avgLossExpected, 1e-6) << "Invalid validation loss value.";
}
