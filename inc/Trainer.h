#pragma once

#ifdef __cplusplus

#include "Dataset.h"
#include "Model.h"
#include "Scheduler.h"
#include "Optimizer.h"

#include <torch/cuda.h>
#include <torch/data/dataloader.h>
#include <torch/data/datasets/map.h>
#include <torch/data/samplers/random.h>
#include <torch/data/samplers/sequential.h>
#include <torch/data/transforms/stack.h>
#include <torch/nn/modules/loss.h>

using torch::data::datasets::MapDataset;
using torch::data::transforms::Stack;
using StackedAudioSubset = MapDataset<AudioSubset, Stack<torch::data::Example<>>>;

using torch::data::samplers::RandomSampler;
using torch::data::samplers::SequentialSampler;
template <typename Sampler>
using AudioDataloader = torch::data::StatelessDataLoader<StackedAudioSubset, Sampler>;

/**
 * @class Device
 * @brief Provides a globally accessible torch::Device instance.
 *
 * The Device class encapsulates device selection logic for the project.
 * It automatically checks if CUDA is available and selects a GPU device;
 * otherwise, it falls back to CPU. The device is constructed once and
 * returned as a reference on subsequent calls to ensure consistent usage
 * across the codebase.
 */
class DeviceManager
{
    public:
        /**
         * @brief Get the torch::Device reference.
         *
         * On first call, determines the appropriate device ("cuda" if available,
         * otherwise "cpu") and constructs a static torch::Device object.
         * Subsequent calls return a reference to the same device.
         *
         * @return torch::Device Global reference.
         */
        static torch::Device & Get()
        {
            std::string deviceStr = torch::cuda::is_available() ? "cuda" : "cpu";
            static torch::Device device{deviceStr};
            return device;
        }
};

/**
 * @class Trainer
 * @brief Handles training and evaluation of the music classification model.
 *
 * The Trainer class manages the optimization process, including
 * forward/backward passes, loss calculation, parameter updates, and optional
 * learning rate scheduling.
 */
class Trainer
{
    public:
        /**
         * @brief Construct a new Trainer object.
         *
         * Initializes the model, optimizer, and loss function.
         * Automatically selects CUDA if available, otherwise uses CPU.
         *
         * @param[in] model Reference to the music model to be trained.
         * @param[in] type Optimizer type (e.g., Adam, SGD).
         * @param[in] cfg Configuration parameters for the optimizer.
         */
        Trainer(MusicModel & model, const OptimizerType & type, const OptimizerConfig & cfg);
        /**
         * @brief Destructor.
         *
         * Default cleanup.
         */
        ~Trainer();

        /**
         * @brief Attach a learning rate scheduler.
         *
         * Links an external scheduler to the optimizer so that learning
         * rate can be dynamically adjusted during training.
         *
         * @param[in, out] scheduler Pointer to a ReduceLROnPlateau scheduler.
         */
        void AttachScheduler(ReduceLROnPlateau * scheduler);
        /**
         * @brief Train the model for one epoch.
         *
         * Iterates over batches from the provided dataloader, performs forward
         * and backward passes, updates model parameters, and computes average
         * loss and accuracy across all samples.
         *
         * @param[in, out] dataloader Data loader providing training batches.
         * @param[in, out] loss Average loss over all batches.
         * @param[in, out] acc Average accuracy over all samples.
         */
        void TrainModel(AudioDataloader<RandomSampler> & dataloader, float & loss, float & acc);
        /**
         * @brief Evaluate the model without gradient updates.
         *
         * Iterates over batches from the provided dataloader in evaluation mode,
         * computes loss and accuracy, but does not update model parameters.
         *
         * @param[in, out] dataloader Data loader providing evaluation batches.
         * @param[in, out] loss Average loss over all batches.
         * @param[in, out] acc Average accuracy over all samples.
         */
        void EvalModel(AudioDataloader<SequentialSampler> & dataloader, float & loss, float & acc);
    private:
        MusicModel & m_model;
        std::unique_ptr<torch::optim::Optimizer> m_opt;
        torch::nn::ModuleHolder<torch::nn::CrossEntropyLossImpl> m_lossFunction;
};

#endif
