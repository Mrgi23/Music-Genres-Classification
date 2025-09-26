#ifndef TRAINER_H
#define TRAINER_H

#ifdef __cplusplus

#include "dataset.h"
#include "model.h"
#include "optimizer.h"
#include "scheduler.h"

using torch::data::datasets::MapDataset;
using torch::data::transforms::Stack;
using StackedAudioSubset = MapDataset<AudioSubset, Stack<torch::data::Example<>>>;

using torch::data::samplers::RandomSampler;
using torch::data::samplers::SequentialSampler;
template <typename Sampler>
using AudioDataloader = torch::data::StatelessDataLoader<StackedAudioSubset, Sampler>;

class DeviceManager
{
    public:
        static torch::Device & get()
        {
            std::string deviceStr = torch::cuda::is_available() ? "cuda" : "cpu";
            static torch::Device device{deviceStr};
            return device;
        }
};

class Trainer
{
    public:
        Trainer(MusicModel & model, OptimizerType type, const OptimizerConfig & cfg);
        ~Trainer();

        void attachScheduler(ReduceLROnPlateau * scheduler);
        void fit(AudioDataloader<RandomSampler> & dataloader, float & loss, float & acc);
        void eval(AudioDataloader<SequentialSampler> & dataloader, float & loss, float & acc);
    private:
        MusicModel & m_model;
        std::unique_ptr<torch::optim::Optimizer> m_opt;
        torch::nn::ModuleHolder<torch::nn::CrossEntropyLossImpl> m_lossFunction;
};

#endif

#endif
