#ifndef TRAINER_H
#define TRAINER_H

#ifdef __cplusplus

#include "dataset.h"
#include "model.h"

using torch::data::datasets::MapDataset;
using torch::data::transforms::Stack;
using StackedAudioSubset = MapDataset<AudioSubset, Stack<torch::data::Example<>>>;

using torch::data::samplers::RandomSampler;
using torch::data::samplers::SequentialSampler;
template <typename Sampler>
using AudioDataloader = torch::data::StatelessDataLoader<StackedAudioSubset, Sampler>;

class Trainer {
    private:
        MusicModel& model;
        torch::optim::Optimizer& opt;
        torch::nn::ModuleHolder<torch::nn::CrossEntropyLossImpl> lossFunction;
        torch::Device device{"cpu"};
    public:
        Trainer(
            MusicModel& model,
            torch::optim::Optimizer& opt
        ) : model(model), opt(opt), lossFunction(torch::nn::CrossEntropyLoss()) {
            device = torch::cuda::is_available() ? torch::Device{"cuda"} : torch::Device{"cpu"};
            model->to(device);
        }
        ~Trainer() {}

        void fit(AudioDataloader<RandomSampler> &dataloader, float& loss, float& acc);
        void eval(AudioDataloader<SequentialSampler> &dataloader, float& loss, float& acc);
};


#endif

#endif
