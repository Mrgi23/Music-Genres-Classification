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

class Trainer
{
    public:
        Trainer(MusicModel & model, torch::optim::Optimizer & opt);
        ~Trainer();

        void fit(AudioDataloader<RandomSampler> & dataloader, float & loss, float & acc);
        void eval(AudioDataloader<SequentialSampler> & dataloader, float & loss, float & acc);
    private:
        MusicModel & m_model;
        torch::optim::Optimizer & m_opt;
        torch::nn::ModuleHolder<torch::nn::CrossEntropyLossImpl> m_lossFunction;
        torch::Device m_device{"cpu"};
};

#endif

#endif
