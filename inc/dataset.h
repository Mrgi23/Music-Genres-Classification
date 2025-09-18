#ifndef DATASET_H
#define DATASET_H

#ifdef __cplusplus

#include <filesystem>
#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>
#include "preprocessor.h"

namespace fs = std::filesystem;

class AudioDataset : public torch::data::Dataset<AudioDataset>
{
    public:
        AudioDataset(const fs::path & rootPath, Preprocessor * preprocessor);
        ~AudioDataset();

        torch::optional<size_t> size() const override;
        virtual torch::data::Example<> get(size_t index) override;

        Preprocessor * preprocessor() const;
        c10::Dict<std::string, torch::Tensor> classes() const;
    private:
        Preprocessor * m_preprocessor;
        std::vector<torch::Tensor> m_data;
        std::vector<torch::Tensor> m_target;
        c10::Dict<std::string, torch::Tensor> m_classes;
};

class AudioSubset : public torch::data::Dataset<AudioSubset>
{
    public:
        AudioSubset(AudioDataset * dataset, std::vector<size_t> indices);
        ~AudioSubset();

        torch::data::Example<> get(size_t index) override;
        torch::optional<size_t> size() const override;

        torch::Tensor data() const;
    private:
        AudioDataset * m_dataset;
        std::vector<size_t> m_indices;
};

#endif

#endif
