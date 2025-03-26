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

class AudioDataset : public torch::data::Dataset<AudioDataset> {
    private:
        Preprocessor * preprocessor;
        std::vector<torch::Tensor> data;
        std::vector<torch::Tensor> target;
        c10::Dict<std::string, torch::Tensor> classes;
    public:
        AudioDataset(const fs::path& rootPath, Preprocessor * preprocessor);
        ~AudioDataset() {}

        inline Preprocessor * getPreprocessor() const { return preprocessor; }
        inline c10::Dict<std::string, torch::Tensor> const getClasses() { return classes; }

        inline torch::optional<size_t> size() const override { return data.size(); }
        torch::data::Example<> get(size_t index) override;
};

#endif

#endif
