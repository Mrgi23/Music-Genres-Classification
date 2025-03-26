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
        std::vector<fs::path> files;
        std::vector<std::string> labels;
        std::map<std::string, uint> classes;
    public:
        AudioDataset(const fs::path& rootPath, Preprocessor * preprocessor);
        ~AudioDataset() { preprocessor = nullptr; }

        inline torch::optional<size_t> size() const override { return files.size(); }
        torch::data::Example<> get(size_t index) override;
        inline std::map<std::string, uint> const getClasses() { return classes; }
};

#endif

#endif
