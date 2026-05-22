#pragma once

#include "preprocessor.hpp"
#include <ATen/core/Dict.h>
#include <string>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/serialize/archive.h>

namespace fs = std::filesystem;

/**
 * @class AudioDataset
 * @brief Dataset class for loading and preprocessing audio data.
 *
 * This dataset loads audio files from a directory, applies preprocessing
 * (via Preprocessor), and provides data/target pairs as torch::data::Example<>.
 * Data can be preloaded from a serialized .pt archive to save time.
 */
class AudioDataset : public torch::data::datasets::Dataset<AudioDataset>
{
  public:
    /**
     * @brief Construct a new AudioDataset object.
     *
     * If a serialized dataset exists, it is loaded from disk.
     * Otherwise, the dataset is built by preprocessing audio files.
     *
     * @param[in] rootPath Root directory containing genre subfolders.
     * @param[in] preprocessor Pointer to a Preprocessor for feature extraction.
     */
    AudioDataset(const fs::path& rootPath, Preprocessor* preprocessor);

    /**
     * @brief Destructor.
     *
     * Default cleanup.
     */
    ~AudioDataset();

    /**
     * @brief Get the mapping from tensor indices to class names.
     * @return c10::Dict<int64_t, std::string> Class dictionary.
     */
    c10::Dict<int64_t, std::string> classes() const;

    /**
     * @brief Get the associated Preprocessor instance.
     * @return Preprocessor* Pointer to preprocessor.
     */
    Preprocessor* preprocessor() const;

    /**
     * @brief Get a single data/target example by index.
     * @param[in] index Sample index.
     * @return torch::data::Example<> Pair of (data, target).
     */
    virtual torch::data::Example<> get(size_t index) override;

    /**
     * @brief Get the dataset size.
     * @return torch::optional<size_t> Number of samples.
     */
    torch::optional<size_t> size() const override;

  private:
    Preprocessor* m_preprocessor;
    std::vector<torch::Tensor> m_data;
    std::vector<torch::Tensor> m_target;
    c10::Dict<int64_t, std::string> m_classes;
};

/**
 * @class AudioSubset
 * @brief Subset wrapper around an AudioDataset with selected indices.
 *
 * Provides a view into a parent dataset for training/validation splits.
 */
class AudioSubset : public torch::data::datasets::Dataset<AudioSubset>
{
  public:
    /**
     * @brief Construct a new AudioSubset object.
     *
     * @param[in] dataset Pointer to the parent dataset.
     * @param[in] indices Indices of samples to include in the subset.
     */
    AudioSubset(AudioDataset* dataset, std::vector<size_t> indices);

    /**
     * @brief Destructor.
     *
     * Default cleanup.
     */
    ~AudioSubset();

    /**
     * @brief Get the parent dataset pointer.
     * @return AudioDataset* Parent dataset.
     */
    AudioDataset* dataset() const;

    /**
     * @brief Get all subset data stacked into a single tensor.
     * @return torch::Tensor Concatenated data tensor.
     */
    torch::Tensor stackedData() const;

    /**
     * @brief Get a single example by subset index.
     * @param[in] index Subset-relative index.
     * @return torch::data::Example<> Pair of (data, target).
     */
    torch::data::Example<> get(size_t index) override;
    
    /**
     * @brief Get the subset size.
     * @return torch::optional<size_t> Number of samples in subset.
     */
    torch::optional<size_t> size() const override;

  private:
    AudioDataset* m_dataset;
    std::vector<size_t> m_indices;
    mutable torch::Tensor m_stackedData;
};
