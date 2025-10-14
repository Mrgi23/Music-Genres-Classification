#pragma once

#ifdef __cplusplus

#include <filesystem>
#include <torch/nn/module.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/pooling.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

namespace fs = std::filesystem;

/**
 * @class MusicModelImpl
 * @brief Convolutional neural network for music genre classification.
 *
 * This class implements the actual network architecture using
 * convolutional, batch normalization, pooling, and linear layers.
 * It inherits from torch::nn::Module and defines the forward pass.
 */
class MusicModelImpl : public torch::nn::Module
{
    public:
        /**
         * @brief Construct a new MusicModelImpl object.
         *
         * Initializes all layers of the model.
         */
    	MusicModelImpl();
        /**
         * @brief Destructor.
         *
         * Default cleanup.
         */
      	~MusicModelImpl();
        /**
         * @brief Forward pass through the network.
         *
         * If the input is a batch, it should have shape (N, C, H, W).
         * If a single sample is provided with shape (C, H, W),
         * the function will automatically add a batch dimension (unsqueeze at dim=0).
         *
         * @param[in] x Input tensor of shape (N, C, H, W) or (C, H, W).
         * @return torch::Tensor Output tensor containing class scores.
         */
      	virtual torch::Tensor forward(torch::Tensor x);
    private:
        torch::nn::Conv2d conv1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};

        torch::nn::Conv2d conv2{nullptr};
        torch::nn::BatchNorm2d bn2{nullptr};

        torch::nn::Conv2d conv3{nullptr};
        torch::nn::BatchNorm2d bn3{nullptr};

        torch::nn::MaxPool2d maxPool{nullptr};
        torch::nn::AdaptiveAvgPool2d adaptivePool{nullptr};

        torch::nn::Linear linear{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear output{nullptr};
};

/**
 * @brief Wrapper around MusicModelImpl.
 *
 * Inherits from `torch::nn::ModuleHolder<MusicModelImpl>` and manages
 * the underlying model instance. Exposes convenience methods to
 * serialize the model to disk and restore it later.
 */
struct MusicModel : torch::nn::ModuleHolder<MusicModelImpl>
{
    using torch::nn::ModuleHolder<MusicModelImpl>::ModuleHolder;

    /**
     * @brief Construct a new MusicModel object.
     *
     * @param[in] impl Shared pointer to a MusicModelImpl instance.
     */
    explicit MusicModel(std::shared_ptr<MusicModelImpl> impl);

    /**
     * @brief Forward pass through the network.
     *
     * If the input is a batch, it should have shape (N, C, H, W).
     * If a single sample is provided with shape (C, H, W),
     * the function will automatically add a batch dimension (unsqueeze at dim=0).
     *
     * @param[in] x Input tensor of shape (N, C, H, W) or (C, H, W).
     * @return torch::Tensor Output tensor containing class scores.
     */
    torch::Tensor forward(torch::Tensor x);
    /**
     * @brief Move the model to the specified device.
     *
     * Transfers all parameters and buffers of the model to the
     * given device (e.g., CPU or GPU).
     *
     * @param[in] device Target device to move the model to.
     */
    void to(const torch::Device & device);
    /**
     * @brief Set the model in training mode.
     *
     * @param[in] on If true, enables training mode (default). If false,
     *               switches to evaluation mode.
     */
    void train(bool on = true);
    /**
     * @brief Set the model in evaluation mode.
     */
    void eval();

    /**
     * @brief Save the model to the specified file.
     *
     * @param[in] filePath Destination file path for the serialized model.
     */
    void Save(const fs::path & filePath);

    /**
     * @brief Load a MusicModel from a file.
     *
     * @param[in] filePath Source file path of the serialized model.
     *
     * @throws std::invalid_argument If model file does not exist.
     */
    void Load(const fs::path & filePath);
};

#endif
