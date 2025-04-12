#ifndef MODEL_H
#define MODEL_H

#ifdef __cplusplus

#include <torch/torch.h>

class MusicModelImpl : public torch::nn::Module {
    private:
        // First layer.
        torch::nn::Conv2d conv1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};

        // Second layer.
        torch::nn::Conv2d conv2{nullptr};
        torch::nn::BatchNorm2d bn2{nullptr};

        // Third layer.
        torch::nn::Conv2d conv3{nullptr};
        torch::nn::BatchNorm2d bn3{nullptr};

        // Pool layers.
        torch::nn::MaxPool2d maxPool{nullptr};
        torch::nn::AdaptiveAvgPool2d adaptivePool{nullptr};

        // FC layers.
        torch::nn::Linear linear{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear output{nullptr};
    public:
        MusicModelImpl();
        ~MusicModelImpl() {}
        virtual torch::Tensor forward(torch::Tensor x);
};

struct MusicModel : torch::nn::ModuleHolder<MusicModelImpl>
{
  using torch::nn::ModuleHolder<MusicModelImpl>::ModuleHolder;

  explicit MusicModel(std::shared_ptr<MusicModelImpl> impl)
  {
    this->impl_ = std::move(impl);
  }
};

#endif

#endif
