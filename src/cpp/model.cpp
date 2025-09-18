#include "model.h"

MusicModelImpl::MusicModelImpl()
{
    // First layer.
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 512, 3).stride(1).padding(1).bias(false));
    register_module("conv1", conv1);
    bn1 = torch::nn::BatchNorm2d(512);
    register_module("bn1", bn1);

    // Second layer.
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 3).stride(1).padding(1).bias(false));
    register_module("conv2", conv2);
    bn2 = torch::nn::BatchNorm2d(256);
    register_module("bn2", bn2);

    // Third layer.
    conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 3).stride(1).padding(1).bias(false));
    register_module("conv3", conv3);
    bn3 = torch::nn::BatchNorm2d(128);
    register_module("bn3", bn3);

    // Pool layers.
    maxPool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
    register_module("maxPool", maxPool);
    adaptivePool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
    register_module("adaptivePool", adaptivePool);

    // FC layers.
    linear = torch::nn::Linear(128, 64);
    register_module("linear", linear);
    dropout = torch::nn::Dropout(0.3);
    register_module("dropout", dropout);
    output = torch::nn::Linear(64, 10);
    register_module("output", output);
}

MusicModelImpl::~MusicModelImpl()
{

}

torch::Tensor MusicModelImpl::forward(torch::Tensor x)
{
    // Apply first layer.
    x = maxPool->forward(torch::relu(bn1->forward(conv1->forward(x))));

    // Apply second layer.
    x = maxPool->forward(torch::relu(bn2->forward(conv2->forward(x))));

    // Apply third layer.
    x = maxPool->forward(torch::relu(bn3->forward(conv3->forward(x))));

    // Flatten the output.
    x = adaptivePool->forward(x);
    x = x.view({x.size(0), -1});

    // Apply FC layers.
    x = torch::relu(linear->forward(x));
    x = dropout->forward(x);
    x = output->forward(x);
    return x;
}

MusicModel::MusicModel(std::shared_ptr<MusicModelImpl> impl)
{
    // Initialize the model from external implementation.
    this->impl_ = impl;
}