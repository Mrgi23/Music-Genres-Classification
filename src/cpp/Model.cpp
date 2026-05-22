#include "Model.h"
#include <stdexcept>
#include <torch/nn/functional/activation.h>
#include <torch/serialize.h>

MusicModelImpl::MusicModelImpl()
{
  conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 512, 3).stride(1).padding(1).bias(false));
  register_module("conv1", conv1);
  bn1 = torch::nn::BatchNorm2d(512);
  register_module("bn1", bn1);

  conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 256, 3).stride(1).padding(1).bias(false));
  register_module("conv2", conv2);
  bn2 = torch::nn::BatchNorm2d(256);
  register_module("bn2", bn2);

  conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 3).stride(1).padding(1).bias(false));
  register_module("conv3", conv3);
  bn3 = torch::nn::BatchNorm2d(128);
  register_module("bn3", bn3);

  maxPool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
  register_module("maxPool", maxPool);
  adaptivePool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
  register_module("adaptivePool", adaptivePool);

  linear = torch::nn::Linear(128, 64);
  register_module("linear", linear);
  dropout = torch::nn::Dropout(0.3);
  register_module("dropout", dropout);
  output = torch::nn::Linear(64, 10);
  register_module("output", output);
}

MusicModelImpl::~MusicModelImpl() = default;

torch::Tensor MusicModelImpl::forward(torch::Tensor x)
{
  if (x.dim() == 3)
    x = x.unsqueeze(1);

  x = maxPool->forward(torch::relu(bn1->forward(conv1->forward(x))));
  x = maxPool->forward(torch::relu(bn2->forward(conv2->forward(x))));
  x = maxPool->forward(torch::relu(bn3->forward(conv3->forward(x))));

  x = adaptivePool->forward(x);
  x = x.view({x.size(0), -1});

  x = torch::relu(linear->forward(x));
  x = dropout->forward(x);
  x = output->forward(x);
  return x;
}

MusicModel::MusicModel(std::shared_ptr<MusicModelImpl> impl)
{
  this->impl_ = impl;
}

torch::Tensor MusicModel::forward(torch::Tensor x)
{
  return this->get()->forward(x);
}

void MusicModel::to(const torch::Device& device)
{
  this->get()->to(device);
}

void MusicModel::train(bool on)
{
  this->get()->train(on);
}

void MusicModel::eval()
{
  this->get()->eval();
}

void MusicModel::save(const fs::path& filePath)
{
  auto device = this->get()->parameters()[0].device();
  this->to(torch::Device{"cpu"});
  torch::save(*this, filePath.string());
  this->to(device);
}

void MusicModel::load(const fs::path& filePath)
{
  if (fs::exists(filePath))
  {
    auto device = this->get()->parameters()[0].device();
    torch::load(*this, filePath.string());
    this->to(device);
  }
  else
    throw std::invalid_argument("MusicModel::load: File: " + filePath.string() + " does not exist.");
}