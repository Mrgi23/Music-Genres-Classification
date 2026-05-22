#include "Trainer.h"

Trainer::Trainer(MusicModel& model, const OptimizerType& type, const OptimizerConfig& cfg)
  : m_model(model), m_lossFunction(torch::nn::CrossEntropyLoss())
{
  createOptimizer(model->parameters(), type, cfg, m_opt);
  m_model->to(DeviceManager::get());
}

Trainer::~Trainer() = default;

void Trainer::attach(ReduceLROnPlateau* scheduler)
{
  scheduler->attach(m_opt.get());
}

void Trainer::train(AudioDataloader<RandomSampler>& dataloader, float& loss, float& acc)
{
  m_model->train();

  float totalLoss = 0.0f;
  size_t numBatches = 0U;
  float correctPred = 0.0f;
  size_t totalSamples = 0U;
  for (auto& batch : dataloader)
  {
    auto data = batch.data.to(DeviceManager::get());
    auto target = batch.target.to(DeviceManager::get());

    m_opt->zero_grad();

    torch::Tensor output = m_model->forward(data);
    auto batchLoss = m_lossFunction(output, target);
    totalLoss += batchLoss.item<float>();

    auto preds = output.argmax(1);
    correctPred += preds.eq(target).sum().item<float>();
    totalSamples += target.size(0);

    batchLoss.backward();

    m_opt->step();

    numBatches++;
  }

  loss = totalLoss / static_cast<float>(numBatches);
  acc = correctPred / static_cast<float>(totalSamples);
}

void Trainer::eval(AudioDataloader<SequentialSampler>& dataloader, float& loss, float& acc)
{
  m_model->eval();
  torch::NoGradGuard noGrad;

  float totalLoss = 0.0f;
  size_t numBatches = 0U;
  float correctPred = 0.0f;
  size_t totalSamples = 0U;
  for (auto& batch : dataloader)
  {
    auto data = batch.data.to(DeviceManager::get());
    auto target = batch.target.to(DeviceManager::get());

    torch::Tensor output = m_model->forward(data);
    auto batchLoss = m_lossFunction(output, target);
    totalLoss += batchLoss.item<float>();

    auto preds = output.argmax(1);
    correctPred += preds.eq(target).sum().item<float>();
    totalSamples += target.size(0);

    numBatches++;
  }

  loss = totalLoss / static_cast<float>(numBatches);
  acc = correctPred / static_cast<float>(totalSamples);
}
