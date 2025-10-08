#include "Trainer.h"

Trainer::Trainer(MusicModel & model, const OptimizerType & type, const OptimizerConfig & cfg)
    : m_model(model), m_lossFunction(torch::nn::CrossEntropyLoss())
{
    CreateOptimizer(model->parameters(), type, cfg, m_opt);
    m_model->to(DeviceManager::Get());
}

Trainer::~Trainer() = default;

void Trainer::AttachScheduler(ReduceLROnPlateau * scheduler)
{
    scheduler->AttachOptimizer(m_opt.get());
}

void Trainer::TrainModel(AudioDataloader<RandomSampler> & dataloader, float & loss, float & acc)
{
    m_model->train();

    float totalLoss = 0.0f;
    size_t numBatches = 0U;
    float correctPred = 0.0f;
    size_t totalSamples = 0U;
    for (auto &batch : dataloader)
    {
        auto data = batch.data.to(DeviceManager::Get());
        auto target = batch.target.to(DeviceManager::Get());

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

void Trainer::EvalModel(AudioDataloader<SequentialSampler> & dataloader, float & loss, float & acc)
{
    m_model->eval();
    torch::NoGradGuard noGrad;

    float totalLoss = 0.0f;
    size_t numBatches = 0U;
    float correctPred = 0.0f;
    size_t totalSamples = 0U;
    for (auto &batch : dataloader)
    {
        auto data = batch.data.to(DeviceManager::Get());
        auto target = batch.target.to(DeviceManager::Get());

        torch::Tensor output = m_model->forward(data);
        auto batchLoss = m_lossFunction(output, target);
        totalLoss += batchLoss.item<float>();

        // Calculate accuracy.
        auto preds = output.argmax(1);
        correctPred += preds.eq(target).sum().item<float>();
        totalSamples += target.size(0);

        numBatches++;
    }

    loss = totalLoss / static_cast<float>(numBatches);
    acc = correctPred / static_cast<float>(totalSamples);
}
