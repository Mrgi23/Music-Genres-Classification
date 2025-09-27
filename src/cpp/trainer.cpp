#include "trainer.h"

Trainer::Trainer(MusicModel & model, torch::optim::Optimizer & opt)
: m_model(model), m_opt(opt), m_lossFunction(torch::nn::CrossEntropyLoss())
{
    m_model->to(DeviceManager::get());
}

Trainer::~Trainer()
{

}

void Trainer::fit(AudioDataloader<RandomSampler> & dataloader, float & loss, float & acc)
{
    // Set model into the training mode.
    m_model->train();

    // Iterate through the data loader and train the model.
    float totalLoss = 0.0f;
    size_t numBatches = 0U;
    float correctPred = 0.0f;
    size_t totalSamples = 0U;
    for (auto &batch : dataloader)
    {
        // Send data to device.
        auto data = batch.data.to(DeviceManager::get());
        auto target = batch.target.to(DeviceManager::get());

        // Zero the gradients.
        m_opt.zero_grad();

        // Compute forward pass and calculate the loss.
        torch::Tensor output = m_model->forward(data);
        auto loss = m_lossFunction(output, target);
        totalLoss += loss.item<float>();

        // Calculate accuracy.
        auto preds = output.argmax(1);
        correctPred += preds.eq(target).sum().item<float>();
        totalSamples += target.size(0);

        // Compute gradients via backward pass.
        loss.backward();

        // Update the model parameters.
        m_opt.step();

        // Increment the total number of batches.
        numBatches++;
    }

    // Calculate average loss over number of batches and accuracy.
    loss = totalLoss / static_cast<float>(numBatches);
    acc = correctPred / static_cast<float>(totalSamples);
}

void Trainer::eval(AudioDataloader<SequentialSampler> & dataloader, float & loss, float & acc)
{
    // Set model into the evaluation mode (i.e., do not calculate gradients).
    m_model->eval();
    torch::NoGradGuard noGrad;

    // Iterate through the data loader and evaluate the model.
    float totalLoss = 0.0f;
    size_t numBatches = 0U;
    float correctPred = 0.0f;
    size_t totalSamples = 0U;
    for (auto &batch : dataloader)
    {
        // Send data to device.
        auto data = batch.data.to(DeviceManager::get());
        auto target = batch.target.to(DeviceManager::get());

        // Compute forward pass and calculate the loss.
        torch::Tensor output = m_model->forward(data);
        auto loss = m_lossFunction(output, target);
        totalLoss += loss.item<float>();

        // Calculate accuracy.
        auto preds = output.argmax(1);
        correctPred += preds.eq(target).sum().item<float>();
        totalSamples += target.size(0);

        // Increment the total number of batches.
        numBatches++;
    }

    // Calculate average loss over number of batches and accuracy.
    loss = totalLoss / static_cast<float>(numBatches);
    acc = correctPred / static_cast<float>(totalSamples);
}
