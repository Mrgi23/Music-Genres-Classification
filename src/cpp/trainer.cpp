#include "trainer.h"

void Trainer::fit(AudioDataloader<RandomSampler>& dataloader, float& loss, float& acc) {
    // Set model into the training mode.
    model->train();

    // Iterate through the data loader and train the model.
    float totalLoss = 0.0f;
    size_t numBatches = 0U;
    float correctPred = 0.0f;
    size_t totalSamples = 0U;
    for (auto& batch : dataloader) {
        // Send data to device.
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        // Zero the gradients.
        opt.zero_grad();

        // Compute forward pass and calculate the loss.
        torch::Tensor output = model->forward(data);
        auto loss = lossFunction(output, target);
        totalLoss += loss.item<float>();

        // Calculate accuracy.
        auto preds = output.argmax(1);
        correctPred += preds.eq(target).sum().item<float>();
        totalSamples += target.size(0);

        // Compute gradients via backward pass.
        loss.backward();

        // Update the model parameters.
        opt.step();

        // Increment the total number of batches.
        numBatches++;
    }

    // Calculate average loss over number of batches and accuracy.
    loss = totalLoss / static_cast<float>(numBatches);
    acc = correctPred / static_cast<float>(totalSamples);
}

void Trainer::eval(AudioDataloader<SequentialSampler> &dataloader, float& loss, float& acc) {
    // Set model into the evaluation mode (i.e., do not calculate gradients).
    model->eval();
    torch::NoGradGuard noGrad;

    // Iterate through the data loader and evaluate the model.
    float totalLoss = 0.0f;
    size_t numBatches = 0U;
    float correctPred = 0.0f;
    size_t totalSamples = 0U;
    for (auto& batch : dataloader) {
        // Send data to device.
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);

        // Compute forward pass and calculate the loss.
        torch::Tensor output = model->forward(data);
        auto loss = lossFunction(output, target);
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
