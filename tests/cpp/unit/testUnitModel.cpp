#include <gtest/gtest.h>
#include "model.h"

using namespace std;

TEST(TestModel, testModel)
{
    // Define the test object.
    MusicModel model;

    // Define the input and the expected result.
    long batchSize = 16;
    long numFrames = 1290;
    long mfccSize = 13;
    torch::Tensor x = torch::rand({batchSize, numFrames, mfccSize}).unsqueeze(1);
    long numClassesExpected = 10;

    // Compute the result.
    model->eval();
    torch::NoGradGuard noGrad;
    torch::Tensor y = model->forward(x);

    // Test the result.
    EXPECT_EQ(y.numel(), batchSize * numClassesExpected) << "Invalid size of the model output.";
}
