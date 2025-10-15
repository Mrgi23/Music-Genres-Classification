#include "Scheduler.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <torch/optim/adam.h>
#include <vector>

using namespace std;

class TestModeParam : public ::testing::TestWithParam<string> {};

TEST_P(TestModeParam, UpdateLR)
{
    // Get testing params.
    string mode = GetParam();

    // Define the test object.
    torch::optim::Adam opt(vector<torch::Tensor>{}, torch::optim::AdamOptions(1e-3));
    if (mode == "invalid")
    {
        EXPECT_THROW(ReduceLROnPlateau scheduler(mode, 0.5, 0), invalid_argument); // ReduceLROnPlateau::ReduceLROnPlateau: Mode must be 'min' or 'max'
        return;
    }
    ReduceLROnPlateau scheduler(mode, 0.5, 0);

    // Define the expected result.
    float lrExpected = 5e-4;

    // Compute the result.
    scheduler.AttachOptimizer(&opt);
    if (mode == "max")
    {
        scheduler.UpdateLR(0.5f);
        scheduler.UpdateLR(0.4f);
    }
    else if (mode == "min")
    {
        scheduler.UpdateLR(0.4f);
        scheduler.UpdateLR(0.5f);
    }

    // Test the result.
    float lr = static_cast<float>(opt.param_groups()[0].options().get_lr());
    ASSERT_NEAR(lr, lrExpected, 1e-6) << "Invalid learning rate value.";
}

INSTANTIATE_TEST_SUITE_P(
    TestSchedulerWithParams,
    TestModeParam,
    ::testing::Values("max", "min", "invalid"),
    [](const testing::TestParamInfo<string> & info)
    {
        return info.param;
    }
);

TEST(TestScheduler, AttachOptimizer)
{
    ReduceLROnPlateau scheduler("max", 0.5, 0); // ReduceLROnPlateau::UpdateLR: Optimizer is not attached.
    EXPECT_THROW(scheduler.UpdateLR(0.5), runtime_error);
}