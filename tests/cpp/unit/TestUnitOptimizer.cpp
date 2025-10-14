#include "Optimizer.h"

#include <gtest/gtest.h>
#include <vector>

using namespace std;

class TestTypeParam : public ::testing::TestWithParam<OptimizerType> {};

TEST_P(TestTypeParam, CreateOptimizer)
{
    // Get testing params.
    OptimizerType type = GetParam();

    // Define the test object.
    OptimizerConfig cfg(1e-3);
    unique_ptr<torch::optim::Optimizer> opt;

    // Compute the result.
    if (type == static_cast<OptimizerType>(-1))
    {
        EXPECT_THROW(CreateOptimizer(vector<torch::Tensor>{}, type, cfg, opt), invalid_argument); // CreateOptimizer: Invalid optimizer type.
        return;
    }
    CreateOptimizer(vector<torch::Tensor>{}, type, cfg, opt);

    // Test the result.
    ASSERT_NE(opt.get(), nullptr) << "Optimizer not created.";
}

INSTANTIATE_TEST_SUITE_P(
    TestOptimizerWithParams,
    TestTypeParam,
    ::testing::Values(
        OptimizerType::Adam,
        OptimizerType::AdamW,
        OptimizerType::SGD,
        OptimizerType::RMSprop,
        static_cast<OptimizerType>(-1)
    ),
    [](const testing::TestParamInfo<OptimizerType> & info)
    {
        switch (info.param)
        {
            case OptimizerType::Adam:
                return "Adam";
            case OptimizerType::AdamW:
                return "AdamW";
            case OptimizerType::SGD:
                return "SGD";
            case OptimizerType::RMSprop:
                return "RMSprop";
            default:
                return "Invalid";
        }
    }
);
