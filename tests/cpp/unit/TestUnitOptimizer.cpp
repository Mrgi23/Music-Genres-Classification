#include "Optimizer.h"
#include <gtest/gtest.h>
#include <vector>
#include <memory>

class TestTypeParam : public ::testing::TestWithParam<OptimizerType>
{
};

TEST_P(TestTypeParam, createOptimizer)
{
  OptimizerConfig cfg(1e-3);
  std::unique_ptr<torch::optim::Optimizer> opt;
  OptimizerType type = GetParam();
  if (type == static_cast<OptimizerType>(-1))
  {
    EXPECT_THROW(createOptimizer(std::vector<torch::Tensor>{}, type, cfg, opt), std::invalid_argument);
    return;
  }

  createOptimizer(std::vector<torch::Tensor>{}, type, cfg, opt);
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
      static_cast<OptimizerType>(-1)),
  [](const testing::TestParamInfo<OptimizerType>& info)
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
