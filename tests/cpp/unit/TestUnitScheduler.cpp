#include "Scheduler.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <torch/optim/adam.h>
#include <vector>

class TestModeParam : public ::testing::TestWithParam<std::string>
{
};

TEST_P(TestModeParam, update)
{
	std::string mode = GetParam();
	torch::optim::Adam opt(std::vector<torch::Tensor>{}, torch::optim::AdamOptions(1e-3));
	if (mode == "invalid")
	{
		EXPECT_THROW(ReduceLROnPlateau scheduler(mode, 0.5, 0), std::invalid_argument);
		return;
	}
	ReduceLROnPlateau scheduler(mode, 0.5, 0);

	float lrExpected = 5e-4;
	scheduler.attach(&opt);
	if (mode == "max")
	{
		scheduler.update(0.5f);
		scheduler.update(0.4f);
	}
	else if (mode == "min")
	{
		scheduler.update(0.4f);
		scheduler.update(0.5f);
	}

	float lr = static_cast<float>(opt.param_groups()[0].options().get_lr());
	ASSERT_NEAR(lr, lrExpected, 1e-6) << "Invalid learning rate value.";
}

INSTANTIATE_TEST_SUITE_P(
	TestSchedulerWithParams,
	TestModeParam,
	::testing::Values("max", "min", "invalid"),
	[](const testing::TestParamInfo<std::string>& info)
	{
		return info.param;
	}
);

TEST(TestScheduler, updateThrowError)
{
	ReduceLROnPlateau scheduler("max", 0.5, 0);
	EXPECT_THROW(scheduler.update(0.5), std::runtime_error);
}
