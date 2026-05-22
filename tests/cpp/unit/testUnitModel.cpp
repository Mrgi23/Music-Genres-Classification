#include "model.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <torch/autograd.h>

class TestModel : public ::testing::Test
{
  protected:
    MusicModel* model;

    void SetUp() override
    {
      impl = std::make_shared<MusicModelImpl>();
      model = new MusicModel(impl);
    }

    void TearDown() override
    {
      delete model;
      model = nullptr;
    }

  private:
    std::shared_ptr<MusicModelImpl> impl;
};

TEST_F(TestModel, forward)
{
  long numFrames = 1290;
  long mfccSize = 13;
  torch::Tensor x = torch::rand({1, numFrames, mfccSize});
  long numClassesExpected = 10;

  torch::Tensor y = model->forward(x);
  ASSERT_EQ(y.numel(), numClassesExpected) << "Invalid size of the model output.";
}

TEST_F(TestModel, train)
{
  model->train(true);
  ASSERT_TRUE(model->get()->is_training()) << "Invalid train mode.";

  model->train(false);
  ASSERT_FALSE(model->get()->is_training()) << "Invalid train mode.";
}

TEST_F(TestModel, eval)
{
  model->train(true);
  model->eval();
  ASSERT_FALSE(model->get()->is_training()) << "Invalid evaluation mode.";
}

TEST_F(TestModel, save)
{
  fs::path filePath = fs::temp_directory_path() / "file.pt";

  model->save(filePath);
  EXPECT_TRUE(fs::exists(filePath)) << "Model is not saved.";

  fs::remove(filePath);
}

class TestExistParam : public ::testing::TestWithParam<bool>
{
};

TEST_P(TestExistParam, load)
{
  bool exists = GetParam();
  fs::path filePath = fs::temp_directory_path() / "file.pt";
  MusicModel modelSave;
  MusicModel modelLoad;

  if (exists)
  {
    modelSave.save(filePath);
    modelLoad.load(filePath);
    for (const auto &item : modelSave.get()->named_parameters())
    {
      const auto& name = item.key();
      const auto& tensorSave = item.value();
      const auto& tensorLoad = modelLoad.get()->named_parameters()[name];
      ASSERT_TRUE(torch::equal(tensorSave, tensorLoad)) << "Model is not loaded properly.";
    }
  }
  else
    EXPECT_THROW(modelLoad.load(filePath), std::invalid_argument);
  fs::remove(filePath);
}

INSTANTIATE_TEST_SUITE_P(
  TestModelWithParams,
  TestExistParam,
  ::testing::Values(true, false),
  [](const testing::TestParamInfo<bool> &info)
  {
    return ::testing::PrintToString(info.param);
  }
);
