#include "Model.h"

#include <gtest/gtest.h>
#include <memory>
#include <torch/autograd.h>

using namespace std;

// Define the test object.
class TestModel : public ::testing::Test
{
    private:
        shared_ptr<MusicModelImpl> impl;
    protected:
        MusicModel * model;

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
};

TEST_F(TestModel, forward)
{
    // Define the input and the expected result.
    long numFrames = 1290;
    long mfccSize = 13;
    torch::Tensor x = torch::rand({1, numFrames, mfccSize});
    long numClassesExpected = 10;

    // Compute the result.
    torch::Tensor y = model->forward(x);

    // Test the result.
    ASSERT_EQ(y.numel(), numClassesExpected) << "Invalid size of the model output.";
}

TEST_F(TestModel, train)
{
    // Compute the result.
    model->train(true);

    // Test the result.
    ASSERT_TRUE(model->get()->is_training()) << "Invalid train mode.";

    // Compute the result.
    model->train(false);

    // Test the result.
    ASSERT_FALSE(model->get()->is_training()) << "Invalid train mode.";
}

TEST_F(TestModel, eval)
{
    // Compute the result.
    model->train(true);
    model->eval();

    // Test the result.
    ASSERT_FALSE(model->get()->is_training()) << "Invalid evaluation mode.";
}

TEST_F(TestModel, Save)
{
    // Create temporary file.
    fs::path filePath = fs::temp_directory_path() / "file.pt";

    // Compute the result.
    model->Save(filePath);

    // Test the result.
    EXPECT_TRUE(fs::exists(filePath)) << "Model is not saved.";

    // Clean up.
    fs::remove(filePath);
}

class TestExistParam : public ::testing::TestWithParam<bool> {};

TEST_P(TestExistParam, Load)
{
    // Get testing params.
    bool exists = GetParam();

    // Create temporary file.
    fs::path filePath = fs::temp_directory_path() / "file.pt";

    // Define the test object.
    MusicModel modelSave;
    MusicModel modelLoad;

    // Test the result.
    if (exists)
    {
        modelSave.Save(filePath);
        modelLoad.Load(filePath);
        for (const auto& item : modelSave.get()->named_parameters())
        {
            const auto& name = item.key();
            const auto& tensorSave = item.value();
            const auto& tensorLoad = modelLoad.get()->named_parameters()[name];
            ASSERT_TRUE(torch::equal(tensorSave, tensorLoad)) << "Model is not loaded properly.";
        }
    }
    else
    {
        EXPECT_THROW(modelLoad.Load(filePath), invalid_argument); // MusicModel::Load: File: ... does not exist."
    }
    fs::remove(filePath);
}

INSTANTIATE_TEST_SUITE_P(
    TestModelWithParams,
    TestExistParam,
    ::testing::Values(true, false),
    [](const testing::TestParamInfo<bool> & info)
    {
        return ::testing::PrintToString(info.param);
    }
);
