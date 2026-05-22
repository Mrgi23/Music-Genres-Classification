#include "Dataset.h"
#include <gtest/gtest.h>

class TestDataset : public ::testing::Test
{
	protected:
		AudioDataset* audioDataset = nullptr;
		Preprocessor* preprocessor = nullptr;

		void SetUp() override
		{
			fs::path datasetPath = "../../../resources";
			preprocessor = new Preprocessor();
			audioDataset = new AudioDataset(datasetPath, preprocessor);
		}

		void TearDown() override
		{
			delete preprocessor;
			preprocessor = nullptr;
			delete audioDataset;
			audioDataset = nullptr;
		}
};

TEST_F(TestDataset, GetClasses)
{
	c10::Dict<int64_t, std::string> classesExpected;
	classesExpected.insert(0, "blues");
	classesExpected.insert(1, "classical");
	classesExpected.insert(2, "country");
	classesExpected.insert(3, "disco");
	classesExpected.insert(4, "hiphop");
	classesExpected.insert(5, "jazz");
	classesExpected.insert(6, "metal");
	classesExpected.insert(7, "pop");
	classesExpected.insert(8, "reggae");
	classesExpected.insert(9, "rock");

	c10::Dict<int64_t, std::string> classes = audioDataset->classes();
	ASSERT_EQ(classes.size(), classesExpected.size()) << "Invalid number of classes.";
	for (const auto& pair : classes)
	{
		ASSERT_TRUE(classesExpected.contains(pair.key())) << "Invalid class.";
		ASSERT_EQ(pair.value(), classesExpected.at(pair.key())) << "Invalid class.";
	}
}

TEST_F(TestDataset, get)
{
	uint size = audioDataset->preprocessor()->config().size;
	uint hop = audioDataset->preprocessor()->config().hop;
	uint nmfcc = audioDataset->preprocessor()->config().nmfcc;

	size_t numFramesExpected = size / hop + 1;
	size_t mfccSizeExpected = nmfcc;
	torch::data::Example<> sample = audioDataset->get(0);
	ASSERT_EQ(sample.data.numel(), numFramesExpected * mfccSizeExpected) << "Invalid size of the sample data.";
	ASSERT_EQ(sample.target.numel(), 1) << "Invalid size of the sample target";
}

TEST_F(TestDataset, size)
{
	size_t sizeExpected = 999;
	torch::optional<size_t> size = audioDataset->size();
	ASSERT_EQ(size.value(), sizeExpected) << "Invalid size of the dataset.";
}
