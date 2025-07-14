#include <algorithm>
#include "dataset.h"

using namespace std;

AudioDataset::AudioDataset(const fs::path & rootPath, Preprocessor * preprocessor) : m_preprocessor(preprocessor)
{

    fs::path datasetPath = rootPath / "dataset_cpp.pt";

    if (fs::exists(datasetPath))
    {
        // Preload existing dataset.
        torch::IValue dataIValue, targetIValue, classesIValue;
        torch::serialize::InputArchive dataset;
        dataset.load_from(datasetPath);
        dataset.read("data", dataIValue);
        dataset.read("target", targetIValue);
        dataset.read("classes", classesIValue);

        m_data = dataIValue.toTensorVector();
        m_target = targetIValue.toTensorVector();
        for (const auto &item : classesIValue.toGenericDict())
        {
            string key = item.key().toStringRef();
            torch::Tensor value = item.value().toTensor();
            m_classes.insert(key, value);
        }
    }
    else
    {
        // Collect the directories (genres) and sort the in alphabetical order.
        vector<fs::path> directories;
        for (const auto &entry : fs::directory_iterator(rootPath))
        {
            if (fs::is_directory(entry)) { directories.push_back(entry.path()); }
        }
        sort(directories.begin(), directories.end());

        // Iterate through the dataset.
        for (const auto &genreEntry : directories)
        {
            // Add new class if it does not exist.
            string genre = genreEntry.filename().string();
            if (m_classes.find(genre) == m_classes.end())
            {
                m_classes.insert(genre, torch::tensor(static_cast<int64_t>(m_classes.size()), torch::kLong));
            }

            // Collect the files for one genre and sort the in alphabetical order.
            vector<fs::path> files;
            for (const auto& entry : fs::directory_iterator(genreEntry))
            {
                if (entry.path().extension() == ".wav")
                {
                    files.push_back(entry.path());
                }
            }
            sort(files.begin(), files.end());

            // Iterate through one class and collect all data.
            for (const auto &sampleEntry : files)
            {
                m_data.push_back(m_preprocessor->run(sampleEntry));
                m_target.push_back(m_classes.at(genre));
            }
        }

        // Save new dataset.
        torch::serialize::OutputArchive dataset;
        dataset.write("data", m_data);
        dataset.write("target", m_target);
        dataset.write("classes", m_classes);
        dataset.save_to(datasetPath);
    }
}

AudioDataset::~AudioDataset()
{

}

torch::data::Example<> AudioDataset::get(size_t index)
{
    // Normalize and extract the data.
    torch::Tensor dataTensor = m_preprocessor->normalize(m_data[index]);
    torch::Tensor targetTensor = m_target[index];
    return torch::data::Example<>(dataTensor, targetTensor);
}

 torch::optional<size_t> AudioDataset::size() const
 {
    // Get dataset size.
    return m_data.size();
 }

Preprocessor * AudioDataset::preprocessor() const
{
    // Get the Preprocessor.
    return m_preprocessor;
}

c10::Dict<std::string, torch::Tensor> AudioDataset::classes() const
{
    // Get the classes.
    return m_classes;
}

AudioSubset::AudioSubset(AudioDataset * dataset, std::vector<size_t> indices) : m_dataset(dataset), m_indices(indices)
{

}

AudioSubset::~AudioSubset()
{

}

torch::data::Example<> AudioSubset::get(size_t index)
{
    // Extract the data.
    return m_dataset->get(m_indices[index]);
}

torch::optional<size_t> AudioSubset::size() const
{
    // Get subset size.
    return m_indices.size();
}

torch::Tensor AudioSubset::data() const
{
    // Initialize vector to store all data.
    vector<torch::Tensor> allData;

    // Retreive all of the data.
    for (size_t index : m_indices)
    {
        allData.push_back(m_dataset->get(index).data);
    }

    // Convert data vector to tensor.
    return torch::stack(allData, 0);
}
