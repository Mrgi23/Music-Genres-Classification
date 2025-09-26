#include "Dataset.h"

#include <algorithm>
#include <omp.h>

using namespace std;

AudioDataset::AudioDataset(const fs::path & rootPath, Preprocessor * preprocessor) : m_preprocessor(preprocessor)
{

    fs::path datasetPath = rootPath / "dataset_cpp.pt";

    if (fs::exists(datasetPath))
    {
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
            int64_t key = item.key().toInt();
            string value = item.value().toStringRef();
            m_classes.insert(key, value);
        }
    }
    else
    {
        vector<pair<fs::path, torch::Tensor>> tasks;

        vector<fs::path> directories;
        for (const auto &entry : fs::directory_iterator(rootPath))
        {
            if (fs::is_directory(entry)) { directories.push_back(entry.path()); }
        }
        sort(directories.begin(), directories.end());

        for (const auto &genreEntry : directories)
        {
            string genre = genreEntry.filename().string();
            torch::Tensor classTarget = torch::tensor(static_cast<int64_t>(m_classes.size()), torch::kLong);
            m_classes.insert(classTarget.item<int64_t>(), genre);

            vector<fs::path> files;
            for (const auto& entry : fs::directory_iterator(genreEntry))
            {
                if (entry.path().extension() == ".wav")
                {
                    files.push_back(entry.path());
                }
            }
            sort(files.begin(), files.end());

            for (const auto &sampleEntry : files)
            {
                tasks.emplace_back(sampleEntry, classTarget);
            }
        }

        size_t nTasks = tasks.size();
        m_data.resize(nTasks);
        m_target.resize(nTasks);

        exception_ptr eptr;
        #pragma omp parallel for num_threads(omp_get_num_procs())
        for (size_t i = 0; i < nTasks; i++)
        {
            try
            {
                const auto& [file, classTarget] = tasks[i];
                auto data = m_preprocessor->ProcessFile(file);
                auto target = classTarget;

                m_data[i] = data;
                m_target[i] = target;
            }
            catch (...)
            {
                #pragma omp critical
                if (!eptr)
                    eptr = current_exception();
            }
        }

        if (eptr)
            rethrow_exception(eptr);

        torch::serialize::OutputArchive dataset;
        dataset.write("data", m_data);
        dataset.write("target", m_target);
        dataset.write("classes", m_classes);
        dataset.save_to(datasetPath);
    }
}

AudioDataset::~AudioDataset() = default;

c10::Dict<int64_t, string> AudioDataset::GetClasses() const
{
    return m_classes;
}

Preprocessor * AudioDataset::GetPreprocessor() const
{
    return m_preprocessor;
}

torch::data::Example<> AudioDataset::get(size_t index)
{
    torch::Tensor dataTensor = m_preprocessor->NormalizeData(m_data[index]);
    torch::Tensor targetTensor = m_target[index];
    return torch::data::Example<>(dataTensor, targetTensor);
}

 torch::optional<size_t> AudioDataset::size() const
 {
    return m_data.size();
 }

AudioSubset::AudioSubset(AudioDataset * dataset, vector<size_t> indices) : m_dataset(dataset), m_indices(indices)
{
    m_stackedData = torch::Tensor();
}

AudioSubset::~AudioSubset() = default;


AudioDataset * AudioSubset::GetDataset() const
{
    return m_dataset;
}

torch::Tensor AudioSubset::GetStackedData() const
{
    if (m_stackedData.defined())
        return m_stackedData;

    vector<torch::Tensor> allData;
    for (size_t index : m_indices)
        allData.push_back(m_dataset->get(index).data);

    m_stackedData = torch::stack(allData, 0);
    return m_stackedData;
}

torch::data::Example<> AudioSubset::get(size_t index)
{
    return m_dataset->get(m_indices[index]);
}

torch::optional<size_t> AudioSubset::size() const
{
    return m_indices.size();
}
