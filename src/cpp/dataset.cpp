#include <algorithm>
#include "dataset.h"

using namespace std;

AudioDataset::AudioDataset(const fs::path& rootPath, Preprocessor * preprocessor) : preprocessor(preprocessor) {

    fs::path datasetPath = rootPath / "dataset_cpp.pt";

    if (fs::exists(datasetPath)) {
        // Preload existing dataset.
        torch::IValue dataIValue, targetIValue, classesIValue;
        torch::serialize::InputArchive dataset;
        dataset.load_from(datasetPath);
        dataset.read("data", dataIValue);
        dataset.read("target", targetIValue);
        dataset.read("classes", classesIValue);

        data = dataIValue.toTensorVector();
        target = targetIValue.toTensorVector();
        for (const auto& item : classesIValue.toGenericDict()) {
            string key = item.key().toStringRef();
            torch::Tensor value = item.value().toTensor();
            classes.insert(key, value);
        }
    }
    else {
        // Collect the directories (genres) and sort the in alphabetical order.
        vector<fs::path> directories;
        for (const auto& entry : fs::directory_iterator(rootPath)) {
            if (fs::is_directory(entry)) { directories.push_back(entry.path()); }
        }
        sort(directories.begin(), directories.end());

        // Iterate through the dataset.
        for (const auto& genreEntry : directories) {
            // Add new class if it does not exist.
            string genre = genreEntry.filename().string();
            if (classes.find(genre) == classes.end()) {
                classes.insert(genre, torch::tensor(static_cast<int64_t>(classes.size()), torch::kLong));
            }

            // Iterate through one class and collect all files.
            for (const auto& fileEntry : fs::directory_iterator(genreEntry)) {
                if (fileEntry.path().extension() == ".wav") {
                    data.push_back(preprocessor->run(fileEntry.path()));
                    target.push_back(classes.at(genre));
                }
            }
        }

        // Save new dataset.
        torch::serialize::OutputArchive dataset;
        dataset.write("data", data);
        dataset.write("target", target);
        dataset.write("classes", classes);
        dataset.save_to(datasetPath);
    }
}

torch::data::Example<> AudioDataset::get(size_t index) {
    // Normalize data.
    torch::Tensor dataTensor = preprocessor->normalize(data[index]);
    torch::Tensor targetTensor = target[index];
    return torch::data::Example<>(dataTensor, targetTensor);
}
