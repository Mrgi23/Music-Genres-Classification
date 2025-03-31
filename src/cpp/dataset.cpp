#include <algorithm>
#include "dataset.h"

using namespace std;

AudioDataset::AudioDataset(const fs::path& rootPath, Preprocessor * preprocessor) : preprocessor(preprocessor) {
    // Collect the directories (genres) and sort the in alphabetical order.
    vector<fs::path> directories;
    for (const auto& entry : fs::directory_iterator(rootPath)) { directories.push_back(entry.path()); }
    sort(directories.begin(), directories.end());

    // Iterate through the dataset.
    for (const auto& genreEntry : directories) {
        // Add new class if it does not exist.
        string genre = genreEntry.filename().string();
        if (classes.find(genre) == classes.end()) { classes[genre] = classes.size(); }

        // Iterate through one class and collect all files.
        for (const auto& fileEntry : fs::directory_iterator(genreEntry)) {
            if (fileEntry.path().extension() == ".wav") {
                files.push_back(fileEntry.path());
                labels.push_back(genre);
            }
        }
    }
}

torch::data::Example<> AudioDataset::get(size_t index) {
    // Retreive the audio file path and the label index.
    const fs::path filePath = files[index];
    const uint label = classes[labels[index]];

    // Extract features.
    vector<vector<float>> spectrogram;
    vector<vector<float>> mfcc;
    (*preprocessor).run(filePath, spectrogram, mfcc);

    // Flatten the 2D vector to a 1D vector for both spectrogram and mfcc.
    std::vector<float> spectrogramFlat;
    std::vector<float> mfccFlat;
    for (uint i = 0; i < spectrogram.size(); i++) {
        spectrogramFlat.insert(spectrogramFlat.end(), spectrogram[i].begin(), spectrogram[i].end());
        mfccFlat.insert(mfccFlat.end(), mfcc[i].begin(), mfcc[i].end());
    }

    // Convert flatten features to tensors.
    torch::Tensor spectrogramTensor = torch::tensor(
        spectrogramFlat,
        torch::kFloat32
    ).view({1, static_cast<int64_t>(spectrogram.size()), static_cast<int64_t>(spectrogram[0].size())});
    torch::Tensor mfccTensor = torch::tensor(
        mfccFlat,
        torch::kFloat32
    ).view({1, static_cast<int64_t>(mfcc.size()), static_cast<int64_t>(mfcc[0].size())});

    // Combine data and convert the label.
    torch::Tensor dataTensor = torch::cat({spectrogramTensor, mfccTensor}, 2);
    torch::Tensor targetTensor = torch::tensor({static_cast<int64_t>(label)}, torch::kLong);

    return torch::data::Example<>(dataTensor, targetTensor);
}
