#include "Downloader.h"
#include "Trainer.h"

#include <cctype>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <toml++/toml.hpp>
#include <torch/cuda.h>
#include <type_traits>

using json = nlohmann::json;
using namespace std;

namespace global
{
    // Global
    fs::path root;
    PreprocessorConfig * pcfg = nullptr;
    Preprocessor * preprocessor = nullptr;
    AudioDataset * dataset = nullptr;
    map<string, AudioSubset*> subsets = {
        {"train", nullptr},
        {"val", nullptr},
        {"test", nullptr}
    };
    MusicModel * model = nullptr;

    // Only if training.
    ReduceLROnPlateau * scheduler = nullptr;
    const map<string, OptimizerType> optTypes = {
        {"Adam", OptimizerType::Adam},
        {"AdamW", OptimizerType::AdamW},
        {"RMSprop", OptimizerType::RMSprop},
        {"SGD", OptimizerType::SGD}
    };
    Trainer * trainer = nullptr;
}

/**
 * @brief Print usage instructions for the application.
 */
void PrintHelp()
{
    cout << "usage: musicnet [-h] [-wd WORKING_DIR] [-p PREDICT] [-f] [-s]" << endl
         << endl
         << "Application" << endl
         << endl
         << "options:" << endl
         << endl
         << "-h, --help                     show this help message and exit" << endl
         << "-wd WORKING_DIR, --working-dir WORKING_DIR"  << endl
         << "                               path to the project working directory" << endl
         << "-p PREDICT, --predict PREDICT" << endl
         << "                               sound file path to predict genre. If not provided, model is evaluated on the  test dataset" << endl
         << "-f, --force                    force model training" << endl
         << "-s, --save                     save new model (only if force training)"
         << endl;
}

/**
 * @brief Get full path relative to the project root.
 *
 * @param[in] path Relative path to append to the root.
 * @return fs::path Absolute path under the root directory.
 */
inline fs::path GetFullPath(const fs::path & path)
{
    return global::root / path;
}

/**
 * @brief Retrieve a typed value from a TOML table by key.
 *
 * This function attempts to extract a value of type `T` from the provided
 * TOML table at the specified key. If the key is present and the type matches,
 * the value is returned. Otherwise, if a default value is provided, it will
 * be returned. If neither condition is met, an exception is thrown.
 *
 * @tparam T Type of the value to retrieve.
 * @param[in] table The TOML table to query.
 * @param[in] key The key whose value should be retrieved.
 * @param[in] defaultValue Optional default value to return if the key is missing
 *                         or invalid. Defaults to `nullopt`.
 *
 * @return The retrieved value of type `T`.
 *
 * @throws invalid_argument If the key is missing or the value type
 *                               does not match and no default value is provided.
 */
template <typename T>
T GetParam(const toml::table & table, const string & key, optional<T> defaultValue = nullopt)
{
    using useT = conditional_t<is_integral_v<T>, int64_t, T>;

    auto opt = table[key].value<useT>();
    if (opt)
        return *opt;

    if (defaultValue)
        return *defaultValue;

    throw invalid_argument("Missing or invalid key: " + key);
}

/**
 * @brief Initialization.
 *
 * Downloads and extracts the dataset, sets up preprocessing
 * configuration, loads the dataset and subsets (train/val/test),
 * and initializes the MusicModel. Computes and stores dataset
 * mean and standard deviation for normalization.
 *
 * @param[in] config TOML configuration table containing parameters.
 * @param[in] rootPath Path where the dataset should be stored.
 * @param[in] indicesPath JSON file containing dataset splits.
 */
void Init(const toml::table & config, const fs::path & rootPath, const fs::path & indicesPath)
{
    cout << "Downloading dataset..." << endl;

    Downloader(rootPath).DownloadAndExtract();

    cout << "Dataset dowloaded." << endl;

    global::pcfg = new PreprocessorConfig(
        GetParam<uint>(config, "SIZE"),
        GetParam<uint>(config, "NUM_FFT"),
        GetParam<uint>(config, "HOP"),
        GetParam<uint>(config, "NUM_MELS"),
        GetParam<uint>(config, "NUM_MFCC")
    );
    global::preprocessor = new Preprocessor(*global::pcfg);

    cout << "Loading dataset..." << endl;

    global::dataset = new AudioDataset(rootPath, global::preprocessor);

    cout << "Dataset loaded." << endl;

    ifstream jsonFile(indicesPath);
    json splits;
    jsonFile >> splits;
    for (const string& key : {"train", "val", "test"})
        global::subsets[key] = new AudioSubset(global::dataset, splits[key].get<vector<size_t>>());

    torch::Tensor data = global::subsets["train"]->GetStackedData().to(torch::kFloat32);
    global::preprocessor->SetMean(data.mean(0));
    global::preprocessor->SetStd(data.std(0, false).clamp_min(1e-8));

    global::model = new MusicModel();
}

/**
 * @brief Load a trained model from disk.
 *
 * If the specified model file does not exist, downloads it from
 * the configured URL before loading. Updates the global model.
 *
 * @param[in] config TOML configuration table containing parameters.
 * @param[in] modelPath Path to the serialized model file (.pt).
 */
void LoadModel(const toml::table & config, const fs::path & modelPath)
{
    cout << "Downloading model..." << endl;

    if (!fs::exists(modelPath))
    {
        string baseUrl = GetParam<string>(config, "BASE_URL");
        string package = GetParam<string>(config, "PACKAGE");
        string version = GetParam<string>(config, "VERSION");
        string url = baseUrl + "/" + package + "/" + version + "/" + package + "-cpp.pt";
        Downloader::DownloadFromUrl(modelPath, url);
    }
    global::model->Load(modelPath);

    cout << "Model dowloaded." << endl;
}

/**
 * @brief Initialize the learning rate scheduler.
 *
 * Creates a ReduceLROnPlateau scheduler if it does not already exist,
 * using configuration values for mode, factor, and patience.
 *
 * @param[in] config TOML configuration table containing parameters.
 */
void LoadScheduler(const toml::table & config)
{
    if (!global::scheduler)
        global::scheduler = new ReduceLROnPlateau(
            GetParam<string>(config, "MODE"),
            GetParam<float>(config, "FACTOR"),
            GetParam<uint>(config, "PATIENCE")
        );
}

/**
 * @brief Train the model.
 *
 * Creates training and validation dataloaders, initializes the
 * trainer (and attaches a scheduler if available), and runs training
 * for the configured number of epochs. Saves the best model based on
 * validation accuracy and restores it after training.
 *
 * @param[in] config TOML configuration table containing parameters.
 */
void Train(const toml::table & config)
{
    unique_ptr<AudioDataloader<RandomSampler>> trainDataloader = torch::data::make_data_loader<RandomSampler>(
        global::subsets["train"]->map(Stack<>()),
        torch::data::DataLoaderOptions()
        .batch_size(GetParam<size_t>(config, "BATCH_SIZE"))
        .workers(GetParam<int>(config, "WORKERS"))
    );
    unique_ptr<AudioDataloader<SequentialSampler>> valDataloader = torch::data::make_data_loader<SequentialSampler>(
        global::subsets["val"]->map(Stack<>()),
        torch::data::DataLoaderOptions()
        .batch_size(GetParam<size_t>(config, "BATCH_SIZE"))
        .workers(GetParam<int>(config, "WORKERS"))
    );

    if (!global::trainer)
    {
        OptimizerType type = global::optTypes.at(GetParam<string>(config, "TYPE"));
        OptimizerConfig ocfg = OptimizerConfig(GetParam<double>(config, "LR"));
        global::trainer = new Trainer(*global::model, type, ocfg);
        if (global::scheduler)
            global::trainer->AttachScheduler(global::scheduler);
    }

    cout << "Training starting..." << endl;

    float bestValAcc = -1.0f;
    fs::path bestModel = fs::temp_directory_path() / "bestModel.pt";
    for (uint epoch = 1; epoch < GetParam<uint>(config, "EPOCHS") + 1; epoch++)
    {
        float trainLoss, trainAcc, valLoss, valAcc;
        global::trainer->TrainModel(*trainDataloader, trainLoss, trainAcc);
        global::trainer->EvalModel(*valDataloader, valLoss, valAcc);

        global::scheduler->UpdateLR(valAcc);

        if (valAcc > bestValAcc)
        {
            bestValAcc = valAcc;
            global::model->Save(bestModel);
        }

        cout << "Epoch "
             << setw(3) << setfill('0') << epoch
             << " | Train loss: "            << fixed << setprecision(6) << trainLoss
             << " | Validation loss: "        << fixed << setprecision(6) << valLoss
             << " | Train accuracy: "       << fixed << setprecision(6) << trainAcc
             << " | Validation accuracy: "   << fixed << setprecision(6) << valAcc
             << endl;
    }

    cout << "Training ended." << endl;

    global::model->Load(bestModel);
    fs::remove(bestModel);
}

/**
 * @brief Evaluate the model on the test set.
 *
 * Creates a test dataloader and runs evaluation with the trainer.
 *
 * @param[in] config TOML configuration table containing parameters.
 */
void Evaluate(const toml::table & config)
{
    unique_ptr<AudioDataloader<SequentialSampler>> testDataloader = torch::data::make_data_loader<SequentialSampler>(
        global::subsets["test"]->map(Stack<>()),
        torch::data::DataLoaderOptions()
        .batch_size(GetParam<size_t>(config, "BATCH_SIZE"))
        .workers(GetParam<int>(config, "WORKERS"))
    );

    if (!global::trainer)
    {
        OptimizerType type = global::optTypes.at(GetParam<string>(config, "TYPE"));
        OptimizerConfig ocfg = OptimizerConfig(GetParam<double>(config, "LR"));
        global::trainer = new Trainer(*global::model, type, ocfg);
    }

    cout << "Evaluating model on test dataset..." << endl;

    float loss, acc;
    global::trainer->EvalModel(*testDataloader, loss, acc);
    cout << "Test loss: "        << fixed << setprecision(6) << loss
         << " | Test accuracy: " << fixed << setprecision(6) << acc
         << endl;
}

/**
 * @brief Predict genre for a single audio file.
 *
 * Runs the preprocessor on the input file, normalizes the features,
 * performs a forward pass with the model in evaluation mode, and
 * extract the predicted genre label.
 *
 * @param[in] filePath Path to the audio file to classify.
 */
void Predict(const fs::path & filePath)
{
    global::model->to(DeviceManager::Get());
    global::model->eval();

    torch::Tensor x = global::preprocessor->NormalizeData(global::preprocessor->ProcessFile(filePath));
    x = x.to(DeviceManager::Get());

    torch::Tensor y = global::model->forward(x).argmax(1).to(torch::kLong);

    string genre = global::dataset->GetClasses().at(y.item<int64_t>());
    genre[0] = toupper(genre[0]);
    cout << "Input sound file " << filePath.c_str() << " is of genre: " << genre << endl;
}

/**
 * @brief Clean up global resources.
 *
 * Frees all allocated global objects including preprocessor
 * config, preprocessor, dataset, subsets, model, scheduler, and trainer.
 * Resets global pointers to nullptr to prevent dangling references.
 */
void cleanUp()
{
    if (global::pcfg)
    {
        delete global::pcfg;
        global::pcfg = nullptr;
    }

    if (global::preprocessor)
    {
        delete global::preprocessor;
        global::preprocessor = nullptr;
    }

    if (global::dataset)
    {
        delete global::dataset;
        global::dataset = nullptr;
    }

    for (const string& key : {"train", "val", "test"})
    {
        if (global::subsets[key])
        {
            delete global::subsets[key];
            global::subsets[key] = nullptr;
        }
    }

    if (global::model)
    {
        delete global::model;
        global::model = nullptr;
    }

    if(global::scheduler)
    {
        delete global::scheduler;
        global::scheduler = nullptr;
    }

    if (global::trainer)
    {
        delete global::trainer;
        global::trainer = nullptr;
    }
}
