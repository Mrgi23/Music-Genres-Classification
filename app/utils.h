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

namespace global
{
  fs::path root;
  PreprocessorConfig* pcfg = nullptr;
  Preprocessor* preprocessor = nullptr;
  AudioDataset* dataset = nullptr;
  std::map<std::string, AudioSubset*> subsets = {
    {"train", nullptr},
    {"val", nullptr},
    {"test", nullptr}
  };
  MusicModel* model = nullptr;

  ReduceLROnPlateau* scheduler = nullptr;
  const std::map<std::string, OptimizerType> optTypes = {
    {"Adam", OptimizerType::Adam},
    {"AdamW", OptimizerType::AdamW},
    {"RMSprop", OptimizerType::RMSprop},
    {"SGD", OptimizerType::SGD}
  };
  Trainer* trainer = nullptr;
}

/**
 * @brief Print usage instructions for the application.
 */
void help()
{
  std::cout << "usage: musicnet [-h] [-wd WORKING_DIR] [-p PREDICT] [-f] [-s]" << std::endl
       << std::endl
       << "Application" << std::endl
       << std::endl
       << "options:" << std::endl
       << std::endl
       << "-h, --help                     show this help message and exit" << std::endl
       << "-wd WORKING_DIR, --working-dir WORKING_DIR" << std::endl
       << "                               path to the project working directory" << std::endl
       << "-p PREDICT, --predict PREDICT" << std::endl
       << "                               sound file path to predict genre. If not provided, model is evaluated on the  test dataset" << std::endl
       << "-f, --force                    force model training" << std::endl
       << "-s, --save                     save new model (only if force training)"
       << std::endl;
}

/**
 * @brief Get full path relative to the project root.
 *
 * @param[in] path Relative path to append to the root.
 * @return fs::path Absolute path under the root directory.
 */
inline fs::path fullPath(const fs::path& path)
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
T param(const toml::table& table, const std::string& key, std::optional<T> defaultValue = std::nullopt)
{
  using useT = std::conditional_t<std::is_integral_v<T>, int64_t, T>;

  auto opt = table[key].value<useT>();
  if (opt)
    return *opt;

  if (defaultValue)
    return *defaultValue;

  throw std::invalid_argument("Missing or invalid key: " + key);
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
void init(const toml::table& config, const fs::path& rootPath, const fs::path& indicesPath)
{
  std::cout << "Downloading dataset..." << std::endl;

  Downloader(rootPath).downloadAndExtract();

  std::cout << "Dataset dowloaded." << std::endl;

  global::pcfg = new PreprocessorConfig(
    param<uint>(config, "SIZE"),
    param<uint>(config, "NUM_FFT"),
    param<uint>(config, "HOP"),
    param<uint>(config, "NUM_MELS"),
    param<uint>(config, "NUM_MFCC")
  );
  global::preprocessor = new Preprocessor(*global::pcfg);

  std::cout << "Loading dataset..." << std::endl;

  global::dataset = new AudioDataset(rootPath, global::preprocessor);

  std::cout << "Dataset loaded." << std::endl;

  std::ifstream jsonFile(indicesPath);
  json splits;
  jsonFile >> splits;
  for (const std::string& key : {"train", "val", "test"})
    global::subsets[key] = new AudioSubset(global::dataset, splits[key].get<std::vector<size_t>>());

  torch::Tensor data = global::subsets["train"]->stackedData().to(torch::kFloat32);
  global::preprocessor->setMean(data.mean(0));
  global::preprocessor->setStd(data.std(0, false).clamp_min(1e-8));

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
void loadModel(const toml::table& config, const fs::path& modelPath)
{
  std::cout << "Downloading model..." << std::endl;

  if (!fs::exists(modelPath))
  {
    std::string baseUrl = param<std::string>(config, "BASE_URL");
    std::string package = param<std::string>(config, "PACKAGE");
    std::string version = param<std::string>(config, "VERSION");
    std::string url = baseUrl + "/" + package + "/" + version + "/" + package + "-cpp.pt";
    Downloader::downloadFromUrl(modelPath, url);
  }

  std::cout << "Model downloaded." << std::endl;
  global::model->load(modelPath);
}

/**
 * @brief Initialize the learning rate scheduler.
 *
 * Creates a ReduceLROnPlateau scheduler if it does not already exist,
 * using configuration values for mode, factor, and patience.
 *
 * @param[in] config TOML configuration table containing parameters.
 */
void loadScheduler(const toml::table& config)
{
  if (!global::scheduler)
    global::scheduler = new ReduceLROnPlateau(
      param<std::string>(config, "MODE"),
      param<float>(config, "FACTOR"),
      param<uint>(config, "PATIENCE")
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
void train(const toml::table& config)
{
  std::unique_ptr<AudioDataloader<RandomSampler>> trainDataloader = torch::data::make_data_loader<RandomSampler>(
    global::subsets["train"]->map(Stack<>()),
    torch::data::DataLoaderOptions()
      .batch_size(param<size_t>(config, "BATCH_SIZE"))
      .workers(param<int>(config, "WORKERS"))
  );
  std::unique_ptr<AudioDataloader<SequentialSampler>> valDataloader = torch::data::make_data_loader<SequentialSampler>(
    global::subsets["val"]->map(Stack<>()),
    torch::data::DataLoaderOptions()
      .batch_size(param<size_t>(config, "BATCH_SIZE"))
      .workers(param<int>(config, "WORKERS"))
  );

  if (!global::trainer)
  {
    OptimizerType type = global::optTypes.at(param<std::string>(config, "TYPE"));
    OptimizerConfig ocfg = OptimizerConfig(param<double>(config, "LR"));
    global::trainer = new Trainer(*global::model, type, ocfg);
    if (global::scheduler)
      global::trainer->attach(global::scheduler);
  }

  std::cout << "Training starting..." << std::endl;

  float bestValAcc = -1.0f;
  fs::path bestModel = fs::temp_directory_path() / "bestModel.pt";
  for (uint epoch = 1; epoch < param<uint>(config, "EPOCHS") + 1; epoch++)
  {
    float trainLoss, trainAcc, valLoss, valAcc;
    global::trainer->train(*trainDataloader, trainLoss, trainAcc);
    global::trainer->eval(*valDataloader, valLoss, valAcc);

    global::scheduler->update(valAcc);

    if (valAcc > bestValAcc)
    {
      bestValAcc = valAcc;
      global::model->save(bestModel);
    }

    std::cout << "Epoch "
         << std::setw(3) << std::setfill('0') << epoch
         << " | Train loss: " << std::fixed << std::setprecision(6) << trainLoss
         << " | Validation loss: " << std::fixed << std::setprecision(6) << valLoss
         << " | Train accuracy: " << std::fixed << std::setprecision(6) << trainAcc
         << " | Validation accuracy: " << std::fixed << std::setprecision(6) << valAcc
         << std::endl;
  }

  std::cout << "Training ended." << std::endl;

  global::model->load(bestModel);
  fs::remove(bestModel);
}

/**
 * @brief Evaluate the model on the test set.
 *
 * Creates a test dataloader and runs evaluation with the trainer.
 *
 * @param[in] config TOML configuration table containing parameters.
 */
void evaluate(const toml::table& config)
{
  std::unique_ptr<AudioDataloader<SequentialSampler>> testDataloader = torch::data::make_data_loader<SequentialSampler>(
    global::subsets["test"]->map(Stack<>()),
    torch::data::DataLoaderOptions()
      .batch_size(param<size_t>(config, "BATCH_SIZE"))
      .workers(param<int>(config, "WORKERS"))
    );

  if (!global::trainer)
  {
    OptimizerType type = global::optTypes.at(param<std::string>(config, "TYPE"));
    OptimizerConfig ocfg = OptimizerConfig(param<double>(config, "LR"));
    global::trainer = new Trainer(*global::model, type, ocfg);
  }

  std::cout << "Evaluating model on test dataset..." << std::endl;

  float loss, acc;
  global::trainer->eval(*testDataloader, loss, acc);
  std::cout << "Test loss: " << std::fixed << std::setprecision(6) << loss
       << " | Test accuracy: " << std::fixed << std::setprecision(6) << acc
       << std::endl;
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
void predict(const fs::path& filePath)
{
  global::model->to(DeviceManager::get());
  global::model->eval();

  torch::Tensor x = global::preprocessor->normalizeData(global::preprocessor->processFile(filePath));
  x = x.to(DeviceManager::get());

  torch::Tensor y = global::model->forward(x).argmax(1).to(torch::kLong);

  std::string genre = global::dataset->classes().at(y.item<int64_t>());
  genre[0] = toupper(genre[0]);
  std::cout << "Input sound file " << filePath.c_str() << " is of genre: " << genre << std::endl;
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

  for (const std::string& key : {"train", "val", "test"})
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

  if (global::scheduler)
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
