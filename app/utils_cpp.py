import musicnet as mgc
import json
from pathlib import Path
import tempfile
import torch

def init(config: dict, root_path: Path, indices_path: Path) -> tuple[mgc.AudioSubset, mgc.AudioSubset, mgc.AudioSubset]:
  """
  Initialization.

  Downloads and extracts the dataset, sets up preprocessing
  configuration, loads the dataset and subsets (train/val/test),
  and initializes the MusicModel. Computes and stores dataset
  mean and standard deviation for normalization.

  Parameters
  ----------
  config : dict
    Configuration dictionary containing parameters.
  root_path : Path
    Path where the dataset should be stored.
  indices_path : Path
    JSON file containing dataset splits.

  Returns
  -------
  tuple[AudioSubset, AudioSubset, AudioSubset]
    Loaded subsets.
  """
  print("Downloading dataset...")

  mgc.Downloader(root_path).download_and_extract()

  print("Dataset dowloaded.")

  pcfg = mgc.PreprocessorConfig(
    config["SIZE"],
    config["NUM_FFT"],
    config["HOP"],
    config["NUM_MELS"],
    config["NUM_MFCC"]
  )
  preprocessor = mgc.Preprocessor(pcfg)

  print("Loading dataset...")

  dataset = mgc.AudioDataset(root_path, preprocessor)

  print("Dataset loaded.")

  with open(indices_path, "r") as jf:
    indices = json.load(jf)
    train_dataset = mgc.AudioSubset(dataset, indices["train"])
    val_dataset = mgc.AudioSubset(dataset, indices["val"])
    test_dataset = mgc.AudioSubset(dataset, indices["test"])

  data = train_dataset.stacked_data()
  preprocessor.set_mean(data.mean(dim=0))
  preprocessor.set_std(data.std(dim=0, unbiased=False).clamp_min(1e-8))

  return train_dataset, val_dataset, test_dataset

def load_model(config: dict, model_path: Path) -> mgc.MusicModel:
  """
  Load a trained model from disk.

  If the specified model file does not exist, downloads it from
  the configured URL before loading.

  Parameters
  ----------
  config : dict
    Configuration dictionary containing parameters.
  model_path : Path
    Path to the model state dictionary file (.pt).

  Returns
  -------
  MusicModel
    Loaded model.
  """
  print("Downloading model...")

  if not model_path.exists():
    url = f"{config['BASE_URL']}/{config['PACKAGE']}/{config['VERSION']}/{config['PACKAGE']}-cpp.pt"
    mgc.Downloader.download_from_url(model_path, url)

  print("Model downloaded.")

  model = mgc.MusicModel()
  model.load(model_path)
  return model

def load_scheduler(config: dict) -> mgc.ReduceLROnPlateau:
  """
  Initialize the learning rate scheduler.

  Creates a ReduceLROnPlateau scheduler if it does not already exist,
  using configuration values for mode, factor, and patience.

  Parameters
  ----------
  config : dict
    Configuration dictionary containing parameters.

  Returns
  -------
  ReduceLROnPlateau
    Loaded scheduler.
  """
  scheduler = mgc.ReduceLROnPlateau(
    config["MODE"],
    config["FACTOR"],
    config["PATIENCE"]
  )
  return scheduler

def train(config: dict, scheduler_config: dict, train_dataset: mgc.AudioSubset, val_dataset: mgc.AudioSubset) -> mgc.MusicModel:
  """
  Train the model.

  Creates training and validation dataloaders, initializes the
  trainer (and attaches a scheduler if available), and runs training
  for the configured number of epochs. Saves the best model based on
  validation accuracy and restores it after training.

  Parameters
  ----------
  config : dict
    Configuration dictionary containing parameters.
  scheduler_config : dict
    Configuration dictionary containing scheduler's parameters.
  train_dataset : AudioSubset
    Dataset used for training.
  val_dataset : AudioSubset
    Dataset used for evaluation.

  Returns
  -------
  MusicModel
    Trained model.
  """
  model = mgc.MusicModel()

  optTypes = {
    "Adam": mgc.OptimizerType.Adam,
    "AdamW": mgc.OptimizerType.AdamW,
    "RMSprop": mgc.OptimizerType.RMSprop,
    "SGD": mgc.OptimizerType.SGD
  }
  ocfg = mgc.OptimizerConfig(config["LR"])

  trainer = mgc.Trainer(model, optTypes[config["TYPE"]], ocfg)

  scheduler = load_scheduler(scheduler_config)
  trainer.attach(scheduler)

  print("Training starting...")

  best_val_acc = -1.0
  best_model = Path(tempfile.gettempdir()) / "best_model.pt"
  for epoch in range(1, config["EPOCHS"] + 1):
    train_loss, train_acc = trainer.train(train_dataset, config["BATCH_SIZE"], config["WORKERS"])
    val_loss, val_acc = trainer.eval(val_dataset, config["BATCH_SIZE"], config["WORKERS"])

    scheduler.update(val_acc)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      model.save(best_model)

    print(
      f"Epoch {epoch:03d} | " +
      f"Train loss: {train_loss:.6f} | Train accuracy: {train_acc:.6f} | " +
      f"Validation loss: {val_loss:.6f} | Validation accuracy: {val_acc:.6f}"
    )

  print("Training ended.")

  model.load(best_model)
  best_model.unlink()
  return model

def evaluate(config: dict, model: mgc.MusicModel, test_dataset: mgc.AudioSubset) -> None:
  """
  Evaluate the model on the test set.

  Creates a test dataloader and runs evaluation with the trainer.

  Parameters
  ----------
  config : dict
    Configuration dictionary containing parameters.
  model : MusicModel
    Trained/Loaded model.
  test_dataset : AudioSubset
    Dataset used for testing.
  """
  optTypes = {
      "Adam": mgc.OptimizerType.Adam,
      "AdamW": mgc.OptimizerType.AdamW,
      "RMSprop": mgc.OptimizerType.RMSprop,
      "SGD": mgc.OptimizerType.SGD
  }
  ocfg = mgc.OptimizerConfig(config["LR"])

  trainer = mgc.Trainer(model, optTypes[config["TYPE"]], ocfg)

  print("Evaluating model on test dataset...")

  test_loss, test_acc  = trainer.eval(test_dataset, config["BATCH_SIZE"], config["WORKERS"])
  print(f"Test loss: {test_loss} | Test accuracy: {test_acc}")

def predict(model: mgc.MusicModel, preprocessor: mgc.Preprocessor, file_path: Path, classes: dict) -> None:
  """
  Predict genre for a single audio file.

  Runs the preprocessor on the input file, normalizes the features,
  performs a forward pass with the model in evaluation mode, and
  extracts the predicted genre label.

  Parameters
  ----------
  model : MusicModel
    Trained/Loaded model.
  preprocessor : Preprocessor
    Preprocessor to process input file.
  file_path : Path
    Path to the audio file to classify.
  classes : dict
    Predictions-to-genres map.
  """
  model.eval()
  model.to(mgc.DeviceManager.get())

  x = preprocessor.normalize_data(preprocessor.process_file(file_path))
  x = x.to(mgc.DeviceManager.get())

  y = model(x).argmax(dim=1).to(torch.long)

  print(f"Input sound file {file_path} is of genre: {classes[y.item()].capitalize()}")
