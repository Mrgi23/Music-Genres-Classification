from downloader import Downloader
from preprocessor import PreprocessorConfig, Preprocessor
from dataset import AudioDataset
from model import MusicModel
from trainer import DeviceManager, Trainer

import copy
import json
from pathlib import Path
import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

def init(config: dict, root_path: Path, indices_path: Path) -> tuple[Subset, Subset, Subset]:
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
    tuple[Subset, Subset, Subset]
        Loaded subsets.
    """
    print("Downloading dataset...")

    Downloader(root_path).download_and_extract()

    print("Dataset dowloaded.")

    pcfg = PreprocessorConfig(
        config["SIZE"],
        config["NUM_FFT"],
        config["HOP"],
        config["NUM_MELS"],
        config["NUM_MFCC"]
    )
    preprocessor = Preprocessor(pcfg)

    print("Loading dataset...")

    dataset = AudioDataset(root_path, preprocessor)

    print("Dataset loaded.")

    with open(indices_path, "r") as jf:
        indices = json.load(jf)
        train_dataset = Subset(dataset, indices["train"])
        val_dataset = Subset(dataset, indices["val"])
        test_dataset = Subset(dataset, indices["test"])

    data = [train_dataset.dataset[i][0] for i in train_dataset.indices]
    data = torch.stack(data, dim=0).float()
    preprocessor.mean = data.mean(dim=0)
    preprocessor.std = data.std(dim=0, unbiased=False).clamp_min(1e-8)

    return train_dataset, val_dataset, test_dataset

def load_model(config: dict, model_path: Path) -> MusicModel:
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
        url = f"{config['BASE_URL']}/{config['PACKAGE']}/{config['VERSION']}/{config['PACKAGE']}-py.pt"
        Downloader.download_from_url(model_path, url)

    print("Model downloaded.")

    model = MusicModel()
    model.load_state_dict(torch.load(model_path))
    return model

def load_scheduler(config: dict, optimizer: Optimizer) -> ReduceLROnPlateau:
    """
    Initialize the learning rate scheduler.

    Creates a ReduceLROnPlateau scheduler if it does not already exist,
    using configuration values for mode, factor, and patience.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing parameters.
    optimizer : Optimizer
        Optimizer to update.

    Returns
    -------
    ReduceLROnPlateau
        Loaded scheduler.
    """
    scheduler = ReduceLROnPlateau(
        optimizer,
        config["MODE"],
        config["FACTOR"],
        config["PATIENCE"]
    )
    return scheduler

def train(config: dict, scheduler_config: dict, train_dataset: Subset, val_dataset: Subset) -> MusicModel:
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
    train_dataset : Subset
        Dataset used for training.
    val_dataset : Subset
        Dataset used for evaluation.

    Returns
    -------
    MusicModel
        Trained model.
    """
    pin = torch.cuda.is_available()
    train_dataloader = DataLoader(train_dataset, config["BATCH_SIZE"], True, num_workers=config["WORKERS"], pin_memory=pin)
    val_dataloader = DataLoader(val_dataset, config["BATCH_SIZE"], False, num_workers=config["WORKERS"], pin_memory=pin)

    model = MusicModel()

    opt = Adam(model.parameters(), lr=config["LR"])
    trainer = Trainer(model, opt)

    scheduler = load_scheduler(scheduler_config, opt)

    print("Training starting...")

    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, config["EPOCHS"] + 1):
        train_loss, train_acc = trainer.train_model(train_dataloader)
        val_loss, val_acc = trainer.eval_model(val_dataloader)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d} | " +
            f"Train loss: {train_loss:.6f} | Train accuracy: {train_acc:.6f} | " +
            f"Validation loss: {val_loss:.6f} | Validation accuracy: {val_acc:.6f}"
        )

    print("Training ended.")

    model.load_state_dict(best_state)
    return model

def evaluate(config: dict, model: MusicModel, test_dataset: Subset) -> None:
    """
    Evaluate the model on the test set.

    Creates a test dataloader and runs evaluation with the trainer.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing parameters.
    model : MusicModel
        Trained/Loaded model.
    test_dataset : Subset
        Dataset used for testing.
    """
    pin = torch.cuda.is_available()
    test_dataloader = DataLoader(test_dataset, config["BATCH_SIZE"], False, num_workers=config["WORKERS"], pin_memory=pin)

    trainer = Trainer(model, None)

    print("Evaluating model on test dataset...")

    test_loss, test_acc  = trainer.eval_model(test_dataloader)
    print(f"Test loss: {test_loss} | Test accuracy: {test_acc}")

def predict(model: MusicModel, preprocessor: Preprocessor, file_path: Path, classes: dict) -> None:
    """
    Predict genre for a single audio file.

    Runs the preprocessor on the input file, normalizes the features,
    performs a forward pass with the model in evaluation mode, and
    extract the predicted genre label.

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
    model.to(DeviceManager.get())

    x = preprocessor.normalize_data(preprocessor.process_file(file_path))
    x = x.to(DeviceManager.get())

    y = model(x).argmax(dim=1).to(torch.long)

    print(f"Input sound file {file_path} is of genre: {classes[y.item()].capitalize()}")
