from enum import Enum
from pathlib import Path
import torch

class Downloader():
    """
    Downloader

    Utility class to download and extract the GTZAN dataset (or any dataset) from a URL.

    The Downloader handles downloading a dataset archive from Kaggle (or other sources),
    saving it locally, and extracting it into the specified root directory.
    It also provides helper methods for file existence checks and writing to disk.
    """
    def __init__(self, root_path: Path, url: str = ...) -> None:
        """
        Construct a new Downloader object.

        Parameters
        ----------
        root_path : Path
            Root path where the dataset will be stored.
        url : str, optional
            URL from which the dataset should be downloaded, by default the GTZAN dataset Kaggle API endpoint.
        """
        ...

    def download_and_extract(self) -> None:
        """
        Download a file from a given URL and save it to the given path.

        Parameters
        ----------
        file_path : Path
            Destination path where the file will be saved.
        url : str
            The URL to download the file from.
        """
        ...

    def get_root_path(self) -> str:
        """
        Get the root path where the dataset will be extracted.

        Returns
        -------
        Path
            The root path.
        """
        ...

    @staticmethod
    def download_from_url(file_path: Path, url: str) -> None:
        """
        Download the dataset and extract it into the root directory.
        This method checks if the dataset already exists, downloads it if missing,
        and extracts the archive into the root path.
        """
        ...

class PreprocessorConfig():
    """
    PreprocessorConfig

    Configuration parameters for audio preprocessing.
    """
    def __init__(
        self,
        size: int = ...,
        nfft: int = ...,
        hop: int = ...,
        nmels: int = ...,
        nmfcc: int = ...,
    ) -> None:
        """
        Construct a new PreprocessorConfig with optional overrides.

        Parameters
        ----------
        size : int, optional
            Input signal length, by default 660,000 samples (≈30s at 22.05kHz).
        nfft : int, optional
            FFT size for spectral analysis, by default 1024.
        hop : int, optional
            Hop size between frames, by default 512.
        nmels : int, optional
            Number of mel bands, by default 128.
        nmfcc : int, optional
            Number of MFCC coefficients, by default 13.
        """
        ...

class Preprocessor():
    """
    Preprocessor

    Audio preprocessing pipeline for feature extraction and normalization.

    This class uses the Aubio library to extract MFCC features from audio files
    and prepares them as Torch tensors for downstream ML tasks. It also supports
    normalization using precomputed mean and standard deviation tensors.
    """
    def __init__(self, cfg: PreprocessorConfig = ...) -> None:
        """
        Construct a new Preprocessor object.

        Parameters
        ----------
        cfg : PreprocessorConfig, optional
            Configuration parameters for preprocessing, by default default config.
        """
        ...

    def get_mean(self) -> torch.Tensor:
        """
        Get the mean tensor used for normalization.

        Returns
        -------
        torch.Tensor
            Mean tensor.
        """
        ...

    def set_mean(self, mean: torch.Tensor) -> None:
        """
        Set the mean tensor used for normalization.

        Returns
        -------
        torch.Tensor
            Mean tensor.
        """
        ...

    def get_std(self) -> torch.Tensor:
        """
        Get the standard deviation tensor used for normalization.

        Returns
        -------
        torch.Tensor
            Standard deviation tensor.
        """
        ...

    def set_std(self, std: torch.Tensor) -> None:
        """
        Set the standard deviation tensor used for normalization.

        Returns
        -------
        torch.Tensor
            Standard deviation tensor.
        """
        ...

    def process_file(self, file_path: Path) -> torch.Tensor:
        """
        Process an audio file and extract MFCC features.

        Loads an audio file, applies windowing and FFT, computes MFCCs for
        each frame, and returns them as a Torch tensor.

        Parameters
        ----------
        file_path : Path
            Path to the audio file.

        Returns
        -------
        torch.Tensor
            Extracted MFCC features of shape (1, NumFrames, NumMFCC)
        """
        ...

    def normalize_data(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize features using stored mean and standard deviation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of features.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        ...

class AudioDataset():
    """
    AudioDataset

    Dataset class for loading and preprocessing audio data.

    This dataset loads audio files from a directory, applies preprocessing
    (via Preprocessor), and provides data/target pairs as tuple[torch.Tensor, torch.Tensor].
    Data can be preloaded from a serialized .pt archive to save time.
    """
    def __init__(self, root_path: Path, preprocessor: Preprocessor) -> None:
        """
        Construct a new AudioDataset object.

        If a serialized dataset exists, it is loaded from disk.
        Otherwise, the dataset is built by preprocessing audio files.

        Parameters
        ----------
        root_path : Path
            Root directory containing genre subfolders
        preprocessor : Preprocessor
            Reference to a Preprocessor for feature extraction.
        """
        ...

    def get_classes(self) -> dict[int, str]:
        """
        Get the mapping from tensor indices to class names.

        Returns
        -------
        dict[int, str]
            Class dictionary.
        """
        ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single data/target example by index.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Pair of (data, target).
        """
        ...

    def __len__(self) -> int:
        """
        Get the dataset size.

        Returns
        -------
        int
            Number of samples.
        """
        ...

class AudioSubset():
    """
    AudioSubset

    Subset wrapper around an AudioDataset with selected indices.

    Provides a view into a parent dataset for training/validation splits.
    """
    def __init__(self, dataset: AudioDataset, indices: list[int]) -> None:
        """
        Construct a new AudioSubset object.

        Parameters
        ----------
        dataset : AudioDataset
            Reference to the parent dataset.
        indices : list[int]
            Indices of samples to include in the subset.
        """
        ...

    def get_stacked_data(self) -> torch.Tensor:
        """
        Get all subset data stacked into a single tensor.

        Returns
        -------
        torch.Tensor
            Concatenated data tensor.
        """
        ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single example by subset index.

        Parameters
        ----------
        index : int
            Subset-relative index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Pair of (data, target).
        """
        ...

    def __len__(self) -> int:
        """
        Get the subset size.

        Returns
        -------
        int
            Number of samples in subset.
        """
        ...

class MusicModel():
    """
    MusicModel

    Convolutional neural network for music genre classification.

    This class implements the network architecture using
    convolutional, batch normalization, pooling, and linear layers.
    It inherits from torch.nn.Module and defines the forward pass.
    In addition to training and inference, the model provides methods
    to persist its parameters to disk and reload them later.
    """
    def __init__(self) -> None:
        """
        Construct a new MusicModelImpl object.

        Initializes all layers of the model.
        """
        ...

    def parameters(self) -> list[torch.Tensor]:
        """
        Return the list of learnable parameters of the model.

        Returns
        -------
        list[torch.Tensor]
            A list of tensors containing the model's parameters
            (weights and biases) that are subject to optimization.
        """
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Invoke the model on an input tensor.

        This is equivalent to calling ``forward(x)`` and is the standard
        PyTorch interface for passing data through a module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W) or (C, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor containing class scores.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        If the input is a batch, it should have shape (N, C, H, W).
        If a single sample is provided with shape (C, H, W),
        the function will automatically add a batch dimension (unsqueeze at dim=0).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W) or (C, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor containing class scores.
        """
        ...

    def to(self, device: str) -> "MusicModel":
        """
        Move the model to the specified device.

        Transfers all parameters and buffers of the model to the
        given device (e.g., CPU or GPU).

        Parameters
        ----------
        device : str
            Target device to move the model to.

        Returns
        -------
        MusicModel
            The model itself, allowing chained calls.
        """
        ...

    def train(self, on: bool = True) -> "MusicModel":
        """
        Set the model in training mode.

        Parameters
        ----------
        on : bool, optional
            If true, enables training mode,
            if false, switches to evaluation mode, by default True.

        Returns
        -------
        MusicModel
            The model itself, allowing chained calls.
        """
        ...

    def eval(self) -> "MusicModel":
        """
        Set the model in evaluation mode.

        Returns
        -------
        MusicModel
            The model itself, allowing chained calls.
        """
        ...

    def is_training(self) -> bool:
        """
        Check whether the model is currently in training mode.

        Returns
        -------
        bool
            True if the model is in training mode,
            False otherwise.
        """
        ...

    def save(self, file_path: Path) -> None:
        """
        Save the model to the specified file.

        Parameters
        ----------
        file_path : Path
            Destination file path for the serialized model.
        """
        ...

    def load(self, file_path: Path) -> None:
        """
        Load a MusicModel from a file.

        Parameters
        ----------
        file_path : Path
            Source file path of the serialized model.

        Raises
        ------
        invalid_argument
            If model file does not exist.
        """
        ...

class ReduceLROnPlateau():
    """
    ReduceLROnPlateau

    Learning rate scheduler that reduces the LR when a metric has stopped improving.

    This scheduler monitors a validation metric and reduces the optimizer's learning rate
    when no improvement has been seen for a number of epochs (patience).
    It supports both minimization (e.g., loss) and maximization (e.g., accuracy) modes.
    """
    def __init__(self, mode: str, factor: float, patience: int) -> None:
        """
        Construct a new ReduceLROnPlateau scheduler.

        Parameters
        ----------
        mode : str
            Either "min" (lower metric is better) or "max" (higher metric is better).
        factor : float
            Multiplicative factor for reducing the learning rate.
        patience : int
            Number of epochs with no improvement after which LR will be reduced.

        Raises
        ------
        invalid_argument
            If the file is invalid or corrupted.
        """
        ...

    def update_lr(self, metric: float) -> None:
        """
        Perform a scheduler step based on the given metric.

        If the metric improves according to the mode, the scheduler resets.
        Otherwise, after `patience` epochs without improvement, the learning rate
        is multiplied by `factor` for all optimizer parameter groups.

        Parameters
        ----------
        metric : float
            The current value of the monitored metric.

        Raises
        ------
        runtime_error
            If called before attaching an optimizer.
        """
        ...

class OptimizerType(Enum):
    """
    Enumeration of supported optimizer types.

    Attributes
    ----------
    Adam : torch::optim::Adam
        Adam optimizer.
    AdamW : torch::optim::AdamW
        AdamW optimizer.
    RMSprop : torch::optim::RMSprop
        RMSprop optimizer.
    SGD : torch::optim::SGD
        Stochastic Gradient Descent optimizer.
    """
    Adam = ...
    AdamW = ...
    RMSprop = ...
    SGD = ...

class OptimizerConfig():
    """
    OptimizerConfig

    Configuration parameters for optimizer creation.

    This struct defines the hyperparameters required to
    construct a torch optimizer.
    """
    def __init__(
        self,
        lr: float,
        momentum: float = ...,
        alpha: float = ...,
        eps: float = ...,
        decay: float = ...,
        nesterov: bool = ...,
        amsgrad: bool = ...
    ) -> None:
        """
        Construct a new PreprocessorConfig with optional overrides.

        Parameters
        ----------
        lr : float
            Learning rate.
        momentum : float, optional
            Momentum factor, by default 0.0.
        alpha : float, optional
            Alpha parameter for RMSprop, by default 0.99.
        eps : float, optional
            Epsilon for numerical stability, by default 1e-8.
        decay : float, optional
            Weight decay factor (L2 regularization), by default 0.0.
        nesterov : bool, optional
            Enable Nesterov momentum for SGD, by default False.
        amsgrad : bool, optional
            Enable AMSGrad variant for Adam/AdamW, by default False.
        """
        ...

class DeviceManager():
    """
    Device

    Provides a globally accessible device type.

    The Device class encapsulates device selection logic for the project.
    It automatically checks if CUDA is available and selects a GPU device;
    otherwise, it falls back to CPU.
    """

    @staticmethod
    def get() -> str:
        """
        Get the device.

        Determines the appropriate device ("cuda" if available, otherwise "cpu").

        Returns
        -------
        str
            Device type.
        """
        ...

class Trainer():
    """
    Trainer

    Handles training and evaluation of the music classification model.

    The Trainer class manages the optimization process, including
    forward/backward passes, loss calculation and parameter updates.
    """
    def __init__(self, model: MusicModel, type: OptimizerType, cfg: OptimizerConfig) -> None:
        """
        Construct a new Trainer object.

        Initializes the model, optimizer, and loss function.
        Automatically selects CUDA if available, otherwise uses CPU.

        Parameters
        ----------
        model : MusicModel
            Reference to the music model to be trained.
        type : OptimizerType
            Optimizer type (e.g., Adam, SGD).
        cfg : OptimizerConfig
            Configuration parameters for the optimizer.
        """
        ...

    def attach_scheduler(self, scheduler: ReduceLROnPlateau) -> None:
        """
        Attach a learning rate scheduler.

        Links an external scheduler to the optimizer so that learning
        rate can be dynamically adjusted during training.

        Parameters
        ----------
        scheduler : ReduceLROnPlateau
            Reference to a ReduceLROnPlateau scheduler.
        """
        ...

    def train_model(self, subset: AudioSubset, batch_size: int, num_workers: int) -> tuple[float, float]:
        """
        Train the model for one epoch.

        Iterates over batches from the provided dataloader, performs forward
        and backward passes, updates model parameters, and computes average
        loss and accuracy across all samples.

        Parameters
        ----------
        subset : AudioSubset
            Subset providing training samples.
        batch_size : int
            Number of samples per batch.
        num_workers : int
            Number of workers for parallelization.

        Returns
        -------
        tuple[float, float]
            Tuple of average loss over all batches and
            Average accuracy over all samples.
        """
        ...

    def eval_model(self, subset: AudioSubset, batch_size: int, num_workers: int) -> tuple[float, float]:
        """
        Evaluate the model without gradient updates.

        Iterates over batches from the provided dataloader in evaluation mode,
        computes loss and accuracy, but does not update model parameters.

        Parameters
        ----------
        subset : AudioSubset
            Subset providing validation samples.
        batch_size : int
            Number of samples per batch.
        num_workers : int
            Number of workers for parallelization.

        Returns
        -------
        tuple[float, float]
            Tuple of average loss over all batches and
            Average accuracy over all samples.
        """
        ...