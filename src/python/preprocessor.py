from dataclasses import dataclass
import librosa as lib
import numpy as np
from pathlib import Path
from scipy.signal.windows import hann
import soundfile as sf
import torch

@dataclass
class PreprocessorConfig:
    """
    PreprocessorConfig

    Configuration parameters for audio preprocessing.

    Attributes
    ----------
    size : int
        Input signal length, by default 660,000 samples (≈30s at 22.05kHz).
    nfft : int
        FFT size for spectral analysis, by default 1024.
    hop : int
        Hop size between frames, by default 512.
    nmels : int
        Number of mel bands, by default 128.
    nmfcc : int
        Number of MFCC coefficients, by default 13.
    """
    size: int = 660000
    nfft: int = 1024
    hop: int = 512
    nmels: int = 128
    nmfcc: int = 13

class Preprocessor():
    """
    Preprocessor

    Audio preprocessing pipeline for feature extraction and normalization.

    This class uses the Aubio library to extract MFCC features from audio files
    and prepares them as Torch tensors for downstream ML tasks. It also supports
    normalization using precomputed mean and standard deviation tensors.
    """
    def __init__(self, cfg: PreprocessorConfig = PreprocessorConfig()) -> None:
        """
        Construct a new Preprocessor object.

        Parameters
        ----------
        cfg : PreprocessorConfig, optional
            Configuration parameters for preprocessing, by default default config.
        """
        self.__cfg = cfg
        self.__window = hann(cfg.nfft, False)

        self.__mean = torch.zeros((1 + self.__cfg.size // self.__cfg.hop, self.__cfg.nmfcc), dtype=torch.float)
        self.__std = torch.ones((1 + self.__cfg.size // self.__cfg.hop, self.__cfg.nmfcc), dtype=torch.float)

    @property
    def cfg(self) -> PreprocessorConfig:
        """
        Get the preprocessing configuration.

        Returns
        -------
        PreprocessorConfig
            A copy of the configuration.
        """
        return self.__cfg

    @property
    def mean(self) -> torch.Tensor:
        """
        Get the mean tensor used for normalization.

        Returns
        -------
        torch.Tensor
            Mean tensor.
        """
        return self.__mean

    @mean.setter
    def mean(self, mean: torch.Tensor) -> None:
        """
        Set the mean tensor used for normalization.

        Parameters
        ----------
        mean : torch.Tensor
            Mean tensor.
        """
        self.__mean = mean

    @property
    def std(self) -> torch.Tensor:
        """
        Get the standard deviation tensor used for normalization.

        Returns
        -------
        torch.Tensor
            Standard deviation tensor.
        """
        return self.__std

    @std.setter
    def std(self, std: torch.Tensor) -> None:
        """
        Set the standard deviation tensor used for normalization.

        Parameters
        ----------
        std : torch.Tensor
            Standard deviation tensor.
        """
        self.__std = std

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
        audio_signal, sr = self.__load_and_crop(file_path)

        stft = lib.stft(
            y=audio_signal,
            n_fft=self.__cfg.nfft,
            hop_length=self.__cfg.hop,
            window=self.__window
        )

        stft = np.sqrt(np.abs(stft))

        power = stft ** 2

        mel_spectrogram = lib.feature.melspectrogram(
            S=power, sr=sr, n_fft=self.__cfg.nfft, hop_length=self.__cfg.hop, n_mels=self.__cfg.nmels
        )

        mfcc = (lib.feature.mfcc(S=np.log10(mel_spectrogram + 1e-18), n_mfcc=self.__cfg.nmfcc)).T
        mfcc = torch.tensor(mfcc, dtype=torch.float).unsqueeze(0)
        return mfcc

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
        y =  (x - self.__mean) / self.__std
        return y

    def __load_and_crop(self, file_path: Path) -> tuple[np.ndarray, float]:
        """
        Load audio file and crop it to the specified size.

        Parameters
        ----------
        file_path : Path
            Path to the audio file.

        Returns
        -------
        tuple[np.ndarray, float]
            Tuple containing the waveform samples and sample rate.

        Raises
        ------
        FileNotFoundError
            If the file is invalid or corrupted.
        """
        try:
            audio_signal, sr = sf.read(file_path)
        except:
            raise FileNotFoundError(f"Preprocessor.__load_and_crop: File: {file_path} is invalid or corrupt.")

        audio_signal = audio_signal[:self.__cfg.size]
        return audio_signal, sr
