import librosa as lib
import numpy as np
from pathlib import Path
from scipy.signal.windows import hann
import soundfile as sf

class Preprocessor():
    def __init__(
        self,
        size: int = 660000,
        n_fft: int = 1024,
        hop: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 13
    ) -> None:
        self.__size = size
        self.__n_fft = n_fft
        self.__hop = hop
        self.__n_mels = n_mels
        self.__n_mfcc = n_mfcc

        # Compute window.
        self.__window = hann(self.__n_fft, False)

    def __load_and_crop(
        self,
        file_path: Path
    ) -> tuple[np.ndarray, float]:
        try:
            # Load audio signal from the file.
            audio_signal, sr = sf.read(file_path)
        except:
            raise FileNotFoundError(f"Preprocessor.__load_and_crop: Invalid or corrupted file.")

        # Crop the audio.
        audio_signal = audio_signal[:self.__size]
        return audio_signal, sr

    def run(
        self,
        file_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        # Load and crop audio signal.
        audio_signal, sr = self.__load_and_crop(file_path)

        # Compute Short-Term Fourier Transform.
        stft = lib.stft(
            y=audio_signal,
            n_fft=self.__n_fft,
            hop_length=self.__hop,
            window=self.__window
        )

        # Compute spectrum and apply correction.
        stft = np.sqrt(np.abs(stft))

        # Calculate power spectrum.
        power = stft ** 2

        # Compute spectrogram.
        spectrogram = (lib.power_to_db(power, ref=1.0)).T

        # Compute mel spectrogram
        mel_spectrogram = lib.feature.melspectrogram(
            S=power, sr=sr, n_fft=self.__n_fft, hop_length=self.__hop, n_mels=self.__n_mels
        )

        # Compute MFCC.
        mfcc = (lib.feature.mfcc(S=np.log10(mel_spectrogram), n_mfcc=self.__n_mfcc)).T
        return spectrogram, mfcc
