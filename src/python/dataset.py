from pathlib import Path
import torch
from torch.utils.data import Dataset
from preprocessor import Preprocessor

class AudioDataset(Dataset):
    def __init__(self, root_path: Path, preprocessor: Preprocessor):
        super().__init__()

        self.__preprocessor = preprocessor
        self.__files = []
        self.__labels = []
        self.__classes = {}

        # Iterate through the dataset.
        root_path = Path(root_path)
        for dir_path in sorted(root_path.iterdir()):
            # Add new class if it does not exist.
            genre = dir_path.name
            if genre not in self.__labels:
                self.__classes[genre] = len(self.__classes)

            # Iterate through one class and collect all files.
            for file_path in dir_path.iterdir():
                if file_path.suffix == ".wav":
                    self.__files.append(file_path)
                    self.__labels.append(genre)

    def __len__(self) -> int:
        return len(self.__files)

    def __getitem__(self, index: int) -> dict:
        # Retreive the audio file path and the label index.
        file_path = self.__files[index]
        label = self.__classes[self.__labels[index]]

        # Extract features.
        spectrogram, mfcc = self.__preprocessor.run(file_path)

        # Convert flatten features to tensors.
        spectrogram = torch.tensor(spectrogram, dtype=torch.float).unsqueeze(0)
        mfcc = torch.tensor(mfcc, dtype=torch.float).unsqueeze(0)

        # Combine data and convert the label.
        data = torch.cat((spectrogram, mfcc), dim=2)
        target = torch.tensor(label, dtype=torch.long)

        # Compute single sample.
        sample = {
            "data": data,
            "target": target
        }
        return sample

    @property
    def classes(self) -> dict:
        return self.__classes
