from pathlib import Path
import torch
from torch.utils.data import Dataset
from preprocessor import Preprocessor

class AudioDataset(Dataset):
    def __init__(self, root_path: Path, preprocessor: Preprocessor):
        super().__init__()

        self.__preprocessor = preprocessor
        root_path = Path(root_path)

        dataset_path = root_path / "dataset_py.pt"

        if dataset_path.exists():
            # Preload existing dataset.
            self.__data, self.__target, self.__classes = torch.load(dataset_path)
        else:
            self.__data = []
            self.__target = []
            self.__classes = {}

            # Iterate through the dataset.
            for dir_path in sorted(root_path.iterdir()):
                if not dir_path.is_dir():
                    continue
                
                # Add new class if it does not exist.
                genre = dir_path.name
                if genre not in self.__classes:
                    self.__classes[genre] = torch.tensor(len(self.__classes), dtype=torch.long)

                # Iterate through one class and collect all files.
                for file_path in dir_path.iterdir():
                    if file_path.suffix == ".wav":
                        self.__data.append(preprocessor.run(file_path))
                        self.__target.append(self.__classes[genre])

            # Save new dataset.
            torch.save((self.__data, self.__target, self.__classes), dataset_path)

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, index: int) -> tuple:
        # Normalize data
        data = self.__preprocessor.normalize(self.__data[index])
        target = self.__target[index]
        return data, target

    @property
    def preprocessor(self) -> Preprocessor:
        return self.__preprocessor

    @property
    def classes(self) -> dict:
        return self.__classes
