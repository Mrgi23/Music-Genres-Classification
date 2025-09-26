from preprocessor import Preprocessor

from joblib import delayed, Parallel
import multiprocessing as mp
from pathlib import Path
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """
    AudioDataset

    Dataset class for loading and preprocessing audio data.

    This dataset loads audio files from a directory, applies preprocessing
    (via Preprocessor), and provides data/target pairs as tuple[torch.Tensor, torch.Tensor].
    Data can be preloaded from a serialized .pt archive to save time.
    """
    def __init__(self, root_path: Path, preprocessor: Preprocessor):
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
        super().__init__()

        self.__preprocessor = preprocessor
        root_path = Path(root_path)

        dataset_path = root_path / "dataset_py.pt"

        if dataset_path.exists():
            self.__data, self.__target, self.__classes = torch.load(dataset_path)
        else:
            self.__data = []
            self.__target = []
            self.__classes = {}

            tasks = []
            for dir_path in sorted(root_path.iterdir()):
                if dir_path.is_dir():
                    genre = dir_path.name
                    class_target = torch.tensor(len(self.__classes), dtype=torch.long)
                    self.__classes[class_target.item()] = genre

                    for file_path in sorted(dir_path.iterdir()):
                        if file_path.suffix == ".wav":
                            tasks.append((file_path, class_target))

            results = Parallel(n_jobs=mp.cpu_count())(
                delayed(lambda f, t: (preprocessor.process_file(f), t))(file_path, target)
                for file_path, target in tasks
            )
            self.__data, self.__target = zip(*results)

            torch.save((self.__data, self.__target, self.__classes), dataset_path)

    @property
    def classes(self) -> dict[int, str]:
        """
        Get the mapping from tensor indices to class names.

        Returns
        -------
        dict[int, str]
            Class dictionary.
        """
        return self.__classes

    @property
    def data(self) -> list[torch.Tensor]:
        """
        Get the list of all data tensors.

        Returns
        -------
        list[torch.Tensor]
            Data list.
        """
        return self.__data

    @property
    def preprocessor(self) -> Preprocessor:
        """
        Get the associated Preprocessor instance.

        Returns
        -------
        Preprocessor
            Preprocessor reference.
        """
        return self.__preprocessor

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
        data = self.__preprocessor.normalize_data(self.__data[index])
        target = self.__target[index]
        return data, target

    def __len__(self) -> int:
        """
        Get the dataset size.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.__data)
