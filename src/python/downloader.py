from pathlib import Path
import requests
import shutil
import zipfile

class Downloader():
    def __init__(self,
        root: Path = Path("./resources"),
        url: str = "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification"
    ) -> None:
        self.__root = Path(root)
        self.__url = url
        self.__zip_file = self.__root / "dataset.zip"

    def run(self) -> None:
        # Download and extract dataset if it does not exists.
        if not self.__exists():
            # Download dataset.
            Downloader.download(self.__url, self.__zip_file)

            # Extract dataset.
            self.__extract()

    @property
    def root(self) -> Path:
        return self.__root

    @staticmethod
    def download(url: str, file_path: Path) -> None:
        # Create file folder.
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the dataset from the URL.
        with requests.get(url, stream=True) as req:
            req.raise_for_status()

            # Store downloaded dataset in the zip file.
            with open(file_path, "wb") as file:
                for chunk in req.iter_content(chunk_size=8192):
                    file.write(chunk)

    def __exists(self) -> bool:
        # Check if .wav files exist.
        return any(file for file in self.__root.rglob("*.wav"))

    def __extract(self):
        # Open the zip file.
        with zipfile.ZipFile(self.__zip_file, "r") as zip_file:
            for member in zip_file.namelist():
                # Select only .wav files.
                if member.startswith("Data/genres_original/") and member.endswith(".wav"):
                    # Define the .wav file relative path.
                    relative_path = Path(member).relative_to("Data/genres_original")

                    # Compute the absoulte save path.
                    save_path = self.__root / relative_path
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy the .wav file and store it in the dataset folder, keeping the tree intact.
                    with zip_file.open(member) as source, open(save_path, "wb") as destination:
                        shutil.copyfileobj(source, destination)

        # Remove the zip file.
        self.__zip_file.unlink()

        # Remove single corrupted file.
        corrupted_file = self.__root / "jazz" / "jazz.00054.wav"
        corrupted_file.unlink()