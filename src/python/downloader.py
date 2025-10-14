from pathlib import Path
import requests
import shutil
import zipfile

class Downloader():
    """
    Downloader

    Utility class to download and extract the GTZAN dataset (or any dataset) from a URL.

    The Downloader handles downloading a dataset archive from Kaggle (or other sources),
    saving it locally, and extracting it into the specified root directory.
    It also provides helper methods for file existence checks and writing to disk.
    """
    def __init__(
        self,
        root_path: Path,
        url: str = "https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification"
    ) -> None:
        """
        Construct a new Downloader object.

        Parameters
        ----------
        root_path : Path
            Root path where the dataset will be stored.
        url : str, optional
            URL from which the dataset should be downloaded, by default the GTZAN dataset Kaggle API endpoint.
        """
        self.__root_path = Path(root_path)
        self.__url = url
        self.__zip_file = self.__root_path / "dataset.zip"

    @staticmethod
    def download_from_url(file_path: Path, url: str) -> None:
        """
        Download a file from a given URL and save it to the given path.

        ----------
        file_path : Path
            Destination path where the file will be saved.
        url : str
            The URL to download the file from.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True) as req:
            req.raise_for_status()

            with open(file_path, "wb") as file:
                for chunk in req.iter_content(chunk_size=8192):
                    file.write(chunk)

    @property
    def root_path(self) -> Path:
        """
        Get the root path where the dataset will be extracted.

        Returns
        -------
        Path
            The root path.
        """
        return self.__root_path

    def download_and_extract(self) -> None:
        """
        Download the dataset and extract it into the root directory.
        This method checks if the dataset already exists, downloads it if missing,
        and extracts the archive into the root path.
        """
        if not self.__is_existing():
            Downloader.download_from_url(self.__zip_file, self.__url)

            self.__extract_file()

    def __is_existing(self) -> bool:
        """
        Check whether the dataset already exists at the root path.

        Returns
        -------
        bool
            True if the dataset already exists,
            False if it does not exist and needs to be downloaded.
        """
        return any(file for file in self.__root_path.rglob("*.wav"))

    def __extract_file(self):
        """
        Extract the downloaded archive into the root path.
        """
        with zipfile.ZipFile(self.__zip_file, "r") as zip_file:
            for member in zip_file.namelist():
                if member.startswith("Data/genres_original/") and member.endswith(".wav"):
                    relative_path = Path(member).relative_to("Data/genres_original")
                    save_path = self.__root_path / relative_path
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    with zip_file.open(member) as source, open(save_path, "wb") as destination:
                        shutil.copyfileobj(source, destination)

        self.__zip_file.unlink()

        corrupted_file = self.__root_path / "jazz" / "jazz.00054.wav"
        corrupted_file.unlink()
