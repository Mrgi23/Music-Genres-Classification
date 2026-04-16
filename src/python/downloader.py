from pathlib import Path
import pycurl
import shutil
import tarfile
import zstandard

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
        url: str = "https://s3.mrgi23.com/artifacts/Music-Genres-Classification/dataset/dataset.tar.zst"
    ) -> None:
        """
        Construct a new Downloader object.

        Parameters
        ----------
        root_path : Path
            Root path where the dataset will be stored.
        url : str, optional
            URL from which the dataset should be downloaded, by default the Cloudflare R2 storage API endpoint.
        """
        self.__root_path = Path(root_path)
        self.__url = url
        self.__archive_file = self.__root_path / "dataset.tar.zst"

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

        curl = pycurl.Curl()
        curl.setopt(pycurl.URL, url)
        curl.setopt(pycurl.BUFFERSIZE, 1024 * 1024)
        curl.setopt(pycurl.FAILONERROR, 1)
        curl.setopt(pycurl.FOLLOWLOCATION, 1)
        curl.setopt(pycurl.USERAGENT, "curl/8.0")

        try:
            with open(file_path, "wb") as out:
                curl.setopt(pycurl.WRITEDATA, out)
                curl.perform()
                curl.close()
                return True
        except pycurl.error:
            curl.close()
            return False

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
            if Downloader.download_from_url(self.__archive_file, self.__url):
                self.__extract_file()
            else:
                raise ConnectionRefusedError(f"Downloader.download_and_extract: URL: {self.__url} is invalid.")

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
        chunk_size = 1024 * 1024
        with open(self.__archive_file, "rb") as archive:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(archive, read_size=chunk_size) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        if member.isfile():
                            save_path = self.__root_path / Path(member.name)
                            save_path.parent.mkdir(parents=True, exist_ok=True)

                            source = tar.extractfile(member)
                            if source is not None:
                                with source, open(save_path, "wb", buffering=chunk_size) as destination:
                                    shutil.copyfileobj(source, destination, length=chunk_size)
        self.__archive_file.unlink()