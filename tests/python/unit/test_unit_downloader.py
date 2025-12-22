from downloader import Downloader

import pytest
import shutil

def test_downloader_download_and_extract():
    # Define the test object.
    shutil.rmtree("./resources", ignore_errors=True)
    downloader = Downloader("./resources")

    # Define the expected result.
    n_folders_expected = 10
    n_files_expected = 999

    # Compute the result.
    downloader.download_and_extract()

    # Test the result.
    n_folders = 0
    n_files = 0
    for entry in downloader.root_path.rglob("*"):
        if entry.is_dir():
            n_folders += 1
        if entry.is_file() and entry.suffix == ".wav":
            n_files += 1
    assert(n_folders == n_folders_expected), "Invalid number of the dataset classes."
    assert(n_files == n_files_expected), "Invalid number of the dataset samples."

def test_downloader_download_and_extract_url_exception(tmp_path):
    # Define the input.
    url = "https://artifacts.mrgi23.com/Music-Genres-Classification/dataset/invalid.tar.zst"
    with pytest.raises(ConnectionRefusedError, match=f"Downloader.download_and_extract: URL: {url} is invalid."):
        Downloader(tmp_path, url).download_and_extract()