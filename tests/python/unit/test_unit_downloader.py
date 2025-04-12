import shutil
from downloader import Downloader

def test_downloader():
    # Define the test object.
    shutil.rmtree("./resources")
    downloader = Downloader("./resources")

    # Define the expected result.
    n_folders_expected = 10
    n_files_expected = 999

    # Compute the result.
    root = downloader.root
    downloader.run()

    # Test the result.
    n_folders = 0
    n_files = 0
    for entry in root.rglob("*"):
        if entry.is_dir():
            n_folders += 1
        if entry.is_file() and entry.suffix == ".wav":
            n_files += 1
    assert(n_folders == n_folders_expected), "Invalid number of the dataset classes."
    assert(n_files == n_files_expected), "Invalid number of the dataset samples."
