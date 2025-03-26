import pytest
import torch
from downloader import Downloader
from preprocessor import Preprocessor
from dataset import AudioDataset

# Define the test object.
@pytest.fixture
def audio_dataset():
    Downloader().run()
    preprocessor = Preprocessor()
    return AudioDataset("./resources", preprocessor)

def test_dataset_getitem_valid_output(audio_dataset):
    # Collect the parameters.
    size = audio_dataset.preprocessor._Preprocessor__size
    hop = audio_dataset.preprocessor._Preprocessor__hop
    n_mfcc = audio_dataset.preprocessor._Preprocessor__n_mfcc

    # Define the expected result.
    num_frames = size // hop + 1
    mfcc_size = n_mfcc

    # Compute the result.
    data, target = audio_dataset[0]

    # Test the result.
    assert(data.numel() == num_frames * mfcc_size), "Invalid size of the samle data."
    assert(target.numel() == 1), "Invalid size of the sample target."

def test_dataset_len_valid_output(audio_dataset):
    # Define the expected result.
    size_expected = 999

    # Compute the result.
    size = len(audio_dataset)

    # Test the result.
    assert(size == size_expected), "Invalid size of the dataset."

def test_dataset_classes_valid_output(audio_dataset):
    # Define the expected result.
    classes_expected = {
        "blues": torch.tensor(0, dtype=torch.long),
        "classical": torch.tensor(1, dtype=torch.long),
        "country": torch.tensor(2, dtype=torch.long),
        "disco": torch.tensor(3, dtype=torch.long),
        "hiphop": torch.tensor(4, dtype=torch.long),
        "jazz": torch.tensor(5, dtype=torch.long),
        "metal": torch.tensor(6, dtype=torch.long),
        "pop": torch.tensor(7, dtype=torch.long),
        "reggae": torch.tensor(8, dtype=torch.long),
        "rock": torch.tensor(9, dtype=torch.long)
    }

    # Compute the result.
    classes = audio_dataset.classes

    # Test the result.
    assert(classes == classes_expected), "Invalid dataset classes."
