from preprocessor import Preprocessor
from dataset import AudioDataset

import pytest

# Define the test object.
@pytest.fixture
def audio_dataset():
    preprocessor = Preprocessor()
    return AudioDataset("./resources", preprocessor)

def test_dataset_classes(audio_dataset):
    # Define the expected result.
    classes_expected = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock"
    }

    # Compute the result.
    classes = audio_dataset.classes

    # Test the result.
    assert(classes == classes_expected), "Invalid dataset classes."

def test_dataset_getitem(audio_dataset):
    # Collect the parameters.
    size = audio_dataset.preprocessor.cfg.size
    hop = audio_dataset.preprocessor.cfg.hop
    n_mfcc = audio_dataset.preprocessor.cfg.nmfcc

    # Define the expected result.
    num_frames = size // hop + 1
    mfcc_size = n_mfcc

    # Compute the result.
    data, target = audio_dataset[0]

    # Test the result.
    assert(data.numel() == num_frames * mfcc_size), "Invalid size of the samle data."
    assert(target.numel() == 1), "Invalid size of the sample target."

def test_dataset_len(audio_dataset):
    # Define the expected result.
    size_expected = 999

    # Compute the result.
    size = len(audio_dataset)

    # Test the result.
    assert(size == size_expected), "Invalid size of the dataset."
