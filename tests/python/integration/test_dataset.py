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
    return AudioDataset("./dataset", preprocessor)

def test_dataset_getitem_valid_output(audio_dataset):
    # Collect the parameters.
    size = audio_dataset._AudioDataset__preprocessor._Preprocessor__size
    n_fft = audio_dataset._AudioDataset__preprocessor._Preprocessor__n_fft
    hop = audio_dataset._AudioDataset__preprocessor._Preprocessor__hop
    n_mfcc = audio_dataset._AudioDataset__preprocessor._Preprocessor__n_mfcc

    # Define the expected result.
    size_expected = 2
    type_expected = torch.Tensor
    num_frames = size // hop + 1
    spec_size = n_fft // 2 + 1
    mfcc_size = n_mfcc

    # Compute the result.
    sample = audio_dataset[0]

    # Test the result.
    assert(len(sample) == size_expected), "Invalid number of sample element."
    for key in ["data", "target"]:
        assert(sample.get(key) != None), "Missing sample element."
        assert(isinstance(sample[key], type_expected)), "Sample element must be Tensor."
    assert(sample["data"].numel() == num_frames * (spec_size + mfcc_size)), "Invalid size of the samle data."
    assert(sample["target"].numel() == 1), "Invalid size of the sample target."


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
        "blues": 0,
        "classical": 1,
        "country": 2,
        "disco": 3,
        "hiphop": 4,
        "jazz": 5,
        "metal": 6,
        "pop": 7,
        "reggae": 8,
        "rock": 9
    }

    # Compute the result.
    classes = audio_dataset.classes

    assert(classes == classes_expected), "Invalid dataset classes."
