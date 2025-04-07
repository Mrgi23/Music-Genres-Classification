import numpy as np
import pytest
import torch
from dataset import AudioDataset

@pytest.fixture
def mock_dataset(tmp_path):
    # Create temporary root dir.
    root_path = tmp_path / "dataset"
    root_path.mkdir(parents=True, exist_ok=True)

    # Create temporary genres dirs.
    genres = ["jazz", "rock"]
    for genre in genres:
        dir_path = root_path / genre
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create 3 temporary .wav files
        for i in range(3):
            file_path = dir_path / f"{genre}.0000{i}.wav"
            file_path.write_bytes(b"Mock...Data")
    return root_path

# Define the test object.
@pytest.fixture
def audio_dataset(mock_dataset, mocker):
    mock_preprocessor = mocker.Mock()
    return AudioDataset(mock_dataset, mock_preprocessor)

def test_dataset_getitem_valid_output(audio_dataset):
    # Define the expected result.
    type_expected = torch.Tensor

    # Mock the Preprocessor
    num_frames = 1290
    mfcc_size = 13
    mock_mfcc = np.random.random((num_frames, mfcc_size))
    audio_dataset._AudioDataset__preprocessor.run.side_effect = lambda _: mock_mfcc

    # Compute the result.
    data, target = audio_dataset[0]

    # Test the result.
    assert(isinstance(data, type_expected)), "Sample data must be Tensor."
    assert(data.numel() == num_frames * mfcc_size), "Invalid size of the samle data."
    assert(isinstance(target, type_expected)), "Sample target must be Tensor."
    assert(target.numel() == 1), "Invalid size of the sample target."


def test_dataset_len_valid_output(audio_dataset):
    # Define the expected result.
    size_expected = 6

    # Compute the result.
    size = len(audio_dataset)

    # Test the result.
    assert(size == size_expected), "Invalid size of the dataset."

def test_dataset_classes_valid_output(audio_dataset):
    # Define the expected result.
    classes_expected = {
        "jazz": 0,
        "rock": 1
    }

    # Compute the result.
    classes = audio_dataset.classes

    assert(classes == classes_expected), "Invalid dataset classes."
