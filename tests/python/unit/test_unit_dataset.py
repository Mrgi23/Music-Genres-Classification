import pytest
import torch
from dataset import AudioDataset

@pytest.fixture(params=[True, False])
def preload(request):
    return request.param

@pytest.fixture
def mock_dataset(tmp_path, preload):
    # Create temporary root dir.
    root_path = tmp_path / "resources"
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

    if preload:
        dataset_path = root_path / "dataset_py.pt"
        dataset = (
            [torch.rand((1290, 13)) for _ in range(6)],
            [torch.rand(1) for _ in range(6)],
            {"jazz": torch.tensor(0, dtype=torch.long), "rock": torch.tensor(1, dtype=torch.long)}
        )
        torch.save(dataset, dataset_path)
    return root_path

# Define the test object.
@pytest.fixture
def audio_dataset(mock_dataset, mocker):
    # Mock the Preprocessor
    mock_preprocessor = mocker.Mock()
    mock_preprocessor.run.side_effect = lambda _: torch.rand((1290, 13))
    mock_preprocessor.normalize.side_effect = lambda _: torch.rand((1290, 13))
    return AudioDataset(mock_dataset, mock_preprocessor)

def test_dataset_getitem_valid_output(audio_dataset):
    # Define the expected result.
    num_frames = 1290
    mfcc_size = 13

    # Compute the result.
    data, target = audio_dataset[0]

    # Test the result.
    assert(data.numel() == num_frames * mfcc_size), "Invalid size of the samle data."
    assert(target.numel() == 1), "Invalid size of the sample target."

def test_dataset_len_valid_output(audio_dataset):
    # Define the expected result.
    size_expected = 6

    # Compute the result.
    size = len(audio_dataset)

    # Test the result.
    assert(size == size_expected), "Invalid size of the dataset."

def test_dataset_data_valid_output(audio_dataset):
    # Define the expected result.
    size_expected = 6
    num_frames = 1290
    mfcc_size = 13

    # Compute the result.
    size = len(audio_dataset.data)
    data = audio_dataset.data[0]

    # Test the result.
    assert(size == size_expected), "Invalid size of the dataset."
    assert(data.numel() == num_frames * mfcc_size), "Invalid size of the samle data."

def test_dataset_classes_valid_output(audio_dataset):
    # Define the expected result.
    classes_expected = {
        "jazz": torch.tensor(0, dtype=torch.long),
        "rock": torch.tensor(1, dtype=torch.long)
    }

    # Compute the result.
    classes = audio_dataset.classes

    assert(classes == classes_expected), "Invalid dataset classes."
