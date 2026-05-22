from dataset import AudioDataset
import pytest
import torch

@pytest.fixture(params=[True, False])
def preload(request):
  return request.param

@pytest.fixture
def mock_dataset(tmp_path):
  def _make_dataset(preload: bool):
    root_path = tmp_path / "resources"
    root_path.mkdir(parents=True, exist_ok=True)

    if not preload:
      genres = ["jazz", "rock"]
      for genre in genres:
        dir_path = root_path / genre
        dir_path.mkdir(parents=True, exist_ok=True)

        for i in range(3):
          file_path = dir_path / f"{genre}.0000{i}.wav"
          file_path.write_bytes(b"Mock...Data")
    else:
      dataset_path = root_path / "dataset_py.pt"
      dataset = (
        [torch.rand((1290, 13)) for _ in range(6)],
        [torch.rand(1) for _ in range(6)],
        {0: "jazz", 1: "rock"}
      )
      torch.save(dataset, dataset_path)
    return root_path
  return _make_dataset

@pytest.fixture
def audio_dataset_with_params(preload, mock_dataset, mocker):
  root_path = mock_dataset(preload=preload)

  mock_preprocessor = mocker.Mock()
  mock_preprocessor.process_file.side_effect = lambda _: torch.rand((1290, 13))
  mock_preprocessor.normalize_data.side_effect = lambda _: torch.rand((1290, 13))
  return AudioDataset(root_path, mock_preprocessor)

def test_dataset_getitem(audio_dataset_with_params):
  num_frames = 1290
  mfcc_size = 13

  data, target = audio_dataset_with_params[0]
  assert(data.numel() == num_frames * mfcc_size), "Invalid size of the samle data."
  assert(target.numel() == 1), "Invalid size of the sample target."

@pytest.fixture
def audio_dataset(mock_dataset):
  root_path = mock_dataset(preload=True)
  return AudioDataset(root_path, None)

def test_dataset_classes(audio_dataset):
  classes_expected = {0: "jazz", 1: "rock"}
  classes = audio_dataset.classes
  assert(classes == classes_expected), "Invalid dataset classes."

def test_dataset_data(audio_dataset):
  size_expected = 6
  num_frames = 1290
  mfcc_size = 13

  size = len(audio_dataset.data)
  data = audio_dataset.data[0]
  assert(size == size_expected), "Invalid size of the dataset."
  assert(data.numel() == num_frames * mfcc_size), "Invalid size of the samle data."

def test_dataset_len(audio_dataset):
  size_expected = 6
  size = len(audio_dataset)
  assert(size == size_expected), "Invalid size of the dataset."
