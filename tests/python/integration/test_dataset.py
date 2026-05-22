from preprocessor import Preprocessor
from dataset import AudioDataset
import pytest

@pytest.fixture
def audio_dataset():
  preprocessor = Preprocessor()
  return AudioDataset("./resources", preprocessor)

def test_dataset_classes(audio_dataset):
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
  classes = audio_dataset.classes
  assert(classes == classes_expected), "Invalid dataset classes."

def test_dataset_getitem(audio_dataset):
  size = audio_dataset.preprocessor.config.size
  hop = audio_dataset.preprocessor.config.hop
  n_mfcc = audio_dataset.preprocessor.config.nmfcc

  num_frames = size // hop + 1
  mfcc_size = n_mfcc
  data, target = audio_dataset[0]
  assert(data.numel() == num_frames * mfcc_size), "Invalid size of the samle data."
  assert(target.numel() == 1), "Invalid size of the sample target."

def test_dataset_len(audio_dataset):
  size_expected = 999
  size = len(audio_dataset)
  assert(size == size_expected), "Invalid size of the dataset."
