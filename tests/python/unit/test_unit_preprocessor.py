from preprocessor import Preprocessor
import pytest
import torch

@pytest.fixture
def preprocessor():
  return Preprocessor()

def test_preprocessor_process_file(preprocessor):
  file_path = "./resources/jazz/jazz.00001.wav"
  num_frames_expected = 1 + int(preprocessor.config.size / preprocessor.config.hop)
  mfcc_size_expected = preprocessor.config.nmfcc
  mfcc = preprocessor.process_file(file_path)
  assert(mfcc.numel() == num_frames_expected * mfcc_size_expected), "Invalid size of the MFCC."

def test_preprocessor_process_file_exception(preprocessor):
  file_path = "./resources/jazz/jazz.00054.txt"
  with pytest.raises(FileNotFoundError, match=f"Preprocessor.__load_and_crop: File: {file_path} is invalid or corrupt."):
    preprocessor.process_file(file_path)

def test_preprocessor_normalize_data(preprocessor):
  num_frames = 1 + int(preprocessor.config.size / preprocessor.config.hop)
  mfcc_size = preprocessor.config.nmfcc
  x = 2 * torch.ones((num_frames, mfcc_size))
  mean_expected = torch.ones((num_frames, mfcc_size))
  std_expected = 2 * torch.ones((num_frames, mfcc_size))
  y_expected = 0.5 * torch.ones((num_frames, mfcc_size))

  preprocessor.mean = mean_expected
  preprocessor.std = std_expected
  y = preprocessor.normalize_data(x)
  assert(torch.equal(preprocessor.mean, mean_expected)), "Invalid mean value."
  assert(torch.equal(preprocessor.std, std_expected)), "Invalid standard deviation value."
  assert(torch.equal(y, y_expected)), "Invalid normalized signal."
