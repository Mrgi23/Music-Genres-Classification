from preprocessor import Preprocessor

import pytest
import torch

# Define the test object.
@pytest.fixture
def preprocessor():
    return Preprocessor()

def test_preprocessor_process_file(preprocessor):
    # Define the input and expected result.
    file_path = "./resources/jazz/jazz.00001.wav"
    num_frames_expected = 1 + int(preprocessor.cfg.size / preprocessor.cfg.hop)
    mfcc_size_expected = preprocessor.cfg.nmfcc

    # Compute the result.
    mfcc = preprocessor.process_file(file_path)

    # Test the result.
    assert(mfcc.numel() == num_frames_expected * mfcc_size_expected), "Invalid size of the MFCC."

def test_preprocessor_process_file_exceltion(preprocessor):
    # Define the input.
    file_path = "./resources/jazz/jazz.00054.txt"
    with pytest.raises(FileNotFoundError, match=f"Preprocessor.__load_and_crop: File: {file_path} is invalid or corrupt."):
        preprocessor.process_file(file_path)

def test_preprocessor_normalize_data(preprocessor):
    # Define the input and expected result.
    num_frames = 1 + int(preprocessor.cfg.size / preprocessor.cfg.hop)
    mfcc_size = preprocessor.cfg.nmfcc
    x = 2 * torch.ones((num_frames, mfcc_size))
    mean_expected = torch.ones((num_frames, mfcc_size))
    std_expected = 2 * torch.ones((num_frames, mfcc_size))
    y_expected = 0.5 * torch.ones((num_frames, mfcc_size))

    # Compute the result.
    preprocessor.mean = mean_expected
    preprocessor.std = std_expected
    y = preprocessor.normalize_data(x)

    # Test the result.
    assert(torch.equal(preprocessor.mean, mean_expected)), "Invalid mean value."
    assert(torch.equal(preprocessor.std, std_expected)), "Invalid standard deviation value."
    assert(torch.equal(y, y_expected)), "Invalid normalized signal."
