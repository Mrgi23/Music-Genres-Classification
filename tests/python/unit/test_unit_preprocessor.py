import pytest
import torch
from preprocessor import Preprocessor

# Define the test object.
@pytest.fixture
def preprocessor():
    return Preprocessor()

def test_preprocessor_run_valid_output(preprocessor):
    # Define the input and expected result.
    file_path = "./resources/jazz/jazz.00001.wav"
    num_frames_expected = 1 + int(preprocessor._Preprocessor__size / preprocessor._Preprocessor__hop)
    mfcc_size_expected = preprocessor._Preprocessor__n_mfcc

    # Compute the result.
    mfcc = preprocessor.run(file_path)

    # Test the result.
    assert(mfcc.numel() == num_frames_expected * mfcc_size_expected), "Invalid size of the MFCC."

def test_preprocessor_run_invalid_input(preprocessor):
    with pytest.raises(FileNotFoundError, match="Preprocessor.__load_and_crop: Invalid or corrupted file."):
        preprocessor.run("./resources/jazz/jazz.00054.txt")

def test_preprocessor_normalize_valid_output(preprocessor):
    # Define the input and expected result.
    num_frames = 1 + int(preprocessor._Preprocessor__size / preprocessor._Preprocessor__hop)
    mfcc_size = preprocessor._Preprocessor__n_mfcc
    x = 2 * torch.ones((num_frames, mfcc_size))
    mean_expected = torch.ones((num_frames, mfcc_size))
    std_expected = 2 * torch.ones((num_frames, mfcc_size))
    y_expected = 0.5 * torch.ones((num_frames, mfcc_size))

    # Compute the result.
    preprocessor.mean = mean_expected
    preprocessor.std = std_expected
    y = preprocessor.normalize(x)

    # Test the result.
    assert(torch.equal(preprocessor.mean, mean_expected)), "Invalid mean value."
    assert(torch.equal(preprocessor.std, std_expected)), "Invalid standard deviation value."
    assert(torch.equal(y, y_expected)), "Invalid normalized signal."
