import pytest
from downloader import Downloader
from preprocessor import Preprocessor

# Define the test object.
@pytest.fixture
def preprocessor():
    return Preprocessor()

def test_preprocessor_valid_output(preprocessor):
    # Define the input and expected result.
    file_path = "./dataset/jazz/jazz.00001.wav"
    n_frames_expected = 1 + int(preprocessor._Preprocessor__size / preprocessor._Preprocessor__hop)
    spec_size_expected = preprocessor._Preprocessor__n_fft // 2 + 1
    mfcc_size_expected = preprocessor._Preprocessor__n_mfcc

    # Compute the result.
    Downloader().run()
    spectrogram, mfcc = preprocessor.run(file_path)

    # Test the result.
    assert(spectrogram.shape == (n_frames_expected, spec_size_expected)), "Invalid size of the spectrogram."
    assert(mfcc.shape == (n_frames_expected, mfcc_size_expected)), "Invalid size of the MFCC."

def test_preprocessor_invalid_input(preprocessor):
    with pytest.raises(FileNotFoundError, match="Preprocessor.__load_and_crop: Invalid or corrupted file."):
        preprocessor.run("./dataset/jazz/jazz.00054.txt")
