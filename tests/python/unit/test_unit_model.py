import pytest
import torch
from model import MusicModel

def test_model():
    # Define the test object.
    model = MusicModel()

    # Define the input and expected result.
    batch_size = 16
    num_frames = 1290
    mfcc_size = 13
    x = torch.rand((batch_size, num_frames, mfcc_size))
    num_classes_expected = 10

    # Compute the result.
    model.eval()
    with torch.no_grad():
        y = model(x)

    # Test the result.
    assert(y.numel() == batch_size * num_classes_expected), "Invalid size of the model output."
