import pytest
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from preprocessor import Preprocessor
from dataset import AudioDataset
from model import MusicModel
from trainer import Trainer

# Define the test object.
@pytest.fixture
def trainer():
    model = MusicModel()
    return Trainer(model, Adam(model.parameters(), lr=1e-3))

def test_trainer_fit_valid_output(trainer):
    # Create the DataLoader.
    audio_dataset = AudioDataset("./resources", Preprocessor())
    size = 2
    audio_dataset, _ = random_split(audio_dataset, [size, len(audio_dataset) - size])
    dataloader = DataLoader(audio_dataset, batch_size=size, num_workers=4)

    # Compute the result.
    avg_loss, _ = trainer.fit(dataloader)

    # Test the result.
    assert(avg_loss > 0), "Invalid train loss value."

def test_trainer_eval_valid_output(trainer):
    # Create the DataLoader.
    audio_dataset = AudioDataset("./resources", Preprocessor())
    size = 2
    audio_dataset, _ = random_split(audio_dataset, [size, len(audio_dataset) - size])
    dataloader = DataLoader(audio_dataset, batch_size=size, num_workers=4)

    # Compute the result.
    avg_loss, _ = trainer.eval(dataloader)

    # Test the result.
    assert(avg_loss > 0), "Invalid validation loss value."
