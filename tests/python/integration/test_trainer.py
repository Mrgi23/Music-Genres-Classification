from preprocessor import Preprocessor
from dataset import AudioDataset
from model import MusicModel
from trainer import Trainer

import pytest
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Define the test object.
@pytest.fixture
def trainer():
    model = MusicModel()
    return Trainer(model, Adam(model.parameters(), lr=1e-3))

def test_trainer_train_model(trainer):
    # Create the DataLoader.
    audio_dataset = AudioDataset("./resources", Preprocessor())
    size = 2
    audio_dataset, _ = random_split(audio_dataset, [size, len(audio_dataset) - size])
    dataloader = DataLoader(audio_dataset, batch_size=size)

    # Compute the result.
    avg_loss, _ = trainer.train_model(dataloader)

    # Test the result.
    assert(avg_loss > 0), "Invalid train loss value."

def test_trainer_eval_model(trainer):
    # Create the DataLoader.
    audio_dataset = AudioDataset("./resources", Preprocessor())
    size = 2
    audio_dataset, _ = random_split(audio_dataset, [size, len(audio_dataset) - size])
    dataloader = DataLoader(audio_dataset, batch_size=size)

    # Compute the result.
    avg_loss, _ = trainer.eval_model(dataloader)

    # Test the result.
    assert(avg_loss > 0), "Invalid validation loss value."
