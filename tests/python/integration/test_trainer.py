from preprocessor import Preprocessor
from dataset import AudioDataset
from model import MusicModel
from trainer import Trainer

import pytest
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

@pytest.fixture
def trainer():
  model = MusicModel()
  return Trainer(model, Adam(model.parameters(), lr=1e-3))

def test_trainer_train_model(trainer):
  audio_dataset = AudioDataset("./resources", Preprocessor())
  size = 2
  audio_dataset, _ = random_split(audio_dataset, [size, len(audio_dataset) - size])
  dataloader = DataLoader(audio_dataset, batch_size=size)

  avg_loss, _ = trainer.train(dataloader)
  assert(avg_loss > 0), "Invalid train loss value."

def test_trainer_eval_model(trainer):
  audio_dataset = AudioDataset("./resources", Preprocessor())
  size = 2
  audio_dataset, _ = random_split(audio_dataset, [size, len(audio_dataset) - size])
  dataloader = DataLoader(audio_dataset, batch_size=size)

  avg_loss, _ = trainer.eval(dataloader)
  assert(avg_loss > 0), "Invalid validation loss value."
