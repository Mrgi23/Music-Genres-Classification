import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from trainer import Trainer

# Define the test object.
@pytest.fixture
def trainer(mocker):
    # Mock the MusicModel
    mock_model = mocker.Mock(side_effect=lambda x: torch.zeros((x.size(0), 10), requires_grad=True))
    mock_model.parameters.side_effect = lambda: [torch.nn.Parameter(torch.zeros(1))]
    return Trainer(mock_model, Adam(mock_model.parameters(), lr=1e-3))

def test_trainer_fit_valid_output(trainer, mocker):
    # Define the expected result.
    avg_loss_expected = np.log(10)

    # Mock the DataLoader
    mock_dataset = mocker.MagicMock()
    mock_dataset.__getitem__.side_effect = lambda _: (torch.zeros((1290, 13)), torch.tensor(0, dtype=torch.long))
    mock_dataset.__len__.side_effect = lambda: 6
    mock_dataloader = DataLoader(mock_dataset, len(mock_dataset))

    # Compute the result.
    avg_loss, _ = trainer.fit(mock_dataloader)

    # Test the result.
    assert(np.isclose(avg_loss, avg_loss_expected, atol=1e-6)), "Invalid train loss value."

def test_trainer_eval_valid_output(trainer, mocker):
    # Define the expected result.
    avg_loss_expected = np.log(10)

    # Mock the DataLoader
    mock_dataset = mocker.MagicMock()
    mock_dataset.__getitem__.side_effect = lambda _: (torch.zeros((1290, 13)), torch.tensor(0, dtype=torch.long))
    mock_dataset.__len__.side_effect = lambda: 6
    mock_dataloader = DataLoader(mock_dataset, len(mock_dataset))

    # Compute the result.
    avg_loss, _ = trainer.eval(mock_dataloader)

    # Test the result.
    assert(np.isclose(avg_loss, avg_loss_expected, atol=1e-6)), "Invalid validation loss value."
