from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.functional import relu

class MusicModel(nn.Module):
  """
  MusicModel

  Convolutional neural network for music genre classification.

  This class implements the network architecture using
  convolutional, batch normalization, pooling, and linear layers.
  It inherits from torch.nn.Module and defines the forward pass.
  In addition to training and inference, the model provides methods
  to persist its parameters to disk and reload them later.
  """
  def __init__(self) -> None:
    """
    Construct a new MusicModelImpl object.

    Initializes all layers of the model.
    """
    super(MusicModel, self).__init__()

    self._conv1 = nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1, bias=False)
    self._bn1 = nn.BatchNorm2d(512)
    self._conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
    self._bn2 = nn.BatchNorm2d(256)
    self._conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
    self._bn3 = nn.BatchNorm2d(128)

    self._max_pool = nn.MaxPool2d(2, 2)
    self._adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    self._flat = nn.Flatten()
    self._linear = nn.Linear(128, 64)
    self._dropout = nn.Dropout(0.3)
    self._output = nn.Linear(64, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the network.

    If the input is a batch, it should have shape (N, C, H, W).
    If a single sample is provided with shape (C, H, W),
    the function will automatically add a batch dimension (unsqueeze at dim=0).

    Parameters
    ----------
    x : torch.Tensor
      Input tensor of shape (N, C, H, W) or (C, H, W)

    Returns
    -------
    torch.Tensor
      Output tensor containing class scores.
    """
    if x.dim() == 3:
      x = x.unsqueeze(1)

    x = self._max_pool(relu(self._bn1(self._conv1(x))))
    x = self._max_pool(relu(self._bn2(self._conv2(x))))
    x = self._max_pool(relu(self._bn3(self._conv3(x))))

    x = self._adaptive_pool(x)
    x = self._flat(x)

    x = relu(self._linear(x))
    x = self._dropout(x)
    x = self._output(x)
    return x

  def save(self, file_path: Path) -> None:
    """
    Save the model to the specified file.

    Parameters
    ----------
    file_path : Path
      Destination file path for the state dictionary.
    """
    device = next(self.parameters()).device
    self.to("cpu")
    torch.save(self.state_dict(), file_path)
    self.to(device)

  def load(self, file_path: Path) -> None:
    """
    Load a MusicModel from a file.

    Parameters
    ----------
    file_path : Path
      Source file path of the state dictionary.

    Raises
    ------
    FileNotFoundError
      If model file does not exist.
    """
    if file_path.exists():
      device = next(self.parameters()).device
      state_dict = torch.load(file_path)
      self.load_state_dict(state_dict)
      self.to(device)
    else:
      raise FileNotFoundError(f"MusicModel.load: File: {file_path} does not exist.")
