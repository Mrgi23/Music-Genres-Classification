from model import MusicModel
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class DeviceManager():
  """
  Device

  Provides a globally accessible device type.

  The Device class encapsulates device selection logic for the project.
  It automatically checks if CUDA is available and selects a GPU device;
  otherwise, it falls back to CPU.
  """
  @staticmethod
  def get() -> str:
    """
    Get the device.

    Determines the appropriate device ("cuda" if available, otherwise "cpu").

    Returns
    -------
    str
      Device type.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

class Trainer():
  """
  Trainer

  Handles training and evaluation of the music classification model.

  The Trainer class manages the optimization process, including
  forward/backward passes, loss calculation and parameter updates.
  """
  def __init__(self, model: MusicModel, opt: Optimizer) -> None:
    """
    Construct a new Trainer object.

    Initializes the model, optimizer, and loss function.
    Automatically selects CUDA if available, otherwise uses CPU.

    Parameters
    ----------
    model : MusicModel
      Reference to the music model to be trained.
    opt : torch.optim.Optimizer
      Reference to the optimizer used in training.
    """
    self.__model = model
    self.__opt = opt
    self.__loss_function = torch.nn.CrossEntropyLoss()

    self.__model.to(DeviceManager.get())

  def train(self, dataloader: DataLoader) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Iterates over batches from the provided dataloader, performs forward
    and backward passes, updates model parameters, and computes average
    loss and accuracy across all samples.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
      Data loader providing training batches.

    Returns
    -------
    tuple[float, float]
      Tuple of average loss over all batches and
      average accuracy over all samples.
    """
    self.__model.train()

    total_loss = 0.0
    correct_pred = 0
    total_samples = 0
    for data, target in dataloader:
      data = data.to(DeviceManager.get(), non_blocking=True)
      target = target.to(DeviceManager.get(), non_blocking=True)

      self.__opt.zero_grad()

      output = self.__model(data)
      loss = self.__loss_function(output, target)
      total_loss += loss.item()

      preds = output.argmax(dim=1)
      correct_pred += (preds == target).sum().item()
      total_samples += target.size(0)

      loss.backward()

      self.__opt.step()

    return total_loss / len(dataloader), correct_pred / total_samples

  def eval(self, dataloader: DataLoader) -> tuple[float, float]:
    """
    Evaluate the model without gradient updates.

    Iterates over batches from the provided dataloader in evaluation mode,
    computes loss and accuracy, but does not update model parameters.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
      Data loader providing evaluation batches.

    Returns
    -------
    tuple[float, float]
      Tuple of average loss over all batches and
      average accuracy over all samples.
    """
    self.__model.eval()

    total_loss = 0.0
    correct_pred = 0
    total_samples = 0
    with torch.no_grad():
      for data, target in dataloader:
        data = data.to(DeviceManager.get(), non_blocking=True)
        target = target.to(DeviceManager.get(), non_blocking=True)

        output = self.__model(data)
        loss = self.__loss_function(output, target)
        total_loss += loss.item()

        preds = output.argmax(dim=1)
        correct_pred += (preds == target).sum().item()
        total_samples += target.size(0)

    return total_loss / len(dataloader), correct_pred / total_samples
