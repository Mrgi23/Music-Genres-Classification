import torch
from torch.utils.data import DataLoader
from model import MusicModel

class DeviceManager():
    @staticmethod
    def get() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

class Trainer():
    def __init__(
        self,
        model: MusicModel,
        opt: torch.optim.Optimizer
    ) -> None:
        self.__model = model
        self.__opt = opt
        self.__loss_function = torch.nn.CrossEntropyLoss()

        self.__model.to(DeviceManager.get())

    def fit(self, dataloader: DataLoader) -> tuple[float, float]:
        # Set model int training mode.
        self.__model.train()

        # Iterate through the data loader and train the model.
        total_loss = 0.0
        correct_pred = 0
        total_samples = 0
        for data, target in dataloader:
            # Send data to device.
            data = data.to(DeviceManager.get(), non_blocking=True)
            target = target.to(DeviceManager.get(), non_blocking=True)

            # Zero the gradients.
            self.__opt.zero_grad()

            # Compute forward pass and calculate the loss.
            output = self.__model(data)
            loss = self.__loss_function(output, target)
            total_loss += loss.item()

            # Calculate accuracy.
            preds = output.argmax(dim=1)
            correct_pred += (preds == target).sum().item()
            total_samples += target.size(0)

            # Compute gradients via backward pass.
            loss.backward()

            # Update the model parameters.
            self.__opt.step()

        # Calculate average loss over number of batches and accuracy.
        return total_loss / len(dataloader), correct_pred / total_samples

    def eval(self, dataloader: DataLoader) -> float:
        # Set model into the evaluation mode (i.e., do not calculate gradients).
        self.__model.eval()

        # Iterate through the data loader and evaluate the model.
        total_loss = 0.0
        correct_pred = 0
        total_samples = 0
        with torch.no_grad():
            for data, target in dataloader:
                # Send data to device.
                data = data.to(DeviceManager.get(), non_blocking=True)
                target = target.to(DeviceManager.get(), non_blocking=True)

                # Compute forward pass and calculate the loss.
                output = self.__model(data)
                loss = self.__loss_function(output, target)
                total_loss += loss.item()

                # Calculate accuracy.
                preds = output.argmax(dim=1)
                correct_pred += (preds == target).sum().item()
                total_samples += target.size(0)

        # Calculate average loss over number of batches and accuracy.
        return total_loss / len(dataloader), correct_pred / total_samples
