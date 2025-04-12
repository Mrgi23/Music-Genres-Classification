import torch
import torch.nn as nn
from torch.nn.functional import relu

class MusicModel(nn.Module):
    def __init__(self) -> None:
        super(MusicModel, self).__init__()

        # First layer.
        self._conv1 = nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn1 = nn.BatchNorm2d(512)

        # Second layer.
        self._conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(256)

        # Third layer.
        self._conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn3 = nn.BatchNorm2d(128)

        # Pool layers.
        self._max_pool = nn.MaxPool2d(2, 2)
        self._adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC layers.
        self._flat = nn.Flatten()
        self._linear = nn.Linear(128, 64)
        self._dropout = nn.Dropout(0.3)
        self._output = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)

        # Apply first layer.
        x = self._max_pool(relu(self._bn1(self._conv1(x))))

        # Apply second layer.
        x = self._max_pool(relu(self._bn2(self._conv2(x))))

        # Apply third layer.
        x = self._max_pool(relu(self._bn3(self._conv3(x))))

        # Flatten the output.
        x = self._adaptive_pool(x)
        x = self._flat(x)

        # Apply FC layers.
        x = relu(self._linear(x))
        x = self._dropout(x)
        x = self._output(x)
        return x
