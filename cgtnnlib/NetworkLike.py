from typing import Protocol, runtime_checkable

import torch

@runtime_checkable
class NetworkLike(Protocol):
    """
    Protocol for a neural network with custom ReLU layers.
    """

    @property
    def inputs_count(self) -> int:
        """
        Number of input features.
        """
        ...

    @property
    def outputs_count(self) -> int:
        """
        Number of output features.
        """
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        """
        ...

    def __str__(self) -> str:
        """
        String representation of the network.
        """
        ...