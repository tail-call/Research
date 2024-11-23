## 1.4.3 Learning task types

from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class LearningTask:
    criterion: object
    dtype: torch.dtype


CLASSIFICATION_TASK = LearningTask(
    criterion=nn.CrossEntropyLoss(),
    dtype=torch.long
)
REGRESSION_TASK = LearningTask(
    criterion=nn.MSELoss(),
    dtype=torch.float
)