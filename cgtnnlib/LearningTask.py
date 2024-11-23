## 1.4.3 Learning task types

from dataclasses import dataclass

import torch
import torch.nn as nn

@dataclass
class LearningTask:
    criterion: object
    dtype: torch.dtype


classification_task = LearningTask(
    criterion=nn.CrossEntropyLoss(),
    dtype=torch.long,
)

regression_task = LearningTask(
    criterion=nn.MSELoss(),
    dtype=torch.float,
)