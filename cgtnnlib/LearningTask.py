## 1.4.3 Learning task types

from typing import Literal, TypeAlias
from dataclasses import dataclass

import torch
import torch.nn as nn

LearningTaskName: TypeAlias = Literal['classification', 'regression']

@dataclass
class LearningTask:
    name: LearningTaskName
    criterion: nn.CrossEntropyLoss | nn.MSELoss
    dtype: torch.dtype


classification_task = LearningTask(
    name='classification',
    criterion=nn.CrossEntropyLoss(),
    dtype=torch.long,
)

regression_task = LearningTask(
    name='regression',
    criterion=nn.MSELoss(),
    dtype=torch.float,
)

def is_classification_task(task: LearningTask) -> bool:
    return task.name == classification_task.name

def is_regression_task(task: LearningTask) -> bool:
    return task.name == regression_task.name