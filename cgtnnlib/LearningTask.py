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


CLASSIFICATION_TASK = LearningTask(
    name='classification',
    criterion=nn.CrossEntropyLoss(),
    dtype=torch.long,
)

REGRESSION_TASK = LearningTask(
    name='regression',
    criterion=nn.MSELoss(),
    dtype=torch.float,
)

def is_classification_task(task: LearningTask) -> bool:
    return task.name == CLASSIFICATION_TASK.name

def is_regression_task(task: LearningTask) -> bool:
    return task.name == REGRESSION_TASK.name