from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters


import torch.nn as nn


from dataclasses import dataclass


@dataclass
class TrainingParameters:
    dataset: Dataset
    criterion: nn.CrossEntropyLoss | nn.MSELoss
    experiment_params: ExperimentParameters
    model_a_path: str
    model_b_path: str
    loss_curve_plot_col_index: int