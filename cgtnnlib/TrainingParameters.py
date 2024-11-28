from dataclasses import dataclass

from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.LearningTask import LearningTask


@dataclass
class TrainingParameters:
    dataset: Dataset
    learning_task: LearningTask
    experiment_params: ExperimentParameters