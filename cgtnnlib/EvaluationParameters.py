## EvaluationParameters v.0.1
## Created at Thu 28 Nov

from dataclasses import dataclass

from cgtnnlib.Dataset import Dataset
from cgtnnlib.LearningTask import LearningTask
from cgtnnlib.ExperimentParameters import ExperimentParameters


@dataclass
class EvaluationParameters:
    dataset: Dataset
    model_path: str
    is_binary_classification: bool
    is_regression: bool
    inputs_count: int
    outputs_count: int
    task: LearningTask
    experiment_parameters: ExperimentParameters
    report_key: str