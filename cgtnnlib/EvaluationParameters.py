## EvaluationParameters v.0.2
## Created at Thu 28 Nov 2024
## Modified at Sun 1 Dec 2024
## v.0.2 removed inputs_count and outputs_count
##       (they may be derived from dataset)

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
    task: LearningTask
    experiment_parameters: ExperimentParameters
    report_key: str