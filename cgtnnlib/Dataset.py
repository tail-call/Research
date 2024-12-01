from dataclasses import dataclass

from cgtnnlib.LearningTask import LearningTask

from cgtnnlib.DatasetData import DatasetData
from cgtnnlib.ExperimentParameters import ExperimentParameters


@dataclass
class Dataset:
    name: str
    learning_task: LearningTask
    number: int
    # ::: can be derived from DatasetData
    classes_count: int
    data: DatasetData
    
    @property
    def features_count(self) -> int:
        return self.data.train_dataset[1][0].shape[0]

    def model_a_path(self, params: ExperimentParameters) -> str:
        # ::: PthPath config variable?
        return f'pth/model-{self.number}A-c-P{params.p}_N{params.iteration}.pth'

    def model_b_path(self, params: ExperimentParameters) -> str:
        return f'pth/model-{self.number}B-c-P{params.p}_N{params.iteration}.pth'