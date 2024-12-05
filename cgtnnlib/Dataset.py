from dataclasses import dataclass

from cgtnnlib.LearningTask import LearningTask

from cgtnnlib.DatasetData import DatasetData
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.nn.NetworkLike import NetworkLike


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
        "!!! Please don't use this"

        # ::: PthPath config variable?
        return f'pth/model-{self.number}A-c-P{params.p}_N{params.iteration}.pth'

    def model_b_path(self, params: ExperimentParameters) -> str:
        "!!! Please don't use this"

        return f'pth/model-{self.number}B-c-P{params.p}_N{params.iteration}.pth'

    def model_path(self, params: ExperimentParameters, model: NetworkLike) -> str:
        return f'pth/cgtnn-{self.number}X-{type(model).__name__}-c-P{params.p}_N{params.iteration}.pth'