from dataclasses import dataclass

from .DatasetData import DatasetData
from .ExperimentParameters import ExperimentParameters


@dataclass
class Dataset:
    name: str
    number: int
    features_count: int
    classes_count: int
    data: DatasetData

    def model_a_path(self, params: ExperimentParameters) -> str:
        # XXX PthPath config variable?
        return f'pth/model-{self.number}A-c-P{params.p}_N{params.iteration}.pth'

    def model_b_path(self, params: ExperimentParameters) -> str:
        return f'pth/model-{self.number}B-c-P{params.p}_N{params.iteration}.pth'