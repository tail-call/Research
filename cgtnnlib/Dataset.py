## Dataset class v.0.1
## Created at 23 Nov 2024
## Updated at Tue 14 Jan 2025

from dataclasses import dataclass
from typing import Callable, Union

from cgtnnlib.LearningTask import LearningTask

from cgtnnlib.DatasetData import DatasetData
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.nn.NetworkLike import NetworkLike

from torch.utils.data import TensorDataset, DataLoader

@dataclass
class Dataset:
    number: int
    name: str
    learning_task: LearningTask
    # ::: can be derived from DatasetData
    classes_count: int
    batch_size: int
    random_state: int
    test_size: float
    data_maker: Callable[[float, int], tuple[TensorDataset, TensorDataset]]

    _data: Union[DatasetData, None] = None

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

    @property
    def data(self) -> DatasetData:
        if self._data is None:
            train, test = self.data_maker(self.test_size, self.random_state)
            
            self._data = DatasetData(
                train_dataset=train,
                test_dataset=test,
                train_loader=DataLoader(
                    train,
                    batch_size=self.batch_size,
                    shuffle=True,
                ),
                test_loader=DataLoader(
                    test,
                    batch_size=self.batch_size,
                    shuffle=False,
                ),
            )
        
        _data: DatasetData = self._data
        return _data