from dataclasses import dataclass

import torch.utils.data
import pandas as pd


@dataclass
class DatasetData:
    train_dataset: pd.DataFrame
    test_dataset: pd.DataFrame
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
