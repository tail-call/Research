from dataclasses import dataclass

import pandas as pd
from torch.utils.data import DataLoader


@dataclass
class Dataset:
    features_count: int
    classes_count: int
    train_dataset: pd.DataFrame
    test_dataset: pd.DataFrame
    train_loader: DataLoader
    test_loader: DataLoader
