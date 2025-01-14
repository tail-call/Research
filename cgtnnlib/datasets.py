## Dataset utilities v.0.3
## Created at Tue 26 Nov 2024
## Updated at Tue 14 Jan 2024
## v.0.3 - Drop make_datasetN() functions in favor of datasets list
## v.0.2 - sha1 hash checking to avoid duplicate downloads

import os
import urllib.request
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset

from cgtnnlib.LearningTask import REGRESSION_TASK, CLASSIFICATION_TASK
from cgtnnlib.Dataset import Dataset

TEST_SAMPLE_SIZE = 0.2
RANDOM_STATE = 23432
BATCH_SIZE = 12

def download_csv(
    url: str,
    saved_name: str,
    sha1: str,
    features: list[str] | None = None
) -> pd.DataFrame:
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, saved_name)

    def calculate_sha1(file_path):
        hasher = hashlib.sha1()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    if os.path.exists(file_path):
        file_sha1 = calculate_sha1(file_path)
        if file_sha1 != sha1:
            raise ValueError(f"SHA1 mismatch for existing file: {file_path}. Expected {sha1}, got {file_sha1}")
        else:
            print(f"File {file_path} exists and SHA1 matches, skipping download.")
            if features is None:
                return pd.read_csv(file_path)
            else:
                return pd.read_csv(file_path, header=None, names=features)
    else:
        print(f"Downloading {url} to {file_path}")
        urllib.request.urlretrieve(url, file_path)
        downloaded_sha1 = calculate_sha1(file_path)
        if downloaded_sha1 != sha1:
            os.remove(file_path)
            raise ValueError(f"SHA1 mismatch for downloaded file: {file_path}. Expected {sha1}, got {downloaded_sha1}")


    if features is None:
            return pd.read_csv(file_path)
    else:
            return pd.read_csv(file_path, header=None, names=features)
        
def tensor_dataset_from_dataframe(
    df: pd.DataFrame,
    target: str,
    y_dtype: torch.dtype
) -> TensorDataset:
    X = df.drop(columns=[target]).values
    y = df[target].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=y_dtype)

    return TensorDataset(X_tensor, y_tensor)


## 1.4.5 Dataset #1

def breast_cancer_dataset(
    test_size: float,
    random_state: int,
) -> tuple[TensorDataset, TensorDataset]:
    df = download_csv(
        url='https://raw.githubusercontent.com/dataspelunking/MLwR/refs/heads/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2003/wisc_bc_data.csv',
        saved_name='wisc_bc_data.csv',
        sha1='3b75f889e7e8d140b9eb28df39556b94b4331e33',
    )

    target = 'diagnosis'

    df[target] = df[target].map({ 'M': 0, 'B': 1 })
    df = df.drop(columns=['id'])

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    return (
        tensor_dataset_from_dataframe(
            df=train_df,
            target=target,
            y_dtype=CLASSIFICATION_TASK.dtype
        ),
        tensor_dataset_from_dataframe(
            df=val_df,
            target=target,
            y_dtype=CLASSIFICATION_TASK.dtype
        )
    )


## 1.4.6 Dataset #2

def car_evaluation_dataset(
    test_size: float,
    random_state: int,
) -> tuple[TensorDataset, TensorDataset]:
    df = download_csv(
        url='https://raw.githubusercontent.com/mragpavank/car-evaluation-dataset/refs/heads/master/car_evaluation.csv',
        saved_name='car_evaluation.csv',
        sha1='985852bc1bb34d7cb3c192d6b8e7127cc743e176',
        features=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'],
    )

    target = 'class'

    df[target] = df[target].map({
        'unacc': 0,
        'acc': 1,
        'good': 2,
        'vgood': 3,
    })

    df['doors'] = df['doors'].map({
        '2': 2,
        '3': 3,
        '4': 4,
        '5more': 5
    })

    high_map = {
        'low': 0,
        'med': 1,
        'high': 2,
        'vhigh': 3
    }

    df['buying'] = df['buying'].map(high_map)
    df['safety'] = df['safety'].map(high_map)
    df['maint'] = df['maint'].map(high_map)

    df['persons'] = df['persons'].map({
        '2': 2,
        '4': 4,
        'more': 6
    })

    df['lug_boot'] = df['lug_boot'].map({
        'small': 0,
        'med': 1,
        'big': 2
    })

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    return (
        tensor_dataset_from_dataframe(
            df=train_df,
            target=target,
            y_dtype=CLASSIFICATION_TASK.dtype
        ),
        tensor_dataset_from_dataframe(
            df=val_df,
            target=target,
            y_dtype=CLASSIFICATION_TASK.dtype
        )
    )

## 1.4.7 Dataset #3

def student_performance_factors_dataset(
    test_size: float,
    random_state: int,
) -> tuple[TensorDataset, TensorDataset]:
    df = pd.read_csv('data/StudentPerformanceFactors.csv')

    target = 'Exam_Score'

    lmh = {
        'Low': -1,
        'Medium': 0,
        'High': +1,
    }

    yn = {
        'Yes': +1,
        'No': -1,
    }

    df = df.dropna(subset=['Teacher_Quality'])

    df['Parental_Involvement'] = df['Parental_Involvement'].map(lmh)
    df['Access_to_Resources'] = df['Access_to_Resources'].map(lmh)
    df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map(yn)
    df['Motivation_Level'] = df['Motivation_Level'].map(lmh)
    df['Internet_Access'] = df['Internet_Access'].map(yn)
    df['Family_Income'] = df['Family_Income'].map(lmh)
    df['Teacher_Quality'] = df['Teacher_Quality'].map(lmh)
    df['School_Type'] = df['School_Type'].map({
        'Public': +1,
        'Private': -1,
    })
    df['Peer_Influence'] = df['Peer_Influence'].map({
        'Positive': +1,
        'Neutral': 0,
        'Negative': -1,
    })
    df['Learning_Disabilities'] = df['Learning_Disabilities'].map(yn)
    df['Parental_Education_Level'] = df['Parental_Education_Level'].map({
        'Postgraduate': +3,
        'College': +2,
        'High School': +1,
    }).fillna(0)
    df['Distance_from_Home'] = df['Distance_from_Home'].map({
        'Near': +1,
        'Moderate': 0,
        'Far': -1,
    }).fillna(0)
    df['Gender'] = df['Gender'].map({
        'Female': +1,
        'Male': -1,
    }).fillna(0)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    return (
        tensor_dataset_from_dataframe(
            df=train_df,
            target=target,
            y_dtype=REGRESSION_TASK.dtype
        ),
        tensor_dataset_from_dataframe(
            df=val_df,
            target=target,
            y_dtype=REGRESSION_TASK.dtype
        )
    )

datasets: list[Dataset] = [
    Dataset(
        name='wisc_bc_data.csv',
        learning_task=CLASSIFICATION_TASK,
        number=1,
        classes_count=2,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        test_size=TEST_SAMPLE_SIZE,
        data_maker=car_evaluation_dataset,
    ),
    Dataset(
        number=2,
        name='car_evaluation.csv',
        learning_task=CLASSIFICATION_TASK,
        classes_count=4,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        test_size=TEST_SAMPLE_SIZE,
        data_maker=car_evaluation_dataset,
    ),
    Dataset(
        number=3,
        name='StudentPerformanceFactors.csv',
        learning_task=REGRESSION_TASK,
        classes_count=1,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        test_size=TEST_SAMPLE_SIZE,
        data_maker=student_performance_factors_dataset,
    ),
]