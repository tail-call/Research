## COMMON LIBRARY v.0.8
## Created at Sat 23 Nov 2024
## Updated at Wed 27 Nov 2024
## v.0.8 - evaluate_main()
## v.0.7 - training.py
## v.0.6 - even more classes within their own files
## v.0.5 - improved LearningTask interface
## v.0.4 - datasets module
## v.0.3 - more classes within their own files
## v.0.2 - evaluation declarations

## 1.4.-2 Imports

import os
from typing import Any
from typing_extensions import Literal

from IPython.display import clear_output

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_squared_error

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from cgtnnlib.AugmentedReLUNetwork import AugmentedReLUNetwork
from cgtnnlib.EvaluationParameters import EvaluationParameters
from cgtnnlib.RegularNetwork import RegularNetwork
from cgtnnlib.Report import Report, eval_report_key
from cgtnnlib.training import create_and_train_all_models
from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.datasets import make_dataset1, make_dataset2, make_dataset3
from cgtnnlib.LearningTask import classification_task, is_classification_task, is_regression_task, regression_task

## 1.4.-1 Configuration

DRY_RUN = False

REPORT_DIR = "report/"

TEST_SAMPLE_SIZE = 0.2
ITERATIONS = 10
RANDOM_STATE = 23432
BATCH_SIZE = 12
EPOCHS = 20
LEARNING_RATE = 0.00011
PP = [0, 0.01, 0.05, 0.5, 0.9, 0.95, 0.99]

NOISE_SAMPLES_COUNT = 50
NOISE_FACTORS = [
    x * 2/NOISE_SAMPLES_COUNT for x in range(NOISE_SAMPLES_COUNT)
]

## 1.4.-0,5 Initialization

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

report = Report(dir=REPORT_DIR)

def get_pointless_path(filename_without_extension: str) -> str:
    path = os.path.join(REPORT_DIR, f'{filename_without_extension}.png')
    print(f"NOTE: get_pointless_path called but no figure will be saved. Path: {path}")
    plt.close()
    return path

def plot_loss_curve(
    ax, # Axes
    model_name: str,
    dataset_name: str,
    dataset_number: int,
    running_losses: list[float],
    p: float,
    iteration: int,
):
    X = range(len(running_losses))
    
    ax.plot(X, running_losses, label='Running loss', color='orange')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title(f'Loss curve: {model_name} on {dataset_name} (#{dataset_number}), p = {p}, N = {iteration}')
    ax.legend()

def iterate_experiment_parameters():
    for iteration in range(0, ITERATIONS):
        for p in PP:
            yield ExperimentParameters(iteration, p)

## 1.4.11 Evaluation

def positive_probs_from(probs: torch.Tensor) -> np.ndarray[Any, np.dtype[Any]]:
    return np.array(probs)[:, 0]

def eval_accuracy_f1_rocauc(
    evaluated_model:RegularNetwork,
    dataset: Dataset,
    noise_factor: float,
    is_binary_classification: bool,
) -> tuple[float, float, float]:
    evaluated_model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataset.data.test_loader:
            outputs = evaluated_model(
                inputs + torch.randn(inputs.shape) * noise_factor
            )
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    if is_binary_classification:
        np_all_probs = np.array(all_probs)[:, 0]
    else:
        np_all_probs = np.array(all_probs)

    roc_auc = roc_auc_score(all_labels, np_all_probs, multi_class='ovr')

    return float(accuracy), float(f1), float(roc_auc)

cool_type = np.ndarray[Any, np.dtype[Any]]

def eval_r2_mse(
    evaluated_model: RegularNetwork,
    dataset: Dataset,
    noise_factor: float,
) -> tuple[float, float]:
    evaluated_model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataset.data.test_loader:
            noisy_inputs = inputs + torch.randn(inputs.shape) * noise_factor
            outputs = evaluated_model(noisy_inputs)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    np_all_labels = np.array(all_labels)
    np_all_predictions = np.array(all_predictions)

    r2 = r2_score(np_all_labels, np_all_predictions)
    mse = mean_squared_error(np_all_labels, np_all_predictions)

    return float(r2), float(mse)

def evaluate_regression_model(
    evaluated_model: RegularNetwork,
    dataset: Dataset,
    report_key: str,
)-> pd.DataFrame:
    samples = {
        'noise_factor': NOISE_FACTORS,
        'r2': [],
        'mse': [],
    }

    for noise_factor in NOISE_FACTORS:
        r2, mse = eval_r2_mse(
            evaluated_model=evaluated_model,
            dataset=dataset,
            noise_factor=noise_factor,
        )

        samples['r2'].append(r2)
        samples['mse'].append(mse)

    report.append(report_key, samples)

    return pd.DataFrame(samples)

def evaluate_classification_model(
    evaluated_model: RegularNetwork,
    dataset: Dataset,
    report_key: str,
    is_binary_classification: bool,
)-> pd.DataFrame:
    samples = {
        'noise_factor': NOISE_FACTORS,
        'accuracy': [],
        'f1': [],
        'roc_auc': [],
    }

    for noise_factor in NOISE_FACTORS:
        accuracy, f1, roc_auc = eval_accuracy_f1_rocauc(
            evaluated_model=evaluated_model,
            dataset=dataset,
            noise_factor=noise_factor,
            is_binary_classification=is_binary_classification,
        )

        samples['accuracy'].append(accuracy)
        samples['f1'].append(f1)
        samples['roc_auc'].append(roc_auc)

    report.append(report_key, samples)

    return pd.DataFrame(samples)

def plot_evaluation_of_classification(
    df: pd.DataFrame,
    accuracy_ax,
    f1_ax,
    roc_auc_ax,
    title: str,
):
    accuracy_ax.plot(df['noise_factor'], df['accuracy'], label='Accuracy', marker='o')
    # accuracy_ax.set_xscale('log')
    accuracy_ax.set_xlabel('Noise Factor')
    accuracy_ax.set_ylabel('Metric Value')
    # ???
    accuracy_ax.set_title(f'{title}')
    accuracy_ax.legend()
    accuracy_ax.grid(True, which="both", ls="--")

    f1_ax.plot(df['noise_factor'], df['f1'], label='F1 Score', marker='o')
    # f1_ax.set_xscale('log')
    f1_ax.set_xlabel('Noise Factor')
    f1_ax.set_ylabel('Metric Value')
    # ???
    f1_ax.set_title(f'{title}')
    f1_ax.legend()
    f1_ax.grid(True, which="both", ls="--")

    roc_auc_ax.plot(df['noise_factor'], df['roc_auc'], label='ROC AUC', marker='o')
    # axs[2].set_xscale('log')
    roc_auc_ax.set_xlabel('Noise Factor')
    roc_auc_ax.set_ylabel('Metric Value')
    # ???
    roc_auc_ax.set_title(f'{title}')
    roc_auc_ax.legend()
    roc_auc_ax.grid(True, which="both", ls="--")
    

def plot_evaluation_of_regression(
    df: pd.DataFrame,
    mse_ax,
    r2_ax,
    title: str
):
    mse_ax.plot(df['noise_factor'], df['mse'], label='Mean Square Error', marker='o')
    # mse_ax.set_xscale('log')
    mse_ax.set_xlabel('Noise Factor')
    mse_ax.set_ylabel('Metric Value')
    mse_ax.set_title(f'{title}')
    mse_ax.legend()
    mse_ax.grid(True, which="both", ls="--")

    r2_ax.plot(df['noise_factor'], df['r2'], label='R^2', marker='o')
    # r2_ax.set_xscale('log')
    r2_ax.set_xlabel('Noise Factor')
    r2_ax.set_ylabel('Metric Value')
    r2_ax.set_title(f'{title}')
    r2_ax.legend()
    r2_ax.grid(True, which="both", ls="--")

def model_path_for(
    model_a_or_b: Literal["A", "B"],
    dataset: Dataset,
    experiment_params: ExperimentParameters,
):
    if model_a_or_b == "A":
        return dataset.model_a_path(experiment_params)
    elif model_a_or_b == "B":
        return dataset.model_b_path(experiment_params)
    else:
        raise TypeError('model_a_or_b must be A or B')

def eval_inner(
    eval_params: EvaluationParameters,
    experiment_params: ExperimentParameters,
    constructor: type,
):
    evaluated_model = constructor(
        inputs_count=eval_params.dataset.features_count,
        outputs_count=eval_params.dataset.classes_count,
        p=experiment_params.p
    )

    clear_output(wait=True)
    print(f'Evaluating model at {eval_params.model_path}...')
    evaluated_model.load_state_dict(torch.load(eval_params.model_path))

    if is_classification_task(eval_params.task):
        df = evaluate_classification_model(
            evaluated_model=evaluated_model,
            dataset=eval_params.dataset,
            report_key=eval_params.report_key,
            is_binary_classification=eval_params.is_binary_classification
        )
        print('Evaluation of classification (head):')
        print(df.head())
        # plot_evaluation_of_classification(
        #     df=df,
        #     accuracy_ax=_.accuracy_ax,
        #     f1_ax=_.f1_ax,
        #     roc_auc_ax=_.roc_auc_ax,
        #     title=plot_title
        # )
    elif is_regression_task(eval_params.task):
        df = evaluate_regression_model(
            evaluated_model=evaluated_model,
            dataset=eval_params.dataset,
            report_key=eval_params.report_key
        )
        print('Evaluation of regression (head):')
        print(df.head())
        # plot_evaluation_of_regression(
        #     df=df,
        #     mse_ax=_.mse_ax,
        #     r2_ax=_.r2_ax,
        #     title=plot_title
        # )
    else:
        raise ValueError(f"Unknown task: {eval_params.task}")

def evaluate(
    model_a_or_b: Literal["A", "B"],
    constructor: type,
    experiment_params: ExperimentParameters
):
    """
    Оценивает модель `"A"` (`RegularNetwork`) или `"B"`
    (`AugmentedReLUNetwork`) согласно параметрам `experiment_params` на
    наборах данных из `DATASETS`.
    Рисует графики метрик и сохраняет их на диск.
    
    - `constructor` может быть `RegularNetwork` или `AugmentedReLUNetwork`
      и должен соответствовать переданному `model_a_or_b`.
    """

    eval_params_items: list[EvaluationParameters] = [
        EvaluationParameters(
            DATASETS[0],
            model_path_for(model_a_or_b, DATASETS[0], experiment_params),
            experiment_parameters=experiment_params,
            report_key=eval_report_key(
                model_name=constructor.__name__,
                dataset_number=DATASETS[0].number,
                p=experiment_params.p,
                iteration=experiment_params.iteration,
            )
        ),
        EvaluationParameters(
            DATASETS[1],
            model_path_for(model_a_or_b, DATASETS[1], experiment_params),
            experiment_parameters=experiment_params,
            report_key=eval_report_key(
                model_name=constructor.__name__,
                dataset_number=DATASETS[1].number,
                p=experiment_params.p,
                iteration=experiment_params.iteration,
            )
        ),
        EvaluationParameters(
            DATASETS[2],
            model_path_for(model_a_or_b, DATASETS[2], experiment_params),
            experiment_parameters=experiment_params,
            report_key=eval_report_key(
                model_name=constructor.__name__,
                dataset_number=DATASETS[2].number,
                p=experiment_params.p,
                iteration=experiment_params.iteration,
            )
        ),
    ]
    

    for (i, eval_params) in enumerate(eval_params_items):
        eval_inner(
            eval_params,
            experiment_params,
            constructor
        )

DATASETS: list[Dataset] = [
    make_dataset1(
        batch_size=12,
        test_size=TEST_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        learning_task=classification_task,
    ),
    make_dataset2(
        batch_size=12,
        test_size=TEST_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        learning_task=classification_task,
    ),
    make_dataset3(
        batch_size=12,
        test_size=TEST_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        learning_task=regression_task,
    ),
]

def train_main():
    create_and_train_all_models(
        datasets=DATASETS,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        report=report,
        dry_run=DRY_RUN,
        experiment_params_iter=iterate_experiment_parameters()
    )

def evaluate_main():
    for experiment_params in iterate_experiment_parameters():
        # evaluate(
        #     model_a_or_b='A',
        #     constructor=RegularNetwork,
        #     experiment_params=experiment_params
        # )
        evaluate(
            model_a_or_b='B',
            constructor=AugmentedReLUNetwork,
            experiment_params=experiment_params
        )

if __name__ == "__main__":
    print('# py')
    print('')
    print('Datasets:')
    for dataset in DATASETS:
        print(f"{dataset.number}) {dataset.name}: {dataset.features_count} features, {dataset.classes_count} classes")

if __name__ == '__main__':
    train_main()
    evaluate_main()