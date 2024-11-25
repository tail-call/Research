## COMMON LIBRARY v.0.6
## Created at Sat 23 Nov 2024
## v.0.6 - even more classes within their own files
## v.0.5 - improved LearningTask interface
## v.0.4 - datasets module
## v.0.3 - more classes within their own files
## v.0.2 - evaluation declarations

## 1.4.-2 Imports

import os
import json

from IPython.display import clear_output

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import matplotlib.pyplot as plt

from cgtnnlib.RegularNetwork import RegularNetwork

from .Dataset import Dataset
from .ExperimentParameters import ExperimentParameters
from .datasets import make_dataset1, make_dataset2, make_dataset3


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
PRINT_TRAINING_SPAN = 500

NOISE_SAMPLES_COUNT = 50
NOISE_FACTORS = [
    x * 2/NOISE_SAMPLES_COUNT for x in range(NOISE_SAMPLES_COUNT)
]

## 1.4.-0,5 Initialization

if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

report_data = {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASETS = [
    make_dataset1(
        batch_size=12,
        test_size=TEST_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ),
    make_dataset2(
        batch_size=12,
        test_size=TEST_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ),
    make_dataset3(
        batch_size=12,
        test_size=TEST_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ),
]

## 1.4.-0,25 Various declarations (please do not add more)

def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def print_progress(
    p: float,
    iteration: int,
    epoch: int,
    total_epochs: int,
    total_samples: int,
    running_loss: float,
    dataset_number: int
):
    clear_output(wait=True)
    print(
        f'N={iteration} #{dataset_number} p={p} E{epoch}/{total_epochs} S{total_samples} Loss={running_loss / 100:.4f}'
    )

def train(
    model, 
    dataset: Dataset,
    epochs: int,
    experiment_parameters: ExperimentParameters,
    criterion,
    optimizer
) -> list[float]:
    running_losses: list[float] = []
    
    if DRY_RUN:
        print(f"NOTE: Training model {model} in dry run mode. No changes to weights will be applied. An array of {epochs} -1s is generated for running_losses.")
        return [-1 for _ in range(epochs)]

    for epoch in range(epochs):
        running_loss = 0.0

        model.train()

        for i, (inputs, labels) in enumerate(dataset.data.train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % PRINT_TRAINING_SPAN == 0:
                print_progress(
                    p=experiment_parameters.p,
                    iteration=experiment_parameters.iteration,
                    epoch=epoch + 1,
                    total_epochs=epochs,
                    total_samples=len(dataset.data.train_loader),
                    running_loss=running_loss,
                    dataset_number=dataset.number
                )


            running_losses.append(running_loss)
            running_loss = 0.0
    
    return running_losses

def save_plot(filename_without_extension: str) -> str:
    path = os.path.join(REPORT_DIR, f'{filename_without_extension}.png')
    # plt.savefig(path)
    print(f"NOTE: save_plot called but no figure will be saved. Path: {path}")
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

## 1.4.2 Report generation


def append_to_report(label: str, data: dict):
    report_data[label] = data

def save_report():
    path = os.path.join(REPORT_DIR, 'report.json')
    with open(path, 'w') as file:
        json.dump(report_data, file, indent=4)
    print(f"Отчёт сохранён в {path}")
    

## 1.4.11 Evaluation

def positive_probs_from(probs: torch.Tensor) -> list:
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
        all_probs = np.array(all_probs)[:, 0]

    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    return float(accuracy), float(f1), float(roc_auc)

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

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    r2 = r2_score(all_labels, all_predictions)
    mse = mean_squared_error(all_labels, all_predictions)

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

    append_to_report(report_key, samples)

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

    append_to_report(report_key, samples)

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

if __name__ == "__main__":
    print('# common.py')
    print('')
    print('Datasets:')
    for dataset in DATASETS:
        print(f"{dataset.number}) {dataset.name}: {dataset.features_count} features, {dataset.classes_count} classes")