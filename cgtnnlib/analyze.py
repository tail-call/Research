## Result data analysis routines v.0.1
## Created at Thu 28 Nov 2024

import json
import os

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

def df_head_fraction(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """df_head_fraction(df, frac=0.15)"""
    n_rows = int(len(df) * frac)
    return df.head(n_rows)

# 3. Вывод графиков
@dataclass
class PlotParams:
    measurement: str
    dataset_number: int
    network = 'AugmentedReLUNetwork'
    metric: str
    p: float
    
def compute_dataframe(
    search_index: pd.DataFrame,
    report: dict[str, Any],
    plot_params: PlotParams
) -> pd.DataFrame:
    rows = (
        search_index
            .loc[search_index.Measurement == plot_params.measurement]
            .loc[search_index.Dataset == plot_params.dataset_number]
            .loc[search_index.Network == plot_params.network]
            .loc[search_index.P == plot_params.p]
    )

    if plot_params.measurement == 'loss':
        values = pd.DataFrame([report[row.Key] for row in rows.itertuples()])
    else:
        cols = []
        
        for row in rows.itertuples():
            report_data = report[row.Key]
            cols.append(report_data[plot_params.metric])
            
        values = pd.DataFrame(cols)

    result = values.quantile([0.25, 0.75]).transpose()
    result['mean'] = values.mean()
    return result


def plot_curve_on_ax(
    ax,
    means: pd.Series,
    lowerqs: pd.Series,
    upperqs: pd.Series,
    zmeans: pd.Series,
    zlowerqs: pd.Series,
    zupperqs: pd.Series,
    X: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
):
    ax.plot(X, zmeans, label='Mean of p = 0', color='lightblue')
    ax.fill_between(X, zlowerqs, zupperqs, color='lightgray', alpha=0.5, label='0.25 to 0.75 Quantiles, p = 0')
    ax.plot(X, means, label='Mean', color='blue')
    ax.fill_between(X, lowerqs, upperqs, color='gray', alpha=0.5, label='0.25 to 0.75 Quantiles')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def analyze_outer(
    search_index: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    for (measurement, dataset_number, xlabel, frac) in [
        ('loss', 1, 'iteration', 1),
        ('loss', 1, 'iteration', 0.02),
        ('loss', 1, 'iteration', 0.1),
        ('loss', 1, 'iteration', 0.2),
        ('evaluate', 1, 'noise factor', 1),
        ('loss', 2, 'iteration', 1),
        ('loss', 2, 'iteration', 0.02),
        ('loss', 2, 'iteration', 0.1),
        ('loss', 2, 'iteration', 0.2),
        ('evaluate', 2, 'noise factor', 1),
        ('loss', 3, 'iteration', 1),
        ('loss', 3, 'iteration', 0.02),
        ('loss', 3, 'iteration', 0.1),
        ('loss', 3, 'iteration', 0.2),
        ('evaluate', 3, 'noise factor', 1),
    ]:
        if measurement == 'loss':
            metrics = ['loss']
        else:
            if dataset_number == 3:
                metrics = ['r2', 'mse']
            else:
                metrics = ['f1', 'accuracy', 'roc_auc']
        
        fig, axs = plt.subplots(len(metrics), 6, figsize=(24, len(metrics) * 6))

        for (i, metric) in enumerate(metrics):
            def make_curve(p: float) -> pd.DataFrame:
                return compute_dataframe(
                    search_index=search_index,
                    report=report,
                    plot_params=PlotParams(
                        measurement=measurement,
                        dataset_number=dataset_number,
                        metric=metric,
                        p=p,
                    )
                )

            reference_curve: pd.DataFrame = df_head_fraction(
                df=make_curve(p=0),
                frac=frac
            )

            for (j, p) in enumerate([0.01, 0.05, 0.5, 0.9, 0.95, 0.99]):
                curve: pd.DataFrame = df_head_fraction(
                    df=make_curve(p=p),
                    frac=frac
                )

                plot_curve_on_ax(
                    ax=axs[i, j] if len(metrics) > 1 else axs[j],
                    means=curve['mean'],
                    lowerqs=curve[0.25],
                    upperqs=curve[0.75],
                    zmeans=reference_curve['mean'],
                    zlowerqs=reference_curve[0.25],
                    zupperqs=reference_curve[0.75],
                    X=curve.index,
                    title=f'p = {p}',
                    xlabel=xlabel,
                    ylabel=metric,
                )
        fig.suptitle(f'Dataset #{dataset_number}, zoom factor: {frac}')
        plt.tight_layout()
        path = os.path.join('report/', f'{measurement}_{dataset_number}_f{frac:.02f}.png')
        plt.savefig(path)
        plt.close()

def analyze_main(report_path: str) -> None:
    with open(report_path) as fd:
        report = json.load(fd)

    df = pd.DataFrame([[key] + key.split('_') for key in report.keys()])
    df.columns = ['Key', 'Measurement', 'Network', 'Dataset', 'P', 'N']
    df = df[df['Key'] != 'started']
    df = df[df['Key'] != 'saved']
    df.Dataset = df.Dataset.apply(lambda x: int(x))
    df.P = df.P.apply(lambda x: float(x[1:]))
    df.N = df.N.apply(lambda x: int(x[1:]))
    
    analyze_outer(
        search_index=df,
        report=report
    )