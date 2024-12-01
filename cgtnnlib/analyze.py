## Result data analysis routines v.0.2
## Created at Thu 28 Nov 2024
## v.0.2 search_plot_data raises IndexError on failed search

from cgtnnlib.PlotParams import PlotParams
from cgtnnlib.Report import make_search_index, search_plot_data
import os

from typing import Any, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import cgtnnlib.Report as report
from cgtnnlib.PlotParams import Measurement, Metric

from cgtnnlib.Report import load_raw_report

def df_head_fraction(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    df_head_fraction(df, frac=0.15)
    """
    n_rows = int(len(df) * frac)
    return df.head(n_rows)

# 3. Вывод графиков
def plot_curve_on_ax_or_plt(
    ax_or_plt,
    means: pd.Series,
    lowerqs: pd.Series,
    upperqs: pd.Series,
    zmeans: pd.Series,
    zlowerqs: pd.Series,
    zupperqs: pd.Series,
    X: pd.Index,
    title: str,
    xlabel: str,
    ylabel: str,
):
    ax_or_plt.plot(X, zmeans, label='Mean of p = 0', color='lightblue')
    ax_or_plt.fill_between(X, zlowerqs, zupperqs, color='lightgray', alpha=0.5, label='0.25 to 0.75 Quantiles, p = 0')
    ax_or_plt.plot(X, means, label='Mean', color='blue')
    ax_or_plt.fill_between(X, lowerqs, upperqs, color='gray', alpha=0.5, label='0.25 to 0.75 Quantiles')

    is_ax = hasattr(ax_or_plt, 'set_xlabel')
    if is_ax:
        ax_or_plt.set_xlabel(xlabel)
        ax_or_plt.set_ylabel(ylabel)
        ax_or_plt.set_title(title)
    else:
        ax_or_plt.xlabel(xlabel)
        ax_or_plt.ylabel(xlabel)
        ax_or_plt.title(title)

    ax_or_plt.legend()

def extract_values_from_search_results(
    search_results: pd.DataFrame,
    raw_report: dict[str, Any],
    measurement: Measurement,
    metric: str,
):
    if measurement == 'loss':
        return pd.DataFrame([raw_report[str(row.Key)] for row in search_results.itertuples()])
    else:
        cols = []
        
        for row in search_results.itertuples():
            report_data = raw_report[str(row.Key)]
            cols.append(report_data[metric])
            
        return pd.DataFrame(cols)

class AnalysisParams(TypedDict):
    measurement: Measurement
    dataset_number: int
    xlabel: str
    frac: float


def search_curve(
    search_index: report.SearchIndex,
    plot_params: PlotParams,
    raw_report: report.RawReport,
) -> pd.DataFrame:
    values = extract_values_from_search_results(
        search_results=search_plot_data(
            search_index=search_index,
            plot_params=plot_params,
        ),
        raw_report=raw_report,
        measurement=plot_params.measurement,
        metric=plot_params.metric,
    )

    result = values.quantile([0.25, 0.75]).transpose()
    result['mean'] = values.mean()

    return df_head_fraction(
        df=result,
        frac=plot_params.frac
    )    

def analyze_outer(
    search_index: report.SearchIndex,
    raw_report: report.RawReport,
    analysis_params_list: list[AnalysisParams]
) -> None:
    for analysis_params in analysis_params_list:
        measurement = analysis_params['measurement']
        dataset_number = analysis_params['dataset_number']
        xlabel = analysis_params['xlabel']
        frac = analysis_params['frac']
        
        metrics: list[Metric]

        if measurement == 'loss':
            metrics = ['loss']
        else:
            if dataset_number == 3:
                metrics = ['r2', 'mse']
            else:
                metrics = ['f1', 'accuracy', 'roc_auc']
        
        fig, axs = plt.subplots(len(metrics), 6, figsize=(24, len(metrics) * 6))

        for (i, metric) in enumerate(metrics):
            def make_curve_for_p(p: float) -> pd.DataFrame:
                return search_curve(
                    search_index=search_index,
                    plot_params=PlotParams(
                        measurement=measurement,
                        dataset_number=dataset_number,
                        metric=metric,
                        p=p,
                        frac=frac,
                    ),
                    raw_report=raw_report
                )

            reference_curve: pd.DataFrame = make_curve_for_p(0)

            for (j, p) in enumerate([0.01, 0.05, 0.5, 0.9, 0.95, 0.99]):
                curve: pd.DataFrame = make_curve_for_p(p)
                
                plot_curve_on_ax_or_plt(
                    ax_or_plt=axs[i, j] if len(metrics) > 1 else axs[j],
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

default_analysis_params_list: list[AnalysisParams] = [
    {'measurement': 'loss', 'dataset_number': 1,
     'xlabel': 'iteration', 'frac': 1 },
    {'measurement': 'loss', 'dataset_number': 1,
     'xlabel': 'iteration', 'frac': 0.02},
    {'measurement': 'loss', 'dataset_number': 1,
     'xlabel': 'iteration', 'frac': 0.1},
    {'measurement': 'loss', 'dataset_number': 1,
     'xlabel': 'iteration', 'frac': 0.2},
    {'measurement': 'evaluate', 'dataset_number': 1,
     'xlabel': 'noise factor', 'frac': 1},
    {'measurement': 'loss', 'dataset_number': 2,
     'xlabel': 'iteration', 'frac': 1},
    {'measurement': 'loss', 'dataset_number': 2,
     'xlabel': 'iteration', 'frac': 0.02},
    {'measurement': 'loss', 'dataset_number': 2,
     'xlabel': 'iteration', 'frac': 0.1},
    {'measurement': 'loss', 'dataset_number': 2,
     'xlabel': 'iteration', 'frac': 0.2},
    {'measurement': 'evaluate', 'dataset_number': 2,
     'xlabel': 'noise factor', 'frac': 1},
    {'measurement': 'loss', 'dataset_number': 3,
     'xlabel': 'iteration', 'frac': 1},
    {'measurement': 'loss', 'dataset_number': 3,
     'xlabel': 'iteration', 'frac': 0.02},
    {'measurement': 'loss', 'dataset_number': 3,
     'xlabel': 'iteration', 'frac': 0.1},
    {'measurement': 'loss', 'dataset_number': 3,
     'xlabel': 'iteration', 'frac': 0.2},
    {'measurement': 'evaluate', 'dataset_number': 3,
     'xlabel': 'noise factor', 'frac': 1},
]

def analyze_main(
    report_path: str,
    analysis_params_list: list[AnalysisParams] = default_analysis_params_list,
) -> None:
    raw_report = load_raw_report(report_path)
    search_index = make_search_index(raw_report)

    analyze_outer(
        search_index=search_index,
        raw_report=raw_report,
        analysis_params_list=analysis_params_list,
    )

def analyze_just_one(
    report_path: str,
    plot_params: PlotParams,
):
    raw_report = load_raw_report(report_path)
    search_index = make_search_index(raw_report)
    
    return search_curve(search_index, plot_params, raw_report)