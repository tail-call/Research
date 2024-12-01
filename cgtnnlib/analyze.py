## Result data analysis routines v.0.2
## Created at Thu 28 Nov 2024
## v.0.2 search_plot_data raises IndexError on failed search

from cgtnnlib.Report import make_search_index, search_plot_data
import os

from typing import Any, TypeAlias, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import cgtnnlib.Report as report
from cgtnnlib.PlotModel import PlotModel, Measurement, Metric

from cgtnnlib.Report import load_raw_report

def df_head_fraction(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    df_head_fraction(df, frac=0.15)
    """
    n_rows = int(len(df) * frac)
    return df.head(n_rows)

Color: TypeAlias = str 
"Like 'lightblue', 'lightgray', etc"

class DeviantCurvePlotModel(TypedDict):
    curve: pd.DataFrame
    color: Color
    label: str
    quantiles_color: Color
    quantiles_label: str
    pass

# 3. Вывод графиков
def plot_deviant_curves_on_ax_or_plt(
    ax_or_plt,
    models: list[DeviantCurvePlotModel],
    title: str,
    xlabel: str,
    ylabel: str,
):
    if len(models) == 0:
        raise TypeError("models should not be empty")
    
    X: pd.Index = models[0]['curve'].index
    
    for model in models:
        expected_columns = [0.25, 0.75, 'mean']
        columns = model['curve'].columns.to_list()
        
        assert columns == expected_columns, f"Bad value of curve_df.columns: should be {expected_columns}, instead got {columns}"

        ax_or_plt.plot(
            X,
            model['curve']['mean'],
            label=model['label'],
            color=model['color'],
        )

        ax_or_plt.fill_between(
            X,
            model['curve'][0.25],
            model['curve'][0.75],
            color=model['quantiles_color'],
            alpha=0.5,
            label=model['quantiles_label'],
        )
        

    is_ax = hasattr(ax_or_plt, 'set_xlabel')
    if is_ax:
        ax_or_plt.set_xlabel(xlabel)
        ax_or_plt.set_ylabel(ylabel)
        ax_or_plt.set_title(title)
    else:
        ax_or_plt.xlabel(xlabel)
        ax_or_plt.ylabel(ylabel)
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
    plot_params: PlotModel,
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

def plot_analysis_fig(
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
                    plot_params=PlotModel(
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
                plot_deviant_curves_on_ax_or_plt(
                    ax_or_plt=axs[i, j] if len(metrics) > 1 else axs[j],
                    models=[{
                        'curve': reference_curve,
                        'color': 'lightblue',
                        'label': 'Mean of p = 0',
                        'quantiles_color': 'lightgray',
                        'quantiles_label': '0.25 to 0.75 Quantiles',
                    }, {
                        'curve': make_curve_for_p(p),
                        'color': 'Mean',
                        'label': 'blue',
                        'quantiles_color': 'gray',
                        'quantiles_label': '0.25 to 0.75 Quantiles',
                    },],
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

    plot_analysis_fig(
        search_index=search_index,
        raw_report=raw_report,
        analysis_params_list=analysis_params_list,
    )

def search_curve_for_report(
    report_path: str,
    model: PlotModel,
):
    raw_report = load_raw_report(report_path)
    search_index = make_search_index(raw_report)
    
    return search_curve(search_index, model, raw_report)