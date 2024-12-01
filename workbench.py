from cgtnnlib.AugmentedReLUNetwork import AugmentedReLUNetwork
from cgtnnlib.EvaluationParameters import EvaluationParameters
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.PlotModel import PlotModel
from cgtnnlib.Report import Report, eval_report_key
from cgtnnlib.TrainingParameters import TrainingParameters
from cgtnnlib.analyze import search_curve_for_report
from cgtnnlib.common import DATASETS, LEARNING_RATE, eval_inner
from cgtnnlib.training import train_model_outer

REPORT = Report(dir="workbench/")
DATASET = DATASETS[2]
P = 0.5

print(DATASET)

MODEL_CONSTRUCTOR = AugmentedReLUNetwork


for iteration in range(1, 11):
    model_path = f"WorkbenchModel{iteration}.pth"
    experiment_params = ExperimentParameters(
        iteration=iteration,
        p=P
    )
    training_params = TrainingParameters(
        dataset=DATASET,
        learning_task=DATASET.learning_task,
        experiment_params=experiment_params,
    )
    eval_params = EvaluationParameters(
        dataset=DATASET,
        model_path=model_path,
        experiment_parameters=experiment_params,
        report_key=eval_report_key(
            model_name=MODEL_CONSTRUCTOR.__name__,
            dataset_number=DATASET.number,
            p=P,
            iteration=iteration,
        )
    )
    model = MODEL_CONSTRUCTOR(
        inputs_count=DATASET.features_count,
        outputs_count=DATASET.classes_count,
        p=experiment_params.p,
    )

    train_model_outer(
        model=model,
        filename=model_path,
        epochs=5,
        training_params=training_params,
        experiment_params=experiment_params,
        learning_rate=LEARNING_RATE,
        report=REPORT,
        dry_run=False
    )

    eval_inner(
        eval_params=eval_params,
        experiment_params=experiment_params,
        constructor=MODEL_CONSTRUCTOR
    )

REPORT.see()
REPORT.save()

search_curve_for_report(
    report_path=REPORT.path,
    model=PlotModel(
        measurement='loss',
        dataset_number=DATASET.number,
        metric='loss',
        p=P,
        frac=1.0,
    )
)