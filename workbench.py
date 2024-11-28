from cgtnnlib.AugmentedReLUNetwork import AugmentedReLUNetwork
from cgtnnlib.EvaluationParameters import EvaluationParameters
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.Report import Report, eval_report_key
from cgtnnlib.TrainingParameters import TrainingParameters
from cgtnnlib.LearningTask import classification_task
from cgtnnlib.common import DATASETS, LEARNING_RATE, eval_inner
from cgtnnlib.training import train_model_outer

report = Report(dir="workbench/")

dataset = DATASETS[0]
model_constructor = AugmentedReLUNetwork
model_path = "WorkbenchModel1.pth"
experiment_params = ExperimentParameters(
    iteration=777,
    p=0.05
)
training_params = TrainingParameters(
    dataset=dataset,
    learning_task=classification_task,
    experiment_params=experiment_params,
)
eval_params = EvaluationParameters(
    dataset=dataset,
    model_path=model_path,
    is_binary_classification=True,
    is_regression=False,
    inputs_count=30,
    outputs_count=2,
    task=classification_task,
    experiment_parameters=experiment_params,
    report_key=eval_report_key(
        model_name=model_constructor.__name__,
        dataset_number=DATASETS[0].number,
        p=experiment_params.p,
        iteration=experiment_params.iteration,
    )
)
model = model_constructor(
    inputs_count=dataset.features_count,
    outputs_count=dataset.classes_count,
    p=experiment_params.p,
)

train_model_outer(
    model=model,
    filename=model_path,
    epochs=5,
    training_params=training_params,
    experiment_params=experiment_params,
    learning_rate=LEARNING_RATE,
    report=report,
    dry_run=False
)

report.see()
eval_inner(
    eval_params=eval_params,
    experiment_params=experiment_params,
    constructor=model_constructor
)