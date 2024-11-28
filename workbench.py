from cgtnnlib.AugmentedReLUNetwork import AugmentedReLUNetwork
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.Report import Report
from cgtnnlib.TrainingParameters import TrainingParameters
from cgtnnlib.LearningTask import classification_task
from cgtnnlib.common import DATASETS, LEARNING_RATE
from cgtnnlib.training import train_model_outer

experiment_params = ExperimentParameters(
    iteration=777,
    p=0.05
)

dataset = DATASETS[0]

model = AugmentedReLUNetwork(
    inputs_count=dataset.features_count,
    outputs_count=dataset.classes_count,
    p=experiment_params.p,
)

training_params = TrainingParameters(
    dataset=dataset,
    learning_task=classification_task,
    experiment_params=experiment_params,
)

report = Report(dir="workbench/")

train_model_outer(
    model=model,
    filename="WorkbenchModel1.pth",
    epochs=5,
    training_params=training_params,
    experiment_params=experiment_params,
    learning_rate=LEARNING_RATE,
    report=report,
    dry_run=False
)

report.see()