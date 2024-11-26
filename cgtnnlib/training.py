## Training module v.0.2
## Created at Tue 26 Nov 2024
## Updated at Tue 27 Nov 2024

from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from IPython.display import clear_output

from cgtnnlib.AugmentedReLUNetwork import AugmentedReLUNetwork
from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.LearningTask import classification_task, regression_task
# from cgtnnlib.RegularNetwork import RegularNetwork
from cgtnnlib.Report import Report
from cgtnnlib.TrainingParameters import TrainingParameters
from cgtnnlib.torch_device import torch_device

PRINT_TRAINING_SPAN = 500

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

def init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_model(
    model,
    dataset: Dataset,
    epochs: int,
    experiment_parameters: ExperimentParameters,
    criterion,
    optimizer,
    dry_run: bool,
) -> list[float]:
    running_losses: list[float] = []

    if dry_run:
        print(f"NOTE: Training model {model} in dry run mode. No changes to weights will be applied. An array of {epochs} -1s is generated for running_losses.")
        return [-1 for _ in range(epochs)]

    for epoch in range(epochs):
        running_loss = 0.0

        model.train()

        for i, (inputs, labels) in enumerate(dataset.data.train_loader):
            inputs, labels = inputs.to(torch_device), labels.to(torch_device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(torch_device)
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


def train_all_models(
    datasets: list[Dataset],
    epochs: int,
    learning_rate: float,
    report: Report,
    dry_run: bool,
    experiment_params_iter: Iterable[ExperimentParameters]
):
    for experiment_params in experiment_params_iter:
        # fig, axs = plt.subplots(2, 3, sharey='col', figsize=(10, 12))
        # fig.set_size_inches(35, 20)

        for training_params in [
            TrainingParameters(
                dataset=datasets[0],
                criterion=classification_task.criterion,
                experiment_params=experiment_params,
                model_a_path=datasets[0].model_a_path(experiment_params),
                model_b_path=datasets[0].model_b_path(experiment_params),
                loss_curve_plot_col_index=0,
            ),
            TrainingParameters(
                dataset=datasets[1],
                criterion=classification_task.criterion,
                experiment_params=experiment_params,
                model_a_path=datasets[1].model_a_path(experiment_params),
                model_b_path=datasets[1].model_b_path(experiment_params),
                loss_curve_plot_col_index=1,
            ),
            TrainingParameters(
                dataset=datasets[2],
                criterion=regression_task.criterion,
                experiment_params=experiment_params,
                model_a_path=datasets[2].model_a_path(experiment_params),
                model_b_path=datasets[2].model_b_path(experiment_params),
                loss_curve_plot_col_index=2,
            )
        ]:
            inputs_count = training_params.dataset.features_count
            outputs_count = training_params.dataset.classes_count


            for (model, name, row) in [
                ## Uncomment to train RegularNetwork
                # (RegularNetwork(
                #     inputs_count=inputs_count,
                #     outputs_count=outputs_count,
                #     p=experiment_params.p
                # ), training_params.model_a_path, 0),
                (AugmentedReLUNetwork(
                    inputs_count=inputs_count,
                    outputs_count=outputs_count,
                    p=experiment_params.p
                ), training_params.model_b_path, 1)
            ]:
                model.apply(init_weights)

                model = model.to(torch_device)

                running_losses = train_model(
                    model=model,
                    dataset=training_params.dataset,
                    epochs=epochs,
                    experiment_parameters=experiment_params,
                    criterion=training_params.criterion,
                    optimizer=optim.Adam(
                        model.parameters(),
                        lr=learning_rate,
                    ),
                    dry_run=dry_run,
                )

                torch.save(model.state_dict(), name)

                report_key = f'loss_{type(model).__name__}_{training_params.dataset.number}_p{experiment_params.p}_N{experiment_params.iteration}'

                report.append(report_key, running_losses)

                # XXX Unneeded?
                # col = training_params.loss_curve_plot_col_index
                # plot_loss_curve(
                #     ax=axs[row, col],
                #     model_name=model.__class__.__name__,
                #     dataset_name=training_params.dataset.name,
                #     dataset_number=training_params.dataset.number,
                #     running_losses=running_losses,
                #     p=experiment_params.p,
                #     iteration=experiment_params.iteration
                # )

        path = f'loss__p{experiment_params.p}_N{experiment_params.iteration}'
        print('train_main(): path = ', path)