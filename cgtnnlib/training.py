## Training module v.0.1
## Created at Tue 26 Nov 2024

import torch.optim as optim
import torch

from IPython.display import clear_output

from cgtnnlib.AugmentedReLUNetwork import AugmentedReLUNetwork
from cgtnnlib.Dataset import Dataset
from cgtnnlib.ExperimentParameters import ExperimentParameters
from cgtnnlib.LearningTask import classification_task, regression_task
# from cgtnnlib.RegularNetwork import RegularNetwork
from cgtnnlib.TrainingParameters import TrainingParameters
from cgtnnlib.common import DATASETS, DRY_RUN, EPOCHS, LEARNING_RATE, PRINT_TRAINING_SPAN, append_to_report, device, init_weights, iterate_experiment_parameters, plot_loss_curve, save_plot

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

def train_model(
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


def train_all_models():
    for experiment_params in iterate_experiment_parameters():
        # fig, axs = plt.subplots(2, 3, sharey='col', figsize=(10, 12))
        # fig.set_size_inches(35, 20)

        for training_params in [
            TrainingParameters(
                dataset=DATASETS[0],
                criterion=classification_task.criterion,
                experiment_params=experiment_params,
                model_a_path=DATASETS[0].model_a_path(experiment_params),
                model_b_path=DATASETS[0].model_b_path(experiment_params),
                loss_curve_plot_col_index=0,
            ),
            TrainingParameters(
                dataset=DATASETS[1],
                criterion=classification_task.criterion,
                experiment_params=experiment_params,
                model_a_path=DATASETS[1].model_a_path(experiment_params),
                model_b_path=DATASETS[1].model_b_path(experiment_params),
                loss_curve_plot_col_index=1,
            ),
            TrainingParameters(
                dataset=DATASETS[2],
                criterion=regression_task.criterion,
                experiment_params=experiment_params,
                model_a_path=DATASETS[2].model_a_path(experiment_params),
                model_b_path=DATASETS[2].model_b_path(experiment_params),
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

                model = model.to(device)

                running_losses = train_model(
                    model=model,
                    dataset=training_params.dataset,
                    epochs=EPOCHS,
                    experiment_parameters=experiment_params,
                    criterion=training_params.criterion,
                    optimizer=optim.Adam(
                        model.parameters(),
                        lr=LEARNING_RATE
                    )
                )

                torch.save(model.state_dict(), name)

                report_key = f'loss_{type(model).__name__}_{training_params.dataset.number}_p{experiment_params.p}_N{experiment_params.iteration}'

                append_to_report(report_key, running_losses)

                col = training_params.loss_curve_plot_col_index

                plot_loss_curve(
                    ax=axs[row, col],
                    model_name=model.__class__.__name__,
                    dataset_name=training_params.dataset.name,
                    dataset_number=training_params.dataset.number,
                    running_losses=running_losses,
                    p=experiment_params.p,
                    iteration=experiment_params.iteration
                )

        path = save_plot(f'loss__p{experiment_params.p}_N{experiment_params.iteration}')
        print('train_main(): path = ', path)