from dataclasses import dataclass

import torch
import matplotlib as plt

from torchvision import datasets, transforms
import torch.optim as optim


@dataclass
class Dataset:
    train_dataset: datasets.MNIST
    test_dataset: datasets.MNIST
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader

def make_dataset(root: str, batch_size: int) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    return Dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        ),
        test_loader=torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    )

dataset = make_dataset(root='./data', batch_size=64)



import torch.nn as nn
import torch.nn.functional as F

class CustomBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)

        output = input.mm(weight.t())
        if bias is not None:
            output += bias

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)

        if bias is not None:
            grad_bias = grad_output.sum(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias

    @staticmethod
    def XXXbackward(ctx, grad_output: torch.Tensor):
        # XXX Doesn't work

        # TODO: проверять, на каком слое находимся.

        input, weights, bias = ctx.saved_tensors

        # XXX [0, 1] сделать внешним параметром линейного слоя!!!
        p = 0.5

        # XXX не очень хорошо
        # нужно отключить передачу сигнала какому-то из нейронов
        # взять размер одной из осей матрицы весов
        # сгенерировать одномерный вектор
        # и на эту диагональну матрицу (справа или слева?) умножить
        # на матрицу весов
        diag_matrix = torch.bernoulli(
            torch.ones_like(weights) * 0.5
        )
        diag_matrix = torch.diag(diag_matrix)

        grad_output = grad_output @ diag_matrix
        grad_input = grad_output @ weights.t()
        grad_weights = input.t() @ grad_output

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        else:
            grad_bias = None

        return grad_input, grad_weights, grad_bias

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28 * 28, 32 * 32)
        self.fc2 = nn.Linear(32 * 32, 32 * 32)
        self.fc3 = nn.Linear(32 * 32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class DigitRecognizerCustomBackward(nn.Module):
    def __init__(self):
        super(DigitRecognizerCustomBackward, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(28 * 28, 32 * 32)
        self.fc2 = nn.Linear(32 * 32, 32 * 32)
        self.fc3 = nn.Linear(32 * 32, 10)

        self.custom_backward = CustomBackwardFunction.apply

    def forward(self, x):
        x = self.flatten(x)
        x = self.custom_backward(x, self.fc1.weight, self.fc1.bias)
        x = F.relu(x)
        x = self.custom_backward(x, self.fc2.weight, self.fc2.bias)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    
    import torch.optim as optim

MeanStd = tuple[float, float]

def dabs_mean_std(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor
) -> MeanStd:
    dabs: torch.Tensor = torch.abs(tensor2 - tensor1)

    mean = torch.mean(dabs)
    std = torch.std(dabs)

    return (mean.item(), std.item())


def train(model: DigitRecognizer, loader, epochs, criterion, optimizer):
    # Наблюдаемые величины
    running_losses: list[float] = []
    fc1_dabs_mean: list[float] = []
    fc1_dabs_std: list[float] = []
    fc2_dabs_mean: list[float] = []
    fc2_dabs_std: list[float] = []
    fc3_dabs_mean: list[float] = []
    fc3_dabs_std: list[float] = []

    for epoch in range(epochs):
        running_loss = 0.0

        model.train()

        for i, (inputs, labels) in enumerate(loader):
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            fc1_weight = model.fc1.weight.clone()
            fc2_weight = model.fc2.weight.clone()
            fc3_weight = model.fc3.weight.clone()

            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                (mean1, std1) = (dabs_mean_std(fc1_weight, model.fc1.weight))
                fc1_dabs_mean.append(mean1)
                fc1_dabs_std.append(std1)

                (mean2, std2) = (dabs_mean_std(fc2_weight, model.fc2.weight))
                fc2_dabs_mean.append(mean2)
                fc2_dabs_std.append(std2)

                (mean3, std3) = (dabs_mean_std(fc3_weight, model.fc3.weight))
                fc3_dabs_mean.append(mean3)
                fc3_dabs_std.append(std3)

                print(f'E{epoch + 1}/{epochs} S{i + 1}/{len(loader)} Loss={running_loss / 100:.4f}')
                running_losses.append(running_loss)
                running_loss = 0.0

    X = range(len(fc1_dabs_mean))

    plt.figure(figsize=(10, 6))

    def plot_mean_std(label, color, series_mean, series_std):
        plt.plot(X, series_mean, label=label, color=color)

        plt.fill_between(
            X,
            [m - s for m, s in zip(series_mean, series_std)],
            [m + s for m, s in zip(series_mean, series_std)],
            color=color,
            alpha=0.2
        )

    plot_mean_std('FC1 Mean', 'blue', fc1_dabs_mean, fc1_dabs_std)
    plot_mean_std('FC2 Mean', 'green', fc2_dabs_mean, fc2_dabs_std)
    plot_mean_std('FC3 Mean', 'red', fc3_dabs_mean, fc3_dabs_std)

    plt.plot(X, running_losses, label='Running loss', color='orange')

    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Mean Values with Standard Deviation for FC Layers')
    plt.legend()

    plt.show()


# model1 = DigitRecognizer()

# train(
#     model=model1,
#     loader=dataset.train_loader,
#     epochs=5,
#     criterion=nn.CrossEntropyLoss(),
#     optimizer=optim.Adam(
#         model1.parameters(),
#         lr=0.001
#     )
# )

# torch.save(model1.state_dict(), 'digit_recognizer1.pth')

model2 = DigitRecognizerCustomBackward()

train(
    model=model2,
    loader=dataset.train_loader,
    epochs=5,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(
        model2.parameters(),
        lr=0.001
    )
)

torch.save(model2.state_dict(), 'digit_recognizer2.pth')

