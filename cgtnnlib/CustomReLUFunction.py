import torch

class MockCtx:
    @property
    def saved_tensors(self):
        return (self.input, self.tensor)
        
    def save_for_backward(self, input, tensor):
        self.input = input
        self.tensor = tensor

class CustomReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, p: float):
        ctx.save_for_backward(input, torch.scalar_tensor(p))
        # ReLU is happening in clamp
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, p = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0

        # У матриц ось 0 это Y (Добавляем аргумент device=grad_output.device для указания устройства для создания тензора grad_input)
        # XXX 2. grad_input.size(0) на grad_input.size(1)
        bernoulli_mask = torch.bernoulli(torch.ones(grad_input.size(0), device=grad_output.device) * (1 - p.item()))
        # YYY 1. Попробовать запустить без деления
        diagonal_mask = torch.diag(bernoulli_mask) # / (1 - p.item()+1e-5)

        # Перемещаем diagonal_mask на Cuda
        diagonal_mask = diagonal_mask.to(grad_output.device)

        # Multiply grad_input with the diagonal matrix
        # XXX 2. Заменить на grad_input @ diagonal_mask
        grad_input = diagonal_mask @ grad_input

        return grad_input, None
    
