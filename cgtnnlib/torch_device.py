## Project-global torch device

import torch

if torch.cuda.is_available():
    TORCH_DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    TORCH_DEVICE = 'mps'
else:
    TORCH_DEVICE = 'cpu'