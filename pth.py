import torch

import matplotlib.pyplot as plt
import matplotlib.colors as colors

path = 'pth/model-1B-c-P0.05_N7.pth'
layer = 'fc2.weight'

model_weights = torch.load(path)

for key, value in model_weights.items():
    print(f"Key: {key}, Shape: {value.shape}")


fc2w: torch.Tensor = model_weights[layer]
min_val = fc2w.min()
max_val = fc2w.max()

fc2w_norm = (fc2w - min_val) / (max_val - min_val)

cmap = colors.LinearSegmentedColormap.from_list(
    'custom_cmap', [(0, 'black'), (1, 'red')]
)

fc2w_norm
plt.imshow(fc2w_norm, cmap=cmap, vmin=0, vmax=1)
plt.colorbar(label=F"Normalized value, domain of source: ({min_val:0.3f}, {max_val:0.3f})")
plt.title(f"{path}: {layer}")
plt.show()

dir(plt)