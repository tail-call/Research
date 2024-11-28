## Report v.0.2
## Created at Tue 26 Nov 2024
## Modified at Thu 28 Nov 2024
## v.0.2 - .path, .filename properties; .see() method

from datetime import datetime
import os
import json

import torch
import numpy as np

def now_isoformat() -> str:
    return datetime.now().isoformat()

def see_value(value) -> str:
    if isinstance(value, (list, np.ndarray, torch.Tensor)):
        # Get the number of items in the list, numpy array, or tensor
        num_items = len(value) if isinstance(value, list) else value.size
        return f"{type(value).__name__}({num_items} items)"
    elif isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, (int, float, bool)):
        return str(value)
    else:
        return f"{type(value).__name__}(...)"

class Report:
    dir: str
    data: dict[str, dict | list | str] = {
        'started': now_isoformat()
    }
    
    def __init__(self, dir: str):
        self.dir = dir
    
    @property 
    def filename(self):
        return 'report.json'
    
    @property
    def path(self):
        return os.path.join(self.dir, self.filename)

    def append(self, key: str, data: dict | list):
        self.data[key] = data

    def save(self):
        self.append('saved', now_isoformat())
        with open(self.path, 'w') as file:
            json.dump(self.data, file, indent=4)
        print(f"Report saved to {self.path}.")
    
    def see(self):
        title = f"Report {self.path}"
        print(title)
        print(''.join(['=' for _ in range(0, len(title))]))
        
        for key in self.data:
            print(f"{key}: {see_value(self.data[key])}")