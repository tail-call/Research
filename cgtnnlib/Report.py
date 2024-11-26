from datetime import datetime
import os
import json

def now_isoformat() -> str:
    return datetime.now().isoformat()

class Report:
    dir: str
    data: dict[str, dict | list | str] = {
        'started': now_isoformat()
    }
    
    def __init__(self, dir: str):
        self.dir = dir

    def append(self, key: str, data: dict | list):
        self.data[key] = data

    def save(self):
        self.append('saved', now_isoformat())
        path = os.path.join(self.dir, 'report.json')
        with open(path, 'w') as file:
            json.dump(self.data, file, indent=4)
        print(f"Report saved to {path}.")