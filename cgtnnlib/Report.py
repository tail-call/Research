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
    
    @property
    def path(self):
        return os.path.join(self.dir, 'report.json')

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
            print(f"{key}: {type(self.data[key])}")