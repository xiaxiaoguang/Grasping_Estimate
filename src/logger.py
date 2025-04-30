import os
from dataclasses import asdict
from typing import Dict
import warnings
import yaml
import torch
from datetime import datetime
import csv


from .config import Config
from .path import (
    get_exp_dir,
    get_log_path,
    get_exp_config_path,
    get_checkpoint_dir,
    get_checkpoint_path,
)
# from .git import save_code_and_git
def get_csv_log_path(exp_name): return os.path.join(get_log_path(exp_name), "metrics.csv")

class Logger:
    exp_name: str

    def __init__(self, config):
        self.exp_name = config.exp_name
        self.enable_logging = self.exp_name != "debug"

        if not self.enable_logging:
            warnings.warn("exp_name is 'debug', logging is disabled.")

        # Create necessary directories
        os.makedirs(get_exp_dir(self.exp_name), exist_ok=True)
        os.makedirs(get_log_path(self.exp_name), exist_ok=True)
        os.makedirs(get_checkpoint_dir(self.exp_name), exist_ok=True)

        # Save config
        with open(get_exp_config_path(self.exp_name), "w") as f:
            yaml.dump(asdict(config), f)

        # Initialize CSV logging file
        if self.enable_logging:
            self.log_file = get_csv_log_path(self.exp_name)
            if not os.path.exists(self.log_file):
                with open(self.log_file, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "mode", "key", "value", "timestamp"])

    def log(self, dic: Dict[str, float], mode: str, step: int):
        if not self.enable_logging:
            return

        timestamp = datetime.now().isoformat()
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            for k, v in dic.items():
                writer.writerow([step, mode, k, v, timestamp])

    def save(self, dic: dict, step: int):
        torch.save(dic, get_checkpoint_path(self.exp_name, step))
