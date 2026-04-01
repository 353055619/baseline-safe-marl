"""
实验日志：TensorBoard + CSV
"""
import csv, os
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentLogger:
    def __init__(self, exp_dir: Path, algo: str, fields: List[str]):
        self.exp_dir = Path(exp_dir)
        self.algo = algo
        self.fields = fields
        self.csv_path = self.exp_dir / f"{algo}.csv"
        self._file_is_new = not self.csv_path.exists()

    def log(self, row: Dict[str, float]):
        write_header = self._file_is_new
        self._file_is_new = False
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if write_header:
                w.writeheader()
            w.writerow(row)

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @csv_path.setter
    def csv_path(self, path):
        self._csv_path = path
