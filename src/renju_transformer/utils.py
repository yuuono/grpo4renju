"""Shared utility helpers."""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import TextIO

import mlflow
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(configured_device: str) -> torch.device:
    if configured_device != "auto":
        return torch.device(configured_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_run_output_dir() -> Path:
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except ValueError:
        output_dir = "outputs/manual"
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


class TeeStream:
    def __init__(self, primary: TextIO, log_file: TextIO) -> None:
        self.primary = primary
        self.log_file = log_file
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, text: str) -> int:
        written = self.primary.write(text)
        self.log_file.write(text)
        return written

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()


def configure_run_logging() -> Path:
    output_dir = get_run_output_dir()
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not getattr(configure_run_logging, "_configured", False):
        stdout_file = (logs_dir / "stdout.log").open("a", encoding="utf-8")
        stderr_file = (logs_dir / "stderr.log").open("a", encoding="utf-8")
        sys.stdout = TeeStream(sys.stdout, stdout_file)  # type: ignore[assignment]
        sys.stderr = TeeStream(sys.stderr, stderr_file)  # type: ignore[assignment]

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[
                logging.FileHandler(logs_dir / "run.log", encoding="utf-8"),
                logging.StreamHandler(sys.stderr),
            ],
            force=True,
        )
        configure_run_logging._configured = True  # type: ignore[attr-defined]

    logging.getLogger(__name__).info("run_output_dir=%s", output_dir)
    return output_dir


class JsonlCsvLogger:
    def __init__(self, jsonl_path: Path, csv_path: Path, fieldnames: list[str]) -> None:
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict[str, object]) -> None:
        normalized = {field: row.get(field) for field in self.fieldnames}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")

        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(normalized)


def flatten_config(cfg: DictConfig, prefix: str = "") -> dict[str, str | int | float | bool]:
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        return {}

    flattened: dict[str, str | int | float | bool] = {}
    for key, value in data.items():
        composite_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, composite_key))
        elif isinstance(value, (str, int, float, bool)):
            flattened[composite_key] = value
        elif value is None:
            flattened[composite_key] = "null"
        else:
            flattened[composite_key] = str(value)
    return flattened


def flatten_dict(data: dict, prefix: str) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in data.items():
        composite_key = f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, composite_key))
        elif isinstance(value, (str, int, float, bool)):
            flattened[composite_key] = value
        elif value is None:
            flattened[composite_key] = "null"
        else:
            flattened[composite_key] = str(value)
    return flattened


def ensure_mlflow_experiment(tracking_uri: str, experiment_name: str, artifact_root: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return

    artifact_uri = Path(artifact_root).resolve().as_uri()
    client.create_experiment(experiment_name, artifact_location=artifact_uri)
