#!/usr/bin/env python3
"""Hydra entrypoint for Renju Transformer training and inference."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_transformer.grpo import train_grpo  # noqa: E402
from renju_transformer.predict import predict_from_checkpoint  # noqa: E402
from renju_transformer.train import train_model  # noqa: E402
from renju_transformer.utils import configure_run_logging  # noqa: E402


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = configure_run_logging()
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    print(f"run_output_dir={output_dir}", flush=True)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.mode == "train":
        train_model(cfg)
        return
    if cfg.mode == "grpo":
        train_grpo(cfg)
        return
    if cfg.mode == "predict":
        predict_from_checkpoint(cfg)
        return
    raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
