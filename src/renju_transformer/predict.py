"""Inference helpers for Renju next-move prediction."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from .model import RenjuTransformerModel
from .tokenizer import RenjuTokenizer
from .utils import select_device


def _load_board(cfg: DictConfig, tokenizer: RenjuTokenizer) -> list[int]:
    if cfg.predict.board_csv:
        return tokenizer.parse_board_csv(cfg.predict.board_csv)
    if cfg.predict.board_path:
        content = Path(cfg.predict.board_path).read_text(encoding="utf-8").strip()
        return tokenizer.parse_board_csv(content)
    raise ValueError("Set predict.board_csv or predict.board_path for inference.")


def _build_model_from_checkpoint(cfg: DictConfig, checkpoint: dict) -> RenjuTransformerModel:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        model_cfg = cfg.model
    else:
        model_cfg = checkpoint_config["model"]

    model = RenjuTransformerModel(
        vocab_size=model_cfg["token_vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
        norm_first=model_cfg["norm_first"],
        num_move_labels=model_cfg["num_move_labels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def predict_from_checkpoint(cfg: DictConfig) -> None:
    if not cfg.predict.checkpoint_path:
        raise ValueError("Set predict.checkpoint_path to a saved checkpoint.")

    tokenizer = RenjuTokenizer(
        sep_token_id=cfg.data.sep_token_id,
        move_id_offset=cfg.data.move_id_offset,
    )
    board = _load_board(cfg, tokenizer)
    device = select_device(cfg.train.device)
    checkpoint = torch.load(cfg.predict.checkpoint_path, map_location=device, weights_only=False)
    model = _build_model_from_checkpoint(cfg, checkpoint).to(device)
    model.eval()

    input_ids = tokenizer.encode_input(board).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).squeeze(0)

    if cfg.predict.apply_legal_mask:
        legal_mask = tokenizer.legal_move_mask(board).to(device)
        if not bool(legal_mask.any()):
            raise ValueError("No legal moves available for the provided board.")
        logits = logits.masked_fill(~legal_mask, float("-inf"))

    probabilities = torch.softmax(logits, dim=-1)
    top_k = min(cfg.predict.top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=top_k)

    best_move_id = tokenizer.index_to_move_id(indices[0].item())
    print(f"predicted_move_id={best_move_id}")
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist(), strict=True), start=1):
        print(f"top{rank}_move_id={tokenizer.index_to_move_id(index)} prob={value:.6f}")
