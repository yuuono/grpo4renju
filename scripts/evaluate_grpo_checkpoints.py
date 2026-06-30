from __future__ import annotations

import argparse
import csv
import gc
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_transformer.model import RenjuTransformerModel
from renju_transformer.rules import infer_player, winner_after_move
from renju_transformer.tokenizer import RenjuTokenizer
from renju_transformer.utils import select_device, set_seed


BLACK = 1
WHITE = 2
BOARD_CELLS = 225
BOARD_SIZE = 15


@dataclass(slots=True)
class GameResult:
    winner: int | None
    a_is_black: bool
    plies: int
    final_board: list[int]


@dataclass(slots=True)
class EvaluationSummary:
    checkpoint: Path
    num_games: int
    a_wins: int
    b_wins: int
    draws: int
    a_wins_black: int
    a_wins_white: int
    b_wins_black: int
    b_wins_white: int
    avg_plies: float

    @property
    def a_win_rate(self) -> float:
        return self.a_wins / self.num_games if self.num_games else 0.0

    @property
    def b_win_rate(self) -> float:
        return self.b_wins / self.num_games if self.num_games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.num_games if self.num_games else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate models in artifacts/grpo_checkpoints against a fixed pretrained model.",
    )
    parser.add_argument("--model-a-path", default="models/pretrained.pt")
    parser.add_argument("--checkpoint-dir", default="artifacts/grpo_checkpoints")
    parser.add_argument("--pattern", default="step_*_grpo_model.pt")
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_model(checkpoint_path: str | Path, device: torch.device) -> RenjuTransformerModel:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise ValueError(f"Checkpoint {path} does not contain 'config'.")

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
    model.to(device)
    model.eval()
    return model


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step_(\d+)_grpo_model\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def find_checkpoints(checkpoint_dir: Path, pattern: str) -> list[Path]:
    checkpoints = sorted(
        checkpoint_dir.glob(pattern),
        key=lambda path: (checkpoint_step(path), path.name),
    )
    return [path for path in checkpoints if path.is_file()]


def board_to_text(board: list[int]) -> str:
    symbols = {0: ".", BLACK: "X", WHITE: "O"}
    lines = ["    " + " ".join(f"{col:02d}" for col in range(BOARD_SIZE))]
    for row in range(BOARD_SIZE):
        cells = []
        for col in range(BOARD_SIZE):
            cells.append(f" {symbols[board[row * BOARD_SIZE + col]]}")
        lines.append(f"{row:02d} " + "".join(cells))
    return "\n".join(lines)


@torch.no_grad()
def select_move(
    model: RenjuTransformerModel,
    tokenizer: RenjuTokenizer,
    board: list[int],
    temperature: float,
    device: torch.device,
) -> int | None:
    legal_mask = tokenizer.legal_move_mask(board).to(device)
    if not bool(legal_mask.any()):
        return None

    input_ids = tokenizer.encode_input(board).unsqueeze(0).to(device)
    logits = model(input_ids).squeeze(0)
    masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
    if temperature == 0.0:
        return int(masked_logits.argmax().item())

    probs = torch.softmax(masked_logits / temperature, dim=-1)
    return int(torch.distributions.Categorical(probs=probs).sample().item())


def play_game(
    model_a: RenjuTransformerModel,
    model_b: RenjuTransformerModel,
    tokenizer: RenjuTokenizer,
    game_index: int,
    temperature: float,
    device: torch.device,
) -> GameResult:
    a_is_black = game_index % 2 == 1
    board = [0] * BOARD_CELLS
    winner: int | None = None
    plies = 0

    for _ply in range(1, BOARD_CELLS + 1):
        current_player = infer_player(board)
        current_is_a = (current_player == BLACK) if a_is_black else (current_player == WHITE)
        current_model = model_a if current_is_a else model_b
        move = select_move(current_model, tokenizer, board, temperature, device)
        if move is None:
            break

        board[move] = current_player
        plies += 1
        winner = winner_after_move(board, move, current_player)
        if winner is not None:
            break

    return GameResult(
        winner=winner,
        a_is_black=a_is_black,
        plies=plies,
        final_board=board,
    )


def is_a_winner(result: GameResult) -> bool:
    if result.winner is None:
        return False
    return (result.winner == BLACK) if result.a_is_black else (result.winner == WHITE)


def evaluate_checkpoint(
    checkpoint: Path,
    model_a: RenjuTransformerModel,
    tokenizer: RenjuTokenizer,
    args: argparse.Namespace,
    device: torch.device,
    a_win_log,
) -> EvaluationSummary:
    model_b = load_model(checkpoint, device)
    stats = {
        "a_wins_black": 0,
        "a_wins_white": 0,
        "b_wins_black": 0,
        "b_wins_white": 0,
        "draws": 0,
        "total_plies": 0,
    }

    for game_index in range(1, int(args.num_games) + 1):
        result = play_game(
            model_a=model_a,
            model_b=model_b,
            tokenizer=tokenizer,
            game_index=game_index,
            temperature=float(args.temperature),
            device=device,
        )
        stats["total_plies"] += result.plies

        if result.winner is None:
            stats["draws"] += 1
            continue

        if is_a_winner(result):
            if result.a_is_black:
                stats["a_wins_black"] += 1
                a_color = "black"
            else:
                stats["a_wins_white"] += 1
                a_color = "white"
            a_win_log.write(
                f"checkpoint={checkpoint}\n"
                f"game={game_index}\n"
                f"a_color={a_color}\n"
                f"winner=model_a\n"
                f"plies={result.plies}\n"
                f"final_board_csv={','.join(str(cell) for cell in result.final_board)}\n"
                f"{board_to_text(result.final_board)}\n"
                f"{'=' * 60}\n"
            )
            a_win_log.flush()
            continue

        if result.a_is_black:
            stats["b_wins_white"] += 1
        else:
            stats["b_wins_black"] += 1

    del model_b
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    a_wins = stats["a_wins_black"] + stats["a_wins_white"]
    b_wins = stats["b_wins_black"] + stats["b_wins_white"]
    return EvaluationSummary(
        checkpoint=checkpoint,
        num_games=int(args.num_games),
        a_wins=a_wins,
        b_wins=b_wins,
        draws=stats["draws"],
        a_wins_black=stats["a_wins_black"],
        a_wins_white=stats["a_wins_white"],
        b_wins_black=stats["b_wins_black"],
        b_wins_white=stats["b_wins_white"],
        avg_plies=stats["total_plies"] / int(args.num_games),
    )


def write_summary_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "step",
                "num_games",
                "a_win_rate",
                "b_win_rate",
                "draw_rate",
                "a_wins",
                "b_wins",
                "draws",
                "a_wins_black",
                "a_wins_white",
                "b_wins_black",
                "b_wins_white",
                "avg_plies",
            ]
        )


def append_summary(path: Path, summary: EvaluationSummary) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                str(summary.checkpoint),
                checkpoint_step(summary.checkpoint),
                summary.num_games,
                f"{summary.a_win_rate:.6f}",
                f"{summary.b_win_rate:.6f}",
                f"{summary.draw_rate:.6f}",
                summary.a_wins,
                summary.b_wins,
                summary.draws,
                summary.a_wins_black,
                summary.a_wins_white,
                summary.b_wins_black,
                summary.b_wins_white,
                f"{summary.avg_plies:.2f}",
            ]
        )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = select_device(str(args.device))
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = find_checkpoints(checkpoint_dir, str(args.pattern))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matched {args.pattern!r} under {checkpoint_dir}."
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs" / "versus_sweep" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    a_win_log_path = output_dir / "model_a_wins_final_boards.log"
    write_summary_header(summary_path)

    tokenizer = RenjuTokenizer()
    model_a = load_model(args.model_a_path, device)
    print(f"model_a_path={args.model_a_path}")
    print(f"checkpoint_dir={checkpoint_dir}")
    print(f"matched_checkpoints={len(checkpoints)}")
    print(f"summary_csv={summary_path}")
    print(f"a_win_log={a_win_log_path}")

    with a_win_log_path.open("w", encoding="utf-8") as a_win_log:
        a_win_log.write(f"model_a_path={args.model_a_path}\n")
        a_win_log.write(f"num_games={args.num_games}\n")
        a_win_log.write(f"temperature={args.temperature}\n")
        a_win_log.write(f"{'=' * 60}\n")

        for checkpoint in checkpoints:
            summary = evaluate_checkpoint(
                checkpoint=checkpoint,
                model_a=model_a,
                tokenizer=tokenizer,
                args=args,
                device=device,
                a_win_log=a_win_log,
            )
            append_summary(summary_path, summary)
            print(
                f"{checkpoint.name}: "
                f"A={summary.a_win_rate * 100:.1f}% "
                f"B={summary.b_win_rate * 100:.1f}% "
                f"Draw={summary.draw_rate * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
