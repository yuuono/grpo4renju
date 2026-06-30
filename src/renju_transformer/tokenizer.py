"""Renju board-state CSV rows tokenizer."""

# 型アノテーションの遅延評価を有効化します。
from __future__ import annotations

# dataclass デコレータを使用するために読み込みます。
from dataclasses import dataclass

# テンソル作成と dtype 指定のために PyTorch を読み込みます。
import torch

# 盤面の総セル数と一辺サイズの定数を読み込みます。
from .rules import BOARD_CELLS, BOARD_SIZE
# TSS 実装の合法手マスク関数を読み込みます。
from .rules_tss import legal_move_mask


# トークナイザ設定を保持するデータクラスを定義します。
@dataclass(slots=True)
class RenjuTokenizer:
    # 区切りトークン ID を既定値 228 で定義します。
    sep_token_id: int = 228 # 区切りトークンのid
    # move_id から盤面インデックスへ変換するオフセットを定義します。
    move_id_offset: int = 3 # 指し手id-オフセット=置く場所のインデックス

    # 呼び出し時に関数形式ではなく属性形式で参照できるようにします。
    @property # プロパティは変数みたいなもの。盤面のマス数
    # 盤面総セル数を返します。
    def board_cells(self) -> int:
        # BOARD_CELLS 定数をそのまま返します。
        return BOARD_CELLS

    # 呼び出し時に関数形式ではなく属性形式で参照できるようにします。
    @property # 盤面の一辺のサイズ
    # 盤面の一辺サイズを返します。
    def board_size(self) -> int:
        # BOARD_SIZE 定数をそのまま返します。
        return BOARD_SIZE

    # 呼び出し時に関数形式ではなく属性形式で参照できるようにします。
    @property # モデルの入力トークンの長さ。盤面の大きさ+セパレータトークン
    # モデル入力のトークン長を返します。
    def input_length(self) -> int:
        # 盤面セル数に SEP 1 個を足した長さを返します。
        return self.board_cells + 1

    # 呼び出し時に関数形式ではなく属性形式で参照できるようにします。
    @property # 予測指し手の総数。盤面のマス数と同じ
    # 分類ラベル総数を返します。
    def num_labels(self) -> int:
        # 出力ラベル数は盤面セル数と同じです。
        return self.board_cells

    # 呼び出し時に関数形式ではなく属性形式で参照できるようにします。
    @property # 語彙サイズ。使う文字の数。
    # 語彙サイズを返します。
    def vocab_size(self) -> int:
        # ID が 0 始まり前提なので最大 ID + 1 を返します。
        return self.sep_token_id + 1

    # リストの長さが225であること、かつ含まれる値が0か1か2のどれかであることを確認。
    # 盤面配列の長さと値域を検証します。
    def validate_board(self, board: list[int]) -> None:
        # セル数が想定と異なる場合は例外を送出します。
        if len(board) != self.board_cells:
            # 実際の長さを含めてエラーメッセージを作成します。
            raise ValueError(f"Expected {self.board_cells} cells, got {len(board)}.")
        # 0/1/2 以外の不正値を抽出します。
        invalid = [cell for cell in board if cell not in (0, 1, 2)]
        # 不正値が 1 つでもあれば例外を送出します。
        if invalid:
            # 重複除去して整列した不正値一覧を通知します。
            raise ValueError(f"Board contains invalid tokens: {sorted(set(invalid))}")

    # 盤面リストが与えられたら、セパレータトークンをくっつけ、テンソルにして返す
    # 盤面をモデル入力テンソルへ変換します。
    def encode_input(self, board: list[int]) -> torch.Tensor:
        # 入力盤面の妥当性を先に検証します。
        self.validate_board(board)
        # 盤面末尾に SEP トークンを追加したトークン列を作成します。
        tokens = board + [self.sep_token_id]
        # long 型テンソルとして返します。
        return torch.tensor(tokens, dtype=torch.long)

    # 指し手idを、インデックスに変換。3引くだけ。
    # move_id を分類ラベル index に変換します。
    def encode_label(self, move_id: int) -> int:
        # オフセット分だけ引いて 0 始まりラベルへ変換します。
        label = move_id - self.move_id_offset
        # ラベル範囲外なら例外を送出します。
        if not 0 <= label < self.num_labels:
            # 不正な move_id を含めて通知します。
            raise ValueError(f"Move id {move_id} is out of range.")
        # 検証済みラベルを返します。
        return label

    # インデックスを、指し手idに変換。3足すだけ
    # 分類ラベル index を move_id に戻します。
    def decode_label(self, label: int) -> int:
        # ラベル範囲外なら例外を送出します。
        if not 0 <= label < self.num_labels:
            # 不正な label 値を含めて通知します。
            raise ValueError(f"Label {label} is out of range.")
        # オフセットを戻して move_id を返します。
        return label + self.move_id_offset

    # 上記と同じ。使う場面が違うだけ。
    # move_id から盤面インデックスへ変換する別名メソッドです。
    def move_id_to_index(self, move_id: int) -> int:
        # encode_label をそのまま呼び出します。
        return self.encode_label(move_id)

    # 上記と同じ。使う場面が違うだけ。
    # 盤面インデックスから move_id へ変換する別名メソッドです。
    def index_to_move_id(self, index: int) -> int:
        # decode_label をそのまま呼び出します。
        return self.decode_label(index)

    # CSV 1 行 (盤面 + SEP + move_id) をモデル入出力へ変換します。
    def encode_csv_row(self, row: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        # 期待列数 (225 + 1 + 1) を計算します。
        expected_length = self.board_cells + 2 # 盤面＋セパレータ＋指し手
        # 列数が想定外なら例外を送出します。
        if len(row) != expected_length:
            # 実際の列数を含めてエラーを通知します。
            raise ValueError(f"Expected {expected_length} columns, got {len(row)}.")
        # 先頭 225 列を盤面として切り出します。
        board = row[: self.board_cells]
        # 226 列目を SEP として取得します。
        sep = row[self.board_cells]
        # SEP 値が設定値と違えば例外を送出します。
        if sep != self.sep_token_id:
            # 実際の SEP 値を含めて通知します。
            raise ValueError(f"Expected SEP token {self.sep_token_id}, got {sep}.")
        # 最終列を move_id として取得します。
        move_id = row[-1]
        # 盤面を入力テンソルへ変換します。
        input_ids = self.encode_input(board)
        # move_id をラベル化して long テンソル化します。
        label = torch.tensor(self.encode_label(move_id), dtype=torch.long)
        # 入力テンソルと教師ラベルを返します。
        return input_ids, label

    # 盤面 CSV 文字列を整数盤面リストへ変換します。
    def parse_board_csv(self, board_csv: str) -> list[int]:
        # カンマ分割し、空要素を除外して前後空白を除去します。
        values = [item.strip() for item in board_csv.split(",") if item.strip()]
        # 各要素を整数へ変換します。
        board = [int(value) for value in values]
        # 変換後の盤面妥当性を検証します。
        self.validate_board(board)
        # 検証済み盤面を返します。
        return board

    # boolのlegalマスクを作成
    # 盤面から合法手 bool マスクを PyTorch テンソルで返します。
    def legal_move_mask(self, board: list[int]) -> torch.Tensor:
        # 入力盤面の妥当性を先に検証します。
        self.validate_board(board)
        # TSS 実装で合法手マスク (bool list) を取得します。
        mask = legal_move_mask(board)
        # bool テンソルへ変換して返します。
        return torch.tensor(mask, dtype=torch.bool)
