# TSS 実装を ctypes で呼び出すための合法手ラッパーモジュールです。
"""ctypes wrapper for TSS-backed Renju legality helpers."""

# 型アノテーションの遅延評価を有効化します。
from __future__ import annotations

# C 共有ライブラリを Python から扱うために使用します。
import ctypes
# 環境変数の参照に使用します。
import os
# 実行プラットフォーム判定に使用します。
import sys
# ファイルパスを OS 非依存で扱うために使用します。
from pathlib import Path

# Python 側ルール実装の定数・フォールバック関数を読み込みます。
from .rules import (
    # 黒石を表す定数です。
    BLACK,
    # 盤面セル総数 (225) の定数です。
    BOARD_CELLS,
    # 白石を表す定数です。
    WHITE,
    # 盤面から手番を推定する関数です。
    infer_player,
    # Python 実装の黒禁手判定関数 (フォールバック用) です。
    is_forbidden_for_black as python_is_forbidden_for_black,
    # Python 実装の合法手マスク関数 (フォールバック用) です。
    legal_move_mask as python_legal_move_mask,
)


# TSS 共有ライブラリ探索候補のデフォルト一覧を返します。
def _default_library_candidates() -> list[Path]:
    # OS に応じてライブラリ名を切り替えます。
    library_name = "tss.dll" if sys.platform == "win32" else "tss.so"
    # 探索候補パスを格納するリストを初期化します。
    candidates: list[Path] = []

    # 環境変数による明示指定パスを取得します。
    env_path = os.environ.get("RENJU_TSS_LIBRARY_PATH")
    # 指定がある場合は候補へ追加します。
    if env_path:
        # `~` を展開して実パス候補として登録します。
        candidates.append(Path(env_path).expanduser())

    # カレントディレクトリ相対のライブラリ名を候補へ追加します。
    candidates.append(Path(library_name))
    # プロジェクトルート直下想定のライブラリも候補へ追加します。
    candidates.append(Path(__file__).resolve().parents[2] / library_name)
    # 組み立てた候補一覧を返します。
    return candidates


# TSS 共有ライブラリをロードして合法手 API を提供するクライアントです。
class TssRulesClient:
    # クラスの役割説明です。
    """Calls legal-move helpers exported by tss.so."""

    # クライアント初期化時にライブラリ読込を実行します。
    def __init__(self, library_path: str | Path | None = None, *, required: bool = False) -> None:
        # 明示ライブラリパス指定を保持します。
        self.library_path = library_path
        # ロード失敗を例外化する厳格モードかどうかを保持します。
        self.required = required
        # ctypes でロードしたライブラリ本体の参照を初期化します。
        self._library: ctypes.CDLL | None = None
        # 黒禁手判定関数ポインタを初期化します。
        self._is_forbidden = None
        # 単一盤面の合法手マスク関数ポインタを初期化します。
        self._legal_mask = None
        # バッチ盤面の合法手マスク関数ポインタを初期化します。
        self._legal_mask_batch = None
        # 初期化時にライブラリ読込を試行します。
        self._load()

    # 外部から利用可能状態を属性で参照できるようにします。
    @property
    # 必須 API がロード済みかどうかを返します。
    def available(self) -> bool:
        # 単体 API とバッチ API の両方があるときのみ利用可能です。
        return self._legal_mask is not None and self._legal_mask_batch is not None

    # 黒番で指定手が禁手かどうかを判定します。
    def is_forbidden_for_black(self, board: list[int], move: int) -> bool:
        # 入力盤面の妥当性を先に検証します。
        self._validate_board(board)
        # TSS 側関数が未ロードの場合の分岐です。
        if self._is_forbidden is None:
            # required=True ならフォールバックせず例外を送出します。
            if self.required:
                # 必須シンボル欠落を明示します。
                raise RuntimeError("tss_is_forbidden_for_black is not available.")
            # required=False なら Python 実装へフォールバックします。
            return python_is_forbidden_for_black(board, move)

        # Python リストを C 側に渡せる int 配列へ変換します。
        board_array = (ctypes.c_int * BOARD_CELLS)(*board)
        # C 側戻り値を bool に変換して返します。
        return bool(self._is_forbidden(board_array, int(move)))

    # 単一盤面の合法手マスクを取得します。
    def legal_move_mask(self, board: list[int], player: int | None = None) -> list[bool]:
        # 入力盤面の妥当性を先に検証します。
        self._validate_board(board)
        # TSS 側関数が未ロードの場合の分岐です。
        if self._legal_mask is None:
            # required=True ならフォールバックせず例外を送出します。
            if self.required:
                # 必須シンボル欠落を明示します。
                raise RuntimeError("tss_legal_move_mask is not available.")
            # required=False なら Python 実装へフォールバックします。
            return python_legal_move_mask(board)

        # プレイヤー未指定時は盤面から手番を推定し、指定時は int 化して使います。
        resolved_player = infer_player(board) if player is None else int(player)
        # Python リストを C 側に渡せる int 配列へ変換します。
        board_array = (ctypes.c_int * BOARD_CELLS)(*board)
        # C 側が書き込む合法手マスク配列を確保します。
        mask_array = (ctypes.c_int * BOARD_CELLS)()
        # C 側 API を呼び出し、ステータスコードを受け取ります。
        status = self._legal_mask(board_array, resolved_player, mask_array)
        # 非 0 ステータスは失敗扱いです。
        if status != 0:
            # required=True なら即例外化します。
            if self.required:
                # 失敗コードを含めて例外を送出します。
                raise RuntimeError(f"tss_legal_move_mask returned error code {status}.")
            # required=False なら Python 実装へフォールバックします。
            return python_legal_move_mask(board)
        # C 側 0/1 配列を bool リストへ変換して返します。
        return [bool(value) for value in mask_array]

    # 複数盤面をまとめて合法手マスク化します。
    def legal_move_masks(
        # インスタンスメソッドの self です。
        self,
        # 対象盤面のリストです。
        boards: list[list[int]],
        # 各盤面の手番リスト (省略時は推定) です。
        players: list[int] | None = None,
    ) -> list[list[bool]]:
        # 空入力なら空結果を返して終了します。
        if not boards:
            return []
        # すべての盤面について妥当性を検証します。
        for board in boards:
            # 各盤面の長さ・値域をチェックします。
            self._validate_board(board)

        # バッチ API が未ロードの場合の分岐です。
        if self._legal_mask_batch is None:
            # required=True ならフォールバックせず例外を送出します。
            if self.required:
                # 必須シンボル欠落を明示します。
                raise RuntimeError("tss_legal_move_mask_batch is not available.")
            # required=False なら盤面ごとに Python 実装で計算します。
            return [python_legal_move_mask(board) for board in boards]

        # players 指定があればそれを使い、なければ盤面ごとに手番推定します。
        resolved_players = players if players is not None else [infer_player(board) for board in boards]
        # players と boards の長さ不一致は入力エラーです。
        if len(resolved_players) != len(boards):
            # 対応関係が崩れるため例外を送出します。
            raise ValueError("players length must match boards length.")

        # 2 次元 boards を 1 次元配列へ平坦化します。
        flat_boards = [cell for board in boards for cell in board]
        # 平坦化した盤面を C 側 int 配列へ変換します。
        board_array = (ctypes.c_int * len(flat_boards))(*flat_boards)
        # players を C 側 int 配列へ変換します。
        player_array = (ctypes.c_int * len(resolved_players))(*[int(player) for player in resolved_players])
        # 出力用の (盤面数 x 225) 配列を確保します。
        mask_array = (ctypes.c_int * (len(boards) * BOARD_CELLS))()

        # バッチ API を呼び出し、ステータスコードを受け取ります。
        status = self._legal_mask_batch(board_array, player_array, len(boards), mask_array)
        # 非 0 ステータスは失敗扱いです。
        if status != 0:
            # required=True なら即例外化します。
            if self.required:
                # 失敗コードを含めて例外を送出します。
                raise RuntimeError(f"tss_legal_move_mask_batch returned error code {status}.")
            # required=False なら盤面ごとに Python 実装へフォールバックします。
            return [python_legal_move_mask(board) for board in boards]

        # 返却用の 2 次元 bool リストを初期化します。
        masks: list[list[bool]] = []
        # 各盤面ごとにマスクを切り出して復元します。
        for board_index in range(len(boards)):
            # この盤面の開始オフセットを計算します。
            offset = board_index * BOARD_CELLS
            # 225 要素を bool 化して 1 盤面分として追加します。
            masks.append([bool(mask_array[offset + move]) for move in range(BOARD_CELLS)])
        # 復元した 2 次元マスクを返します。
        return masks

    # 候補パスから TSS 共有ライブラリをロードします。
    def _load(self) -> None:
        # 明示パス指定があればそれのみ、なければデフォルト候補を使用します。
        paths = [Path(self.library_path).expanduser()] if self.library_path else _default_library_candidates()
        # ロード失敗理由を蓄積する配列を初期化します。
        load_errors: list[str] = []

        # 候補パスを順に試します。
        for path in paths:
            # 絶対パスに解決します。
            resolved_path = path.resolve()
            # ファイルが存在しない候補はスキップします。
            if not resolved_path.exists():
                continue
            # 共有ライブラリ読込処理を例外捕捉付きで実行します。
            try:
                # 共有ライブラリをロードします。
                library = ctypes.CDLL(str(resolved_path))
                # 黒禁手判定シンボルを取得します (なければ None)。
                is_forbidden = getattr(library, "tss_is_forbidden_for_black", None)
                # 単体合法手マスクシンボルを取得します (なければ None)。
                legal_mask = getattr(library, "tss_legal_move_mask", None)
                # バッチ合法手マスクシンボルを取得します (なければ None)。
                legal_mask_batch = getattr(library, "tss_legal_move_mask_batch", None)
                # 必須シンボル欠落時は失敗理由を記録して次候補へ進みます。
                if is_forbidden is None or legal_mask is None or legal_mask_batch is None:
                    # どの候補がなぜ失敗したかを記録します。
                    load_errors.append(f"{resolved_path}: missing required legal-move symbols")
                    continue

                # is_forbidden の引数型を設定します。
                is_forbidden.argtypes = [
                    # 盤面配列ポインタです。
                    ctypes.POINTER(ctypes.c_int),
                    # 手位置インデックスです。
                    ctypes.c_int,
                ]
                # is_forbidden の戻り値型を設定します。
                is_forbidden.restype = ctypes.c_int

                # legal_mask の引数型を設定します。
                legal_mask.argtypes = [
                    # 盤面配列ポインタです。
                    ctypes.POINTER(ctypes.c_int),
                    # 手番プレイヤーです。
                    ctypes.c_int,
                    # 出力マスク配列ポインタです。
                    ctypes.POINTER(ctypes.c_int),
                ]
                # legal_mask の戻り値型を設定します。
                legal_mask.restype = ctypes.c_int

                # legal_mask_batch の引数型を設定します。
                legal_mask_batch.argtypes = [
                    # 平坦化盤面配列ポインタです。
                    ctypes.POINTER(ctypes.c_int),
                    # プレイヤー配列ポインタです。
                    ctypes.POINTER(ctypes.c_int),
                    # 盤面数です。
                    ctypes.c_int,
                    # 出力マスク配列ポインタです。
                    ctypes.POINTER(ctypes.c_int),
                ]
                # legal_mask_batch の戻り値型を設定します。
                legal_mask_batch.restype = ctypes.c_int

                # 実際にロードできたライブラリパスを保存します。
                self.library_path = resolved_path
                # ライブラリ参照を保存します。
                self._library = library
                # 黒禁手関数ポインタを保存します。
                self._is_forbidden = is_forbidden
                # 単体合法手関数ポインタを保存します。
                self._legal_mask = legal_mask
                # バッチ合法手関数ポインタを保存します。
                self._legal_mask_batch = legal_mask_batch
                # 1 つ成功したのでロード処理を終了します。
                return
            # OS レベルのロード例外を捕捉します。
            except OSError as exc:
                # 失敗理由を候補ごとに記録します。
                load_errors.append(f"{resolved_path}: {exc}")

        # required=True で最終的にロード失敗した場合は例外を送出します。
        if self.required:
            # 失敗理由があれば連結し、なければ既定メッセージを使用します。
            detail = "; ".join(load_errors) if load_errors else "no tss library found"
            # 失敗詳細付きで RuntimeError を送出します。
            raise RuntimeError(f"Failed to load TSS legality helpers: {detail}")

    # インスタンス不要で呼べる静的メソッドとして定義します。
    @staticmethod
    # 盤面配列の長さと値域を検証します。
    def _validate_board(board: list[int]) -> None:
        # セル数不一致は入力エラーです。
        if len(board) != BOARD_CELLS:
            # 期待値と実値を含む例外を送出します。
            raise ValueError(f"Expected {BOARD_CELLS} cells, got {len(board)}.")
        # 許容値 (0/BLACK/WHITE) 以外を抽出します。
        invalid = [cell for cell in board if cell not in (0, BLACK, WHITE)]
        # 不正値が存在すれば例外を送出します。
        if invalid:
            # 重複除去・整列した不正値一覧を通知します。
            raise ValueError(f"Board contains invalid cell values: {sorted(set(invalid))}.")


# 遅延初期化するデフォルトクライアント参照です。
_default_client: TssRulesClient | None = None


# 共有のデフォルトクライアントを取得します。
def get_default_client() -> TssRulesClient:
    # モジュール変数を書き換えるため global 宣言します。
    global _default_client
    # まだ未初期化ならここで 1 回だけ生成します。
    if _default_client is None:
        # 既定設定でクライアントを構築します。
        _default_client = TssRulesClient()
    # 初期化済みクライアントを返します。
    return _default_client


# モジュール関数として黒禁手判定を提供します。
def is_forbidden_for_black(board: list[int], index: int) -> bool:
    # デフォルトクライアントへ委譲して結果を返します。
    return get_default_client().is_forbidden_for_black(board, index)


# モジュール関数として単体合法手マスク取得を提供します。
def legal_move_mask(board: list[int]) -> list[bool]:
    # デフォルトクライアントへ委譲して結果を返します。
    return get_default_client().legal_move_mask(board)


# モジュール関数としてバッチ合法手マスク取得を提供します。
def legal_move_masks(boards: list[list[int]], players: list[int] | None = None) -> list[list[bool]]:
    # デフォルトクライアントへ委譲して結果を返します。
    return get_default_client().legal_move_masks(boards, players)
