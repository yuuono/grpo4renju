#include <algorithm>   // std::sort、std::findなどのアルゴリズム
#include <array>       // 固定サイズ配列std::array
#include <cctype>      // 文字判定関数isspace・isdigit
#include <chrono>      // 時間計測用クラス
#include <cmath>       // 数値関数sqrt・abs
#include <cstdlib>     // 文字列→数値変換strtol
#include <exception>   // 例外クラス
#include <iostream>    // 標準入出力
#include <limits>      // 型の最大・最小値
#include <sstream>     // 文字列ストリーム
#include <stdexcept>   // 標準例外クラス
#include <string>      // 文字列
#include <thread>      // マルチスレッド
#include <utility>     // std::pair・std::move
#include <vector>      // 可変長配列

namespace {

// 連珠盤の一辺のサイズ（15×15）
constexpr int BOARD_SIZE = 15;
// 盤面の総セル数（225）
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
// 空セルを表す値
constexpr int EMPTY = 0;
// 黒石を表す値
constexpr int BLACK = 1;
// 白石を表す値
constexpr int WHITE = 2;
// 勝者なし（未決着）を表す値
constexpr int NO_WINNER = -1;
// TSS探索のデフォルト最大深さ
constexpr int DEFAULT_MAX_DEPTH = 7;
// 探索する候補手のデフォルト最大数
constexpr int DEFAULT_CANDIDATE_LIMIT = 24;

// 盤面を表す型エイリアス（225要素のint配列）
using Board = std::array<int, BOARD_CELLS>;

// 4方向（水平・垂直・右下り斜め・右上り斜め）の移動ベクトル
const std::array<std::pair<int, int>, 4> DIRECTIONS = {
    std::make_pair(1, 0),    // 横方向
    std::make_pair(0, 1),    // 縦方向
    std::make_pair(1, 1),    // 右下斜め
    std::make_pair(1, -1),   // 右上斜め
};

// TSS評価リクエストを表す構造体
struct Request {
    Board board {};                              // 盤面状態
    int player = BLACK;                          // 手番のプレイヤー
    int move = -1;                               // 評価対象の手のインデックス
    int max_depth = DEFAULT_MAX_DEPTH;           // 最大探索深さ
    int candidate_limit = DEFAULT_CANDIDATE_LIMIT; // 候補手の上限
};

// ゲーム木探索の結果を表す構造体
struct SearchResult {
    bool forced = false;  // 強制勝ちがあるかどうか
    int depth = 0;        // 強制手順の深さ
};

// TSS評価結果を表す構造体
struct EvaluationResult {
    double score = 0.0;        // 静的評価スコア
    bool forced_win = false;   // 強制勝ちありか
    bool forced_loss = false;  // 強制負けありか
    int win_depth = -1;        // 強制勝ちまでの手数
    int loss_depth = -1;       // 強制負けまでの手数
};

// リワード計算の設定パラメータを表す構造体
struct RewardConfig {
    double illegal = -1.0;                 // 非合法手のリワード
    double immediate_win = 1.0;            // 即勝リワード
    double immediate_loss = -1.0;          // 即負リワード
    double allow_immediate_loss = -1.0;    // 相手の即勝を許した際のペナルティ
    double block_immediate_win = 0.3;      // 相手の即勝をブロックしたボーナス
    double tss_weight = 1.0;               // TSSスコアの重み
    double tss_forced_win = 0.7;           // TSS強制勝ちボーナス
    double tss_forced_loss = -0.7;         // TSS強制負けペナルティ
    double create_four = 0.15;             // 四を作ったボーナス
    double create_open_three = 0.05;       // 三を作ったボーナス
    int staged = 1;                        // 段階探索モードを使うか
    int shallow_depth = 1;                 // 浅い探索の深さ
    int deep_top_k = 8;                    // 深い探索を適用する上位K件
    double deep_score_threshold = 0.20;    // 深い探索に進むスコア閾値
};

// リワード評価の結果を表す構造体
struct RewardEvaluationResult {
    double reward = 0.0;        // リワードスコア
    double tss_score = 0.0;     // TSSスコア
    bool forced_win = false;    // 強制勝ちありか
    bool forced_loss = false;   // 強制負けありか
    int win_depth = -1;         // 強制勝ちまでの手数
    int loss_depth = -1;        // 強制負けまでの手数
    bool illegal = false;       // 非合法手か
    bool terminal = false;      // 終了状態か
};

// 盤面中心インデックスを返す
int center_index() {
    return (BOARD_SIZE / 2) * BOARD_SIZE + (BOARD_SIZE / 2);
}

// (行, 列) → 1次元インデックスへ変換
int rc_to_idx(int row, int col) {
    return row * BOARD_SIZE + col;
}

// 1次元インデックス → (行, 列) へ変換
std::pair<int, int> idx_to_rc(int index) {
    return {index / BOARD_SIZE, index % BOARD_SIZE};
}

// 指定した座標が盤面内かどうか判定
bool inside(int row, int col) {
    return 0 <= row && row < BOARD_SIZE && 0 <= col && col < BOARD_SIZE;
}

// 相手プレイヤーを返す（黒↔白）
int other_player(int player) {
    return player == BLACK ? WHITE : BLACK;
}

// 盤面に手を打った新しい盤面を返す
Board board_with_move(const Board& board, int index, int player) {
    Board next_board = board;                                              // 現在の盤面をコピー
    next_board[static_cast<std::size_t>(index)] = player;                  // 指定位置に石を置く
    return next_board;                                                     // 新しい盤面を返す
}

// 盤上の黒・白の石数を集計
std::pair<int, int> stone_counts(const Board& board) {
    int black_count = 0;                                                   // 黒石カウント
    int white_count = 0;                                                   // 白石カウント
    for (int cell : board) {                                               // 全セルを走査
        if (cell == BLACK) {
            ++black_count;                                                 // 黒をインクリメント
        } else if (cell == WHITE) {
            ++white_count;                                                 // 白をインクリメント
        }
    }
    return {black_count, white_count};                                     // (黒, 白) ペアを返す
}

// 盤面の石数から現在の手番プレイヤーを推定
int infer_player(const Board& board) {
    const auto [black_count, white_count] = stone_counts(board);           // 石数を取得
    if (black_count == white_count) {
        return BLACK;                                                      // 同数なら黒番
    }
    if (black_count == white_count + 1) {
        return WHITE;                                                      // 黒が1個多いなら白番
    }
    throw std::runtime_error("Invalid board stone counts.");               // 不正な石数の場合は例外
}

// 指定位置を含む、指定方向の連続する石の数をカウント
int contiguous_count(const Board& board, int index, int player, int dr, int dc) {
    int total = 1;                                                         // 自身を含めて一つから開始
    auto [row, col] = idx_to_rc(index);                                    // 起点座標を取得

    // 正方向に同色の石を順にカウント
    for (int step = 1; inside(row + dr * step, col + dc * step); ++step) {
        if (board[static_cast<std::size_t>(rc_to_idx(row + dr * step, col + dc * step))] != player) {
            break;                                                         // 同色でないならストップ
        }
        ++total;
    }

    // 負方向に同色の石を順にカウント
    for (int step = 1; inside(row - dr * step, col - dc * step); ++step) {
        if (board[static_cast<std::size_t>(rc_to_idx(row - dr * step, col - dc * step))] != player) {
            break;                                                         // 同色でないならストップ
        }
        ++total;
    }

    return total;                                                          // 連続数を返す
}

// 指定手を含むいずれかの方向に5つ以上の連珠があるか
bool has_five_or_more(const Board& board, int index, int player) {
    for (const auto& [dr, dc] : DIRECTIONS) {                              // 4方向をチェック
        if (contiguous_count(board, index, player, dr, dc) >= 5) {
            return true;                                                   // 5連珠以上あり
        }
    }
    return false;
}

// 指定手を含むいずれかの方向に6つ以上の連珠（長連）があるか
bool is_overline(const Board& board, int index, int player) {
    for (const auto& [dr, dc] : DIRECTIONS) {                              // 4方向をチェック
        if (contiguous_count(board, index, player, dr, dc) >= 6) {
            return true;                                                   // 6連珠以上あり（長連）
        }
    }
    return false;
}

// 盤面上に指定プレイヤーの5連珠以上が存在するか
bool player_has_five(const Board& board, int player) {
    for (int index = 0; index < BOARD_CELLS; ++index) {                    // 全セルを走査
        if (board[static_cast<std::size_t>(index)] == player && has_five_or_more(board, index, player)) {
            return true;                                                   // 5連珠を見つけた
        }
    }
    return false;
}

// 盤面上に指定プレイヤーの長連（6連珠以上）が存在するか
bool player_has_overline(const Board& board, int player) {
    for (int index = 0; index < BOARD_CELLS; ++index) {                    // 全セルを走査
        if (board[static_cast<std::size_t>(index)] == player && is_overline(board, index, player)) {
            return true;                                                   // 長連を見つけた
        }
    }
    return false;
}

// 盤面の勝者を判定する（連珠ルール適用）
int board_winner(const Board& board) {
    if (player_has_overline(board, BLACK)) {
        return WHITE;                                                      // 黒に長連があれば黒の反則で白勝ち
    }
    if (player_has_five(board, BLACK)) {
        return BLACK;                                                      // 黒に5連珠があれば黒勝ち
    }
    if (player_has_five(board, WHITE)) {
        return WHITE;                                                      // 白に5連珠以上があれば白勝ち
    }
    return NO_WINNER;                                                      // 未決着
}

// 指定位置を通る、指定方向の直線上の全位置を返す
std::vector<int> line_points_through(int index, int dr, int dc) {
    auto [row, col] = idx_to_rc(index);                                    // 起点座標を取得
    // 逆方向に盤面端まで退く
    while (inside(row - dr, col - dc)) {
        row -= dr;
        col -= dc;
    }

    std::vector<int> points;                                               // 直線上のインデックスリスト
    // 盤面端から反対側まで順にインデックスを追加
    while (inside(row, col)) {
        points.push_back(rc_to_idx(row, col));
        row += dr;
        col += dc;
    }
    return points;
}

// 指定直線上で即勝となる手（5連珠を作る手）を全て返す
std::vector<int> immediate_wins_in_direction(
    const Board& board,
    int player,
    const std::vector<int>& line_points
) {
    std::vector<int> wins;                                                 // 勝ち手リスト
    for (int candidate : line_points) {                                    // 直線上の各点をチェック
        if (board[static_cast<std::size_t>(candidate)] != EMPTY) {
            continue;                                                      // 空セル以外はスキップ
        }
        Board next_board = board_with_move(board, candidate, player);      // 仮に石を置く
        if (player == BLACK && is_overline(next_board, candidate, BLACK)) {
            continue;                                                      // 黒の長連は反則なのでスキップ
        }
        if (has_five_or_more(next_board, candidate, player)) {
            wins.push_back(candidate);                                     // 5連珠を作る手を追加
        }
    }
    return wins;
}

// 指定手が何方向に「四（即勝手を含む直線）」を作るかをカウント
int count_four_directions(const Board& board, int move, int player) {
    int count = 0;                                                         // 「四」の方向数
    for (const auto& [dr, dc] : DIRECTIONS) {                              // 4方向をチェック
        const std::vector<int> line_points = line_points_through(move, dr, dc);  // 直線上の点を取得
        if (!immediate_wins_in_direction(board, player, line_points).empty()) {
            ++count;                                                       // 即勝手があれば「四」としてカウント
        }
    }
    return count;
}

// 指定手が何方向に「三（3連珠で両端から四を作れる）」を作るかをカウント
int count_open_three_directions(const Board& board, int move, int player) {
    int count = 0;                                                         // 「三」の方向数
    for (const auto& [dr, dc] : DIRECTIONS) {                              // 4方向をチェック
        const std::vector<int> line_points = line_points_through(move, dr, dc);  // 直線上の点を取得
        bool found_open_three = false;                                     // 「三」を見つけたか
        for (int candidate : line_points) {                                // 直線上の各点を試す
            if (board[static_cast<std::size_t>(candidate)] != EMPTY) {
                continue;                                                  // 空セル以外はスキップ
            }
            Board next_board = board_with_move(board, candidate, player);  // 仮に石を置く
            if (player == BLACK && is_overline(next_board, candidate, BLACK)) {
                continue;                                                  // 黒の長連はスキップ
            }
            const std::vector<int> winning_points =
                immediate_wins_in_direction(next_board, player, line_points);
            // 次手で即勝手が2つ以上あれば「三」と認定
            if (winning_points.size() >= 2) {
                found_open_three = true;
                break;
            }
        }
        if (found_open_three) {
            ++count;
        }
    }
    return count;
}

// 黒の禁じ手（反則手）判定：長連・四四・三三をチェック
bool is_forbidden_for_black(const Board& board, int index) {
    if (board[static_cast<std::size_t>(index)] != EMPTY) {
        return true;                                                       // 空セルでないなら打てない
    }

    const auto [black_count, white_count] = stone_counts(board);
    const int move_number = black_count + white_count;                     // 現在の手番
    if (move_number == 0) {
        return index != center_index();                                    // 初手は中央限定
    }

    Board next_board = board_with_move(board, index, BLACK);               // 仮に黒を置く
    if (is_overline(next_board, index, BLACK)) {
        return true;                                                       // 長連は反則
    }
    if (count_four_directions(next_board, index, BLACK) >= 2) {
        return true;                                                       // 四四は反則
    }
    if (count_open_three_directions(next_board, index, BLACK) >= 2) {
        return true;                                                       // 三三は反則
    }
    return false;
}

// 指定位置が指定プレイヤーにとって合法手かどうか判定
bool is_legal_move_for(const Board& board, int index, int player) {
    if (index < 0 || index >= BOARD_CELLS) {
        return false;                                                      // 範囲外
    }
    if (board[static_cast<std::size_t>(index)] != EMPTY) {
        return false;                                                      // 空セルでない
    }
    if (player == BLACK && is_forbidden_for_black(board, index)) {
        return false;                                                      // 黒の禁じ手
    }
    return true;
}

// 手を打った後の勝者を返す。未決着ならNO_WINNER
int winner_after_move(const Board& board, int index, int player) {
    if (player == BLACK && is_overline(board, index, BLACK)) {
        return WHITE;                                                      // 黒の長連は黒の反則となり白勝ち
    }
    if (has_five_or_more(board, index, player)) {
        return player;                                                     // 5連珠以上で勝ち
    }
    return NO_WINNER;
}

// 盤面上で石が置かれている位置のインデックスリストを返す
std::vector<int> occupied_indexes(const Board& board) {
    std::vector<int> stones;
    for (int index = 0; index < BOARD_CELLS; ++index) {
        if (board[static_cast<std::size_t>(index)] != EMPTY) {
            stones.push_back(index);                                       // 石がある位置を追加
        }
    }
    return stones;
}

// 既存の石の近傍（半径radius以内）の空セルを候補手として返す
std::vector<int> neighbor_candidates(const Board& board, int radius) {
    const std::vector<int> stones = occupied_indexes(board);               // 既存の石を取得
    if (stones.empty()) {
        return {center_index()};                                           // 石が無ければ中央を返す
    }

    std::array<bool, BOARD_CELLS> seen {};                                 // 重複除外用フラグ
    std::vector<int> candidates;                                           // 候補リスト
    for (int index : stones) {                                             // 各石について
        auto [row, col] = idx_to_rc(index);                                // 座標を取得
        for (int dr = -radius; dr <= radius; ++dr) {                       // 近傍を走査
            for (int dc = -radius; dc <= radius; ++dc) {
                const int nr = row + dr;
                const int nc = col + dc;
                if (!inside(nr, nc)) {
                    continue;                                              // 盤面外はスキップ
                }
                const int candidate = rc_to_idx(nr, nc);
                if (board[static_cast<std::size_t>(candidate)] != EMPTY) {
                    continue;                                              // 空セル以外はスキップ
                }
                if (seen[static_cast<std::size_t>(candidate)]) {
                    continue;                                              // 重複はスキップ
                }
                seen[static_cast<std::size_t>(candidate)] = true;
                candidates.push_back(candidate);
            }
        }
    }
    return candidates;
}

// 指定プレイヤーの合法手リストを返す（近傍優先、無ければ全セル探索）
std::vector<int> legal_moves_for(const Board& board, int player) {
    std::vector<int> moves;
    // 近傍候補を優先探索
    for (int move : neighbor_candidates(board, 2)) {
        if (is_legal_move_for(board, move, player)) {
            moves.push_back(move);
        }
    }
    if (!moves.empty()) {
        return moves;                                                      // 近傍で見つかればそれを返す
    }
    // 近傍に合法手が無ければ全セルを探索
    for (int move = 0; move < BOARD_CELLS; ++move) {
        if (is_legal_move_for(board, move, player)) {
            moves.push_back(move);
        }
    }
    return moves;
}

// 指定プレイヤーの即勝手リストを返す
std::vector<int> immediate_winning_moves(const Board& board, int player) {
    std::vector<int> winning_moves;
    for (int move : legal_moves_for(board, player)) {                      // 全合法手をチェック
        Board next_board = board_with_move(board, move, player);           // 仮に手を打つ
        if (winner_after_move(next_board, move, player) == player) {
            winning_moves.push_back(move);                                 // 勝てる手を追加
        }
    }
    return winning_moves;
}

// 盤面中心からの距離の二乗を返す（手順中央重視に使用）
int center_distance_sq(int index) {
    auto [row, col] = idx_to_rc(index);
    const int center = BOARD_SIZE / 2;
    const int dr = row - center;
    const int dc = col - center;
    return dr * dr + dc * dc;
}

// 手の脅威スコアをヒューリスティックに計算して返す
int move_threat_score(const Board& board, int move, int player) {
    Board next_board = board_with_move(board, move, player);               // 仮に手を打つ
    if (winner_after_move(next_board, move, player) == player) {
        return 100000;                                                     // 即勝手は最大スコア
    }

    const std::vector<int> wins = immediate_winning_moves(next_board, player);  // 仮手後の勝ち手
    int score = 0;
    score += static_cast<int>(wins.size()) * 10000;                        // 複数勝ち手を出すボーナス
    score += count_four_directions(next_board, move, player) * 1000;       // 「四」ボーナス
    score += count_open_three_directions(next_board, move, player) * 250;  // 「三」ボーナス
    // 連珠長の二乗によるスコア（長い連珠を重視）
    for (const auto& [dr, dc] : DIRECTIONS) {
        const int length = contiguous_count(next_board, move, player, dr, dc);
        score += length * length * 8;
    }
    score -= center_distance_sq(move);                                     // 中央近い手を優先
    return score;
}

// 攻撃手（強制勝ちを狙う手）の候補を並べる
std::vector<int> forcing_moves_for(const Board& board, int attacker, int candidate_limit) {
    std::vector<int> moves = legal_moves_for(board, attacker);             // 全合法手を取得
    // 脅威スコア順にソート（同点は中央近いものを先に）
    std::stable_sort(moves.begin(), moves.end(), [&](int lhs, int rhs) {
        const int lhs_score = move_threat_score(board, lhs, attacker);
        const int rhs_score = move_threat_score(board, rhs, attacker);
        if (lhs_score != rhs_score) {
            return lhs_score > rhs_score;
        }
        return center_distance_sq(lhs) < center_distance_sq(rhs);
    });

    std::vector<int> forcing;                                              // 強制手のリスト
    for (int move : moves) {
        Board next_board = board_with_move(board, move, attacker);         // 仮に手を打つ
        if (winner_after_move(next_board, move, attacker) == attacker) {
            forcing.push_back(move);                                       // 即勝手は強制手
            continue;
        }
        if (!immediate_winning_moves(next_board, attacker).empty()) {
            forcing.push_back(move);                                       // 次手に即勝手ありは強制手
            continue;
        }
        if (count_four_directions(next_board, move, attacker) >= 1) {
            forcing.push_back(move);                                       // 「四」を作る手は強制手
            continue;
        }
        if (count_open_three_directions(next_board, move, attacker) >= 1) {
            forcing.push_back(move);                                       // 「三」を作る手は強制手
        }
    }

    // 候補上限を超える場合は上位のみ保持
    if (static_cast<int>(forcing.size()) > candidate_limit) {
        forcing.resize(static_cast<std::size_t>(candidate_limit));
    }
    return forcing;
}

// リストに指定手が含まれるか
bool contains_move(const std::vector<int>& moves, int move) {
    return std::find(moves.begin(), moves.end(), move) != moves.end();
}

// 既存のopen threeをopen fourへ伸ばせるなら強制勝ちとして扱う
SearchResult open_three_extension_forced_win(const Board& board, int attacker) {
    const int defender = other_player(attacker);
    for (int move : legal_moves_for(board, attacker)) {
        Board after_extension = board_with_move(board, move, attacker);
        if (winner_after_move(after_extension, move, attacker) == attacker) {
            return {true, 1};                                                // 念のため即勝も拾う
        }
        if (count_four_directions(after_extension, move, attacker) <= 0) {
            continue;                                                        // 四に伸びない手は対象外
        }

        const std::vector<int> attacker_wins = immediate_winning_moves(after_extension, attacker);
        if (attacker_wins.size() < 2) {
            continue;                                                        // open fourでなければ強制勝ち扱いしない
        }
        if (!immediate_winning_moves(after_extension, defender).empty()) {
            continue;                                                        // 相手の即勝を許すなら成立しない
        }
        return {true, 3};                                                     // 伸ばす手→防御→勝ち
    }
    return {false, 0};
}

// TSS 探索：攻撃者が強制勝ちを実現できるかをミニマックス探索で調べる
SearchResult attacker_can_force_win(
    const Board& board,
    int side_to_move,         // 現在手を打つプレイヤー
    int attacker,             // 勝ちを狙う攻撃者
    int depth_remaining,      // 残り探索深度
    int candidate_limit       // 候補手上限
) {
    const int winner = board_winner(board);                                // 現在の勝者をチェック
    if (winner != NO_WINNER) {
        return {winner == attacker, 0};                                    // 既に決着しているなら結果を返す
    }
    if (depth_remaining <= 0) {
        return {false, 0};                                                 // 探索限界に達した（勝ち未検出）
    }

    const int defender = other_player(attacker);                           // 防御者
    if (side_to_move == attacker) {                                        // 攻撃者の手番のとき
        const std::vector<int> direct_wins = immediate_winning_moves(board, attacker);
        if (!direct_wins.empty()) {
            return {true, 1};                                              // 即勝手あり（深さ1で勝ち）
        }
        const std::vector<int> defender_wins_now = immediate_winning_moves(board, defender);
        if (defender_wins_now.size() >= 2) {
            return {false, 0};                                             // 防御者が複数勝ち手を持つなら勝てない
        }
        const SearchResult open_three_win = open_three_extension_forced_win(board, attacker);
        if (open_three_win.forced) {
            return open_three_win;                                          // open threeをopen fourへ伸ばせる
        }

        SearchResult best {false, std::numeric_limits<int>::max()};        // 最良結果を初期化
        const std::vector<int> candidates = forcing_moves_for(board, attacker, candidate_limit);  // 強制手候補を取得
        for (int move : candidates) {                                      // 各候補手を試す
            Board after_attack = board_with_move(board, move, attacker);   // 攻撃手を打った盤面
            if (winner_after_move(after_attack, move, attacker) == attacker) {
                best = {true, std::min(best.depth, 1)};                    // この手で即勝
                continue;
            }
            if (!immediate_winning_moves(after_attack, defender).empty()) {
                continue;                                                  // 防御者に即勝手を許したのでスキップ
            }

            const std::vector<int> attack_wins = immediate_winning_moves(after_attack, attacker);
            if (attack_wins.size() >= 2) {
                best = {true, std::min(best.depth, 3)};                    // 二重脅威（防げない）
                continue;
            }
            if (attack_wins.empty()) {
                continue;                                                  // 脅威がもう無いので逆転されてしまう
            }

            bool all_defenses_still_lose = true;                           // すべての防御で勝てるか
            int worst_defense_depth = 0;                                   // 最長防御深さ
            for (int response : attack_wins) {                             // 防御者の応手を試す
                if (!is_legal_move_for(after_attack, response, defender)) {
                    continue;                                              // 防御者がそこに打てない
                }
                Board after_response = board_with_move(after_attack, response, defender);  // 防御後の盤面
                SearchResult child = attacker_can_force_win(               // 再帰して探索を続行
                    after_response,
                    attacker,
                    attacker,
                    depth_remaining - 2,
                    candidate_limit
                );
                if (!child.forced) {
                    all_defenses_still_lose = false;                       // 逆転できる防御手があった
                    break;
                }
                worst_defense_depth = std::max(worst_defense_depth, child.depth);
            }
            if (all_defenses_still_lose) {
                best = {true, std::min(best.depth, 2 + worst_defense_depth)};  // この攻撃手で強制勝ち確定
            }
        }
        if (best.forced) {
            return best;                                                   // 強制勝ちあり
        }
        return {false, 0};                                                 // 強制勝ちを見つけられなかった
    }

    // === 以下は防御者の手番のときの処理 ===
    const std::vector<int> defender_wins = immediate_winning_moves(board, defender);
    if (!defender_wins.empty()) {
        return {false, 0};                                                 // 防御者が先に勝てる（攻撃者失敗）
    }

    const std::vector<int> attacker_wins = immediate_winning_moves(board, attacker);
    if (attacker_wins.size() >= 2) {
        return {true, 2};                                                  // 攻撃者が二重脅威を作っている
    }
    if (attacker_wins.empty()) {
        return {false, 0};                                                 // 脅威無し、勝てない
    }

    const int forced_response = attacker_wins.front();                     // 防御者はその1点に応手させられる
    if (!is_legal_move_for(board, forced_response, defender)) {
        return {true, 1};                                                  // 防げないので攻撃者勝ち
    }
    Board after_response = board_with_move(board, forced_response, defender);  // 防御後の盤面
    SearchResult child = attacker_can_force_win(                           // 防御後も強制勝ちできるか再帰探索
        after_response,
        attacker,
        attacker,
        depth_remaining - 1,
        candidate_limit
    );
    if (!child.forced) {
        return {false, 0};                                                 // この先は強制勝ちできない
    }
    return {true, 1 + child.depth};                                        // 強制勝ちを返す
}

// 静的な脅威スコアを計算（[-1, 1] にクリップ）
double static_threat_score(const Board& board, int root_player, int last_move) {
    const int opponent = other_player(root_player);                        // 相手プレイヤー
    double score = 0.0;
    if (0 <= last_move && last_move < BOARD_CELLS) {
        score += count_four_directions(board, last_move, root_player) * 0.18;                // 「四」を作ったボーナス
        score += std::min(count_open_three_directions(board, last_move, root_player), 2) * 0.10;  // 「三」ボーナス（最大2個まで）
    }
    const std::vector<int> own_wins = immediate_winning_moves(board, root_player);          // 自分の勝ち手
    const std::vector<int> opponent_wins = immediate_winning_moves(board, opponent);        // 相手の勝ち手
    score += std::min<int>(static_cast<int>(own_wins.size()), 2) * 0.20;   // 自脅威ボーナス
    score -= std::min<int>(static_cast<int>(opponent_wins.size()), 2) * 0.25;  // 相手の脅威ペナルティ
    if (score > 1.0) {
        return 1.0;                                                        // 上限クリップ
    }
    if (score < -1.0) {
        return -1.0;                                                       // 下限クリップ
    }
    return score;
}

// 探索深さに応じてスコアをスケール（深い勝ちほど価値を低く見る）
// 探索深さに応じてスコアをスケール（深い勝ちほど価値を低く見る）
double depth_scaled(double base_score, int depth) {
    if (depth <= 1) {
        return base_score;                                                 // 浅い勝ちはそのまま
    }
    return base_score / std::sqrt(static_cast<double>(depth));             // 深さの平方根で割って減衰
}

// 盤面と手についてTSS評価を行う（強制勝ち/負けと静的スコアを返す）
EvaluationResult evaluate_tss(
    const Board& board,
    int player,
    int move,
    int max_depth,
    int candidate_limit
) {
    const int side_to_move = infer_player(board);                          // 盤面から手番を推定
    const int opponent = other_player(player);                             // 相手プレイヤー

    // 自分が強制勝ちできるかを探索
    const SearchResult win = attacker_can_force_win(
        board,
        side_to_move,
        player,
        max_depth,
        candidate_limit
    );
    // 相手が強制勝ちできるか（＝自分の強制負け）を探索
    const SearchResult loss = attacker_can_force_win(
        board,
        side_to_move,
        opponent,
        max_depth,
        candidate_limit
    );

    return {
        static_threat_score(board, player, move),                          // 静的評価スコア
        win.forced,                                                        // 強制勝ちあり
        loss.forced,                                                       // 強制負けあり
        win.forced ? win.depth : -1,                                       // 強制勝ち深さ
        loss.forced ? loss.depth : -1,                                     // 強制負け深さ
    };
}

// TSS評価結果からリワードスコアを計算
double tss_reward_score(const EvaluationResult& evaluation, const RewardConfig& config) {
    double score = evaluation.score;                                       // 静的スコアから始める
    if (evaluation.forced_win) {
        score += depth_scaled(config.tss_forced_win, evaluation.win_depth);   // 強制勝ちボーナスを加算
    }
    if (evaluation.forced_loss) {
        score += depth_scaled(config.tss_forced_loss, evaluation.loss_depth); // 強制負けペナルティを加算
    }
    return config.tss_weight * score;                                      // TSS重みを乗算
}

// 形（四・三）を作ったボーナススコアを計算
double shape_reward_score(const Board& board_after_action, int action, int player, const RewardConfig& config) {
    const int fours = count_four_directions(board_after_action, action, player);          // 「四」の方向数
    const int open_threes = count_open_three_directions(board_after_action, action, player); // 「三」の方向数
    return std::min(fours, 1) * config.create_four + std::min(open_threes, 2) * config.create_open_three;
}

// 1手について完全なリワード評価を行う（合法性チェック・即勝/即負け・TSS・形ボーナスを統合）
RewardEvaluationResult evaluate_reward(
    const Board& board,
    int configured_player,
    int move,
    int max_depth,
    int candidate_limit,
    const RewardConfig& config,
    bool force_deep_search
) {
    int player = configured_player;                                        // プレイヤー指定を受け取る
    if (player != BLACK && player != WHITE) {
        player = infer_player(board);                                      // 未指定なら盤面から推定
    }

    // 非合法手のチェック
    if (!is_legal_move_for(board, move, player)) {
        return {
            config.illegal,                                                // 非合法手リワード
            0.0,
            false,
            false,
            -1,
            -1,
            true,                                                          // illegalフラグ
            true,                                                          // terminalフラグ
        };
    }

    double reward = 0.0;                                                   // リワードを初期化
    const int opponent = other_player(player);                             // 相手プレイヤー
    const std::vector<int> opponent_wins_before = immediate_winning_moves(board, opponent); // 相手の即勝手（着手前）
    Board next_board = board_with_move(board, move, player);               // 手を打った後の盤面
    const int winner = winner_after_move(next_board, move, player);        // 着手後の勝者をチェック
    if (winner == player) {
        return {
            config.immediate_win,                                          // 即勝リワード
            0.0,
            false,
            false,
            -1,
            -1,
            false,
            true,                                                          // terminalフラグ
        };
    }
    if (winner == opponent) {
        return {
            config.immediate_loss,                                         // 即負リワード（黒の禁じ手等）
            0.0,
            false,
            false,
            -1,
            -1,
            false,
            true,
        };
    }

    // 相手の即勝手をブロックしていればボーナス
    if (contains_move(opponent_wins_before, move)) {
        reward += config.block_immediate_win;
    }
    // この手で相手に即勝手を許してしまった場合はペナルティ
    if (!immediate_winning_moves(next_board, opponent).empty()) {
        reward += config.allow_immediate_loss;
    }

    // 強制深探索か浅い探索かを決定
    const int search_depth = force_deep_search ? max_depth : std::min(config.shallow_depth, max_depth);
    const EvaluationResult tss = evaluate_tss(next_board, player, move, search_depth, candidate_limit);  // TSS評価
    const double tss_score = tss_reward_score(tss, config);                // TSSスコアを計算
    reward += tss_score;                                                   // リワードに加算
    reward += shape_reward_score(next_board, move, player, config);        // 形ボーナスを加算

    return {
        reward,
        tss_score,
        tss.forced_win,
        tss.forced_loss,
        tss.win_depth,
        tss.loss_depth,
        false,
        false,
    };
}

// TSS探索を行わず静的評価のみでリワードを計算（軽量版）
RewardEvaluationResult evaluate_static_reward(
    const Board& board,
    int configured_player,
    int move,
    const RewardConfig& config
) {
    int player = configured_player;
    if (player != BLACK && player != WHITE) {
        player = infer_player(board);                                      // 未指定なら盤面から推定
    }

    // 非合法手のチェック
    if (!is_legal_move_for(board, move, player)) {
        return {
            config.illegal,
            0.0,
            false,
            false,
            -1,
            -1,
            true,                                                          // illegalフラグ
            true,                                                          // terminalフラグ
        };
    }

    double reward = 0.0;                                                   // リワードを初期化
    const int opponent = other_player(player);                             // 相手プレイヤー
    const std::vector<int> opponent_wins_before = immediate_winning_moves(board, opponent);  // 相手の即勝手
    Board next_board = board_with_move(board, move, player);               // 手を打った後の盤面
    const int winner = winner_after_move(next_board, move, player);        // 着手後の勝者
    if (winner == player) {
        return {
            config.immediate_win,                                          // 即勝リワード
            0.0,
            true,                                                          // forced_win=true
            false,
            1,
            -1,
            false,
            true,
        };
    }
    if (winner == opponent) {
        return {
            config.immediate_loss,                                         // 即負リワード
            0.0,
            false,
            true,                                                          // forced_loss=true
            -1,
            1,
            false,
            true,
        };
    }

    // 相手の即勝手をブロックした場合のボーナス
    if (contains_move(opponent_wins_before, move)) {
        reward += config.block_immediate_win;
    }
    // この手で相手に即勝手を許してしまうかどうか
    const bool allows_immediate_loss = !immediate_winning_moves(next_board, opponent).empty();
    if (allows_immediate_loss) {
        reward += config.allow_immediate_loss;                             // 許した場合はペナルティ
    }

    const double tss_score = config.tss_weight * static_threat_score(next_board, player, move);  // 静的脅威スコア
    reward += tss_score;                                                   // リワードに加算
    reward += shape_reward_score(next_board, move, player, config);        // 形ボーナスを加算

    return {
        reward,
        tss_score,
        false,
        allows_immediate_loss,
        -1,
        allows_immediate_loss ? 1 : -1,
        false,
        false,
    };
}

// 標準入力から全て読み込んで文字列として返す
std::string read_stdin() {
    std::ostringstream buffer;
    buffer << std::cin.rdbuf();
    return buffer.str();
}

// JSON文字列から指定キーの値開始位置（コロンの次）を探す
std::size_t find_field(const std::string& json, const std::string& key) {
    const std::string quoted_key = "\"" + key + "\"";                       // ダブルクォートで囲んだキー
    const std::size_t key_pos = json.find(quoted_key);                     // キーの位置を探す
    if (key_pos == std::string::npos) {
        throw std::runtime_error("Missing JSON field: " + key);            // 見つからなければ例外
    }
    const std::size_t colon_pos = json.find(':', key_pos + quoted_key.size());  // コロンを探す
    if (colon_pos == std::string::npos) {
        throw std::runtime_error("Malformed JSON field: " + key);          // コロンが見つからない
    }
    return colon_pos + 1;                                                  // 値部分の開始位置
}

// JSONから整数フィールドをパースする。存在しない場合はデフォルト値を返す
int parse_int_field(const std::string& json, const std::string& key, int default_value, bool required) {
    const std::string quoted_key = "\"" + key + "\"";
    if (json.find(quoted_key) == std::string::npos) {
        if (required) {
            throw std::runtime_error("Missing JSON field: " + key);        // 必須フィールドが無い
        }
        return default_value;                                              // デフォルトを返す
    }

    std::size_t pos = find_field(json, key);                               // 値位置を取得
    // 先頭の空白をスキップ
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    char* end_ptr = nullptr;
    const long value = std::strtol(json.c_str() + pos, &end_ptr, 10);      // 文字列を整数に変換
    if (end_ptr == json.c_str() + pos) {
        throw std::runtime_error("Expected integer JSON field: " + key);   // 数値として読めない
    }
    return static_cast<int>(value);
}

// JSONからboardフィールド（225要素の配列）をパースして返す
Board parse_board_field(const std::string& json) {
    std::size_t pos = find_field(json, "board");                           // "board"キーの位置を取得
    const std::size_t open_pos = json.find('[', pos);                      // 配列開始 '[' を探す
    const std::size_t close_pos = json.find(']', open_pos);                // 配列終了 ']' を探す
    if (open_pos == std::string::npos || close_pos == std::string::npos) {
        throw std::runtime_error("Expected board array.");
    }

    Board board {};                                                        // 0初期化された盤面
    int count = 0;                                                         // 読んだセル数
    pos = open_pos + 1;                                                    // 配列内部の処理開始位置
    while (pos < close_pos) {
        // 数字または負号までスキップ（カンマ・空白などを跳ばす）
        while (pos < close_pos) {
            const char ch = json[pos];
            if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-') {
                break;
            }
            ++pos;
        }
        if (pos >= close_pos) {
            break;                                                         // 配列終わりに達した
        }
        char* end_ptr = nullptr;
        const long value = std::strtol(json.c_str() + pos, &end_ptr, 10);  // 数値をパース
        if (end_ptr == json.c_str() + pos) {
            throw std::runtime_error("Invalid board integer.");
        }
        if (count >= BOARD_CELLS) {
            throw std::runtime_error("Board has too many cells.");        // セル数超過
        }
        if (value < EMPTY || value > WHITE) {
            throw std::runtime_error("Board contains invalid cell value.");  // 不正値
        }
        board[static_cast<std::size_t>(count)] = static_cast<int>(value);  // セルに格納
        ++count;
        pos = static_cast<std::size_t>(end_ptr - json.c_str());            // 次の位置に進む
    }
    if (count != BOARD_CELLS) {
        throw std::runtime_error("Board must contain exactly 225 cells."); // 225要素に不足
    }
    return board;
}

// JSONリクエスト全体をパースしてRequest構造体にマッピング
Request parse_request(const std::string& json) {
    Request request;
    request.board = parse_board_field(json);                               // 盤面をパース
    request.player = parse_int_field(json, "player", BLACK, true);         // プレイヤーは必須
    request.move = parse_int_field(json, "move", -1, false);               // 手は任意
    request.max_depth = parse_int_field(json, "max_depth", DEFAULT_MAX_DEPTH, false);                // 探索深さ
    request.candidate_limit = parse_int_field(json, "candidate_limit", DEFAULT_CANDIDATE_LIMIT, false);  // 候補手上限
    if (request.player != BLACK && request.player != WHITE) {
        throw std::runtime_error("player must be 1 or 2.");                // プレイヤーは1または2のみ
    }
    if (request.max_depth < 1) {
        request.max_depth = 1;                                             // 下限補正
    }
    if (request.candidate_limit < 1) {
        request.candidate_limit = 1;                                       // 下限補正
    }
    return request;
}

// プログラムの使用方法を標準出力に表示
void print_usage() {
    std::cout
        << "Usage: tss < request.json\n"
        << "Input JSON: {\"board\":[225 ints],\"player\":1|2,\"move\":112,\"max_depth\":7}\n"
        << "Output JSON contains score, forced_win, forced_loss, win_depth, loss_depth.\n";
}

// スレッド数を決定（要求2スレッド、作業量、ハードウェア並列性を考慮）
int resolve_thread_count(int requested_threads, int work_count) {
    if (work_count <= 1) {
        return 1;                                                          // 作業が1件以下なら並列不要
    }
    int thread_count = requested_threads;
    if (thread_count <= 0) {
        thread_count = static_cast<int>(std::thread::hardware_concurrency());  // 自動検出
        if (thread_count <= 0) {
            thread_count = 1;                                              // 検出失敗時は1
        }
    }
    return std::max(1, std::min(thread_count, work_count));                // 作業数を上限にクリップ
}

// 深いTSS探索を並列で実行（スレッドプールとして複数スレッドに分配）
void evaluate_deep_indexes(
    const std::vector<int>& deep_indexes,
    const std::vector<Board>& parsed_boards,
    const std::vector<int>& parsed_players,
    const int* moves,
    int max_depth,
    int candidate_limit,
    const RewardConfig& config,
    int requested_threads,
    std::vector<RewardEvaluationResult>& evaluations
) {
    // 実際に使うスレッド数を決定
    const int thread_count = resolve_thread_count(
        requested_threads,
        static_cast<int>(deep_indexes.size())
    );
    if (thread_count <= 1) {
        // シングルスレッド処理
        for (int request_index : deep_indexes) {
            evaluations[static_cast<std::size_t>(request_index)] = evaluate_reward(
                parsed_boards[static_cast<std::size_t>(request_index)],
                parsed_players[static_cast<std::size_t>(request_index)],
                moves[request_index],
                max_depth,
                candidate_limit,
                config,
                true                                                       // 深探索を強制
            );
        }
        return;
    }

    // マルチスレッド処理: ワーカースレッドを起動して仕事をラウンドロビン分配
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (int worker_id = 0; worker_id < thread_count; ++worker_id) {
        workers.emplace_back([&, worker_id]() {
            // 各ワーカーは自分のIDをオフセットとして thread_count 刻みで担当
            for (
                std::size_t index = static_cast<std::size_t>(worker_id);
                index < deep_indexes.size();
                index += static_cast<std::size_t>(thread_count)
            ) {
                const int request_index = deep_indexes[index];                 // 元のリクエスト番号
                evaluations[static_cast<std::size_t>(request_index)] = evaluate_reward(
                    parsed_boards[static_cast<std::size_t>(request_index)],
                    parsed_players[static_cast<std::size_t>(request_index)],
                    moves[request_index],
                    max_depth,
                    candidate_limit,
                    config,
                    true                                                       // 深探索を強制
                );
            }
        });
    }
    // 全ワーカーの完了を待機
    for (std::thread& worker : workers) {
        worker.join();
    }
}

}  // namespace

// === Python (ctypes) から呼び出すための C インターフェース ===
extern "C" {

// TSS評価結果をPython側に返すための構造体（C ABI）
struct TssResult {
    double score;                                                           // 評価スコア
    int forced_win;                                                         // 強制勝ちフラグ（0/1）
    int forced_loss;                                                        // 強制負けフラグ（0/1）
    int win_depth;                                                          // 勝ちまでの探索深さ
    int loss_depth;                                                         // 負けまでの探索深さ
};

// 報酬計算の設定をPythonから受け取るための構造体
struct TssRewardConfig {
    double illegal;                                                         // 違反手ペナルティ
    double immediate_win;                                                   // 即勝ボーナス
    double immediate_loss;                                                  // 即負ペナルティ
    double allow_immediate_loss;                                            // 相手の即勝を見逃したペナルティ
    double block_immediate_win;                                             // 相手の即勝を阫いだボーナス
    double tss_weight;                                                      // TSSスコアの重み
    double tss_forced_win;                                                  // TSS強制勝ちボーナス
    double tss_forced_loss;                                                 // TSS強制負けペナルティ
    double create_four;                                                     // 四を作ったボーナス
    double create_open_three;                                               // 活三を作ったボーナス
    int staged;                                                             // 2段階探索を使うか（0/1）
    int shallow_depth;                                                      // 浅探索の深さ
    int deep_top_k;                                                         // 深探索に進む上位K件
    double deep_score_threshold;                                            // 深探索トリガーとなるスコア閾値
};

// 報酬計算結果をPython側に返すための構造体
struct TssRewardResult {
    double reward;                                                          // 計算された報酬値
    double tss_score;                                                       // TSS評価スコア
    int forced_win;                                                         // 強制勝ちフラグ
    int forced_loss;                                                        // 強制負けフラグ
    int win_depth;                                                          // 勝ちまでの深さ
    int loss_depth;                                                         // 負けまでの深さ
    int illegal;                                                            // 違反手フラグ
    int terminal;                                                           // 終了状態フラグ
};

// バッチ処理の統計情報を返すための構造体
struct TssBatchStats {
    int deep_count;                                                         // 深探索した件数
    int thread_count;                                                       // 使用スレッド数
    double elapsed_ms;                                                      // 経過ミリ秒
};

// 複数リクエストを一括でTSS評価するバッチ関数（Python側から呼ぶ）
int tss_evaluate_batch(
    const int* boards,
    const int* players,
    const int* moves,
    int num_requests,
    int max_depth,
    int candidate_limit,
    TssResult* results
) {
    // ポインタのヌルチェック
    if (boards == nullptr || players == nullptr || moves == nullptr || results == nullptr) {
        return -1;
    }
    if (num_requests < 0) {
        return -2;                                                          // 不正なリクエスト数
    }
    if (max_depth < 1) {
        max_depth = 1;                                                      // 下限補正
    }
    if (candidate_limit < 1) {
        candidate_limit = 1;                                                // 下限補正
    }

    try {
        // 各リクエストを順に処理
        for (int request_index = 0; request_index < num_requests; ++request_index) {
            Board board {};
            const int board_offset = request_index * BOARD_CELLS;           // フラット配列のオフセット
            // 225セルをコピーして妥当性を検証
            for (int cell_index = 0; cell_index < BOARD_CELLS; ++cell_index) {
                const int cell = boards[board_offset + cell_index];
                if (cell < EMPTY || cell > WHITE) {
                    return -3;                                              // 不正なセル値
                }
                board[static_cast<std::size_t>(cell_index)] = cell;
            }

            const int player = players[request_index];
            if (player != BLACK && player != WHITE) {
                return -4;                                                  // 不正なプレイヤー
            }
            // TSS評価を実行
            const EvaluationResult evaluation = evaluate_tss(
                board,
                player,
                moves[request_index],
                max_depth,
                candidate_limit
            );
            // 結果をPython向けの構造体に詰める
            results[request_index] = {
                evaluation.score,
                evaluation.forced_win ? 1 : 0,
                evaluation.forced_loss ? 1 : 0,
                evaluation.win_depth,
                evaluation.loss_depth,
            };
        }
    } catch (const std::exception&) {
        return -5;                                                          // 例外発生
    }

    return 0;                                                               // 正常終了
}

// バッチ報酬評価を実行し、統計情報も同時に記録する
int tss_evaluate_reward_batch_with_stats(
    const int* boards,
    const int* players,
    const int* moves,
    int num_requests,
    int max_depth,
    int candidate_limit,
    const TssRewardConfig* reward_config,
    TssRewardResult* results,
    int parallel_threads,
    TssBatchStats* stats
) {
    const auto start_time = std::chrono::steady_clock::now();              // 処理開始時刻を記録
    // 必須ポインタをチェック
    if (boards == nullptr || moves == nullptr || reward_config == nullptr || results == nullptr) {
        return -1;                                                          // ポインタが不正
    }
    if (num_requests < 0) {
        return -2;                                                          // リクエスト数が不正
    }
    if (max_depth < 1) {
        max_depth = 1;                                                      // 探索深さ下限補正
    }
    if (candidate_limit < 1) {
        candidate_limit = 1;                                                // 候補手上限下限補正
    }

    // Python側のTssRewardConfigをC++のRewardConfig構造体に変換
    const RewardConfig config {
        reward_config->illegal,                                             // 違反手ペナルティ
        reward_config->immediate_win,                                       // 即勝ボーナス
        reward_config->immediate_loss,                                      // 即負ペナルティ
        reward_config->allow_immediate_loss,                                // 相手の即勝を見逃したペナルティ
        reward_config->block_immediate_win,                                 // 相手の即勝を阻いだボーナス
        reward_config->tss_weight,                                          // TSS重み
        reward_config->tss_forced_win,                                      // TSS強制勝ちボーナス
        reward_config->tss_forced_loss,                                     // TSS強制負けペナルティ
        reward_config->create_four,                                         // 四を作ったボーナス
        reward_config->create_open_three,                                   // 活三を作ったボーナス
        reward_config->staged,                                              // 2段階探索フラグ
        reward_config->shallow_depth,                                       // 浅探索深さ
        reward_config->deep_top_k,                                          // 深探索上位K件
        reward_config->deep_score_threshold,                                // 深探索トリガー閾値
    };

    try {
        // バッチ処理用のベクトルを初期化
        std::vector<Board> parsed_boards(static_cast<std::size_t>(num_requests));              // 盤面キャッシュ
        std::vector<int> parsed_players(static_cast<std::size_t>(num_requests));              // プレイヤー情報
        std::vector<RewardEvaluationResult> evaluations(static_cast<std::size_t>(num_requests)); // 評価結果

        // 各リクエストをパースして浅い評価を実行
        for (int request_index = 0; request_index < num_requests; ++request_index) {
            Board board {};                                                 // 局所盤面
            const int board_offset = request_index * BOARD_CELLS;           // フラット配列のオフセット計算
            // 225セルをコピーして妥当性を検証
            for (int cell_index = 0; cell_index < BOARD_CELLS; ++cell_index) {
                const int cell = boards[board_offset + cell_index];
                if (cell < EMPTY || cell > WHITE) {
                    return -3;                                              // 不正なセル値
                }
                board[static_cast<std::size_t>(cell_index)] = cell;         // セルをコピー
            }

            // プレイヤー情報を取得（playerがnullptrの場合は0をデフォルト）
            const int player = players == nullptr ? 0 : players[request_index];
            parsed_boards[static_cast<std::size_t>(request_index)] = board; // 盤面をキャッシュ
            parsed_players[static_cast<std::size_t>(request_index)] = player;  // プレイヤーをキャッシュ
            // 浅い評価を実行（深探索なし、高速版）
            evaluations[static_cast<std::size_t>(request_index)] = evaluate_static_reward(
                board,
                player,
                moves[request_index],
                config
            );
        }

        // 深探索が必要なリクエストのインデックスを選別
        std::vector<int> deep_indexes;
        deep_indexes.reserve(static_cast<std::size_t>(num_requests));      // メモリ事前確保
        for (int request_index = 0; request_index < num_requests; ++request_index) {
            const RewardEvaluationResult& evaluation = evaluations[static_cast<std::size_t>(request_index)];
            // 違反手または終了状態は深探索をスキップ
            if (evaluation.illegal || evaluation.terminal) {
                continue;                                                   // 次のリクエストへ
            }
            // 2段階探索が無効なら全て深探索対象
            if (!config.staged) {
                deep_indexes.push_back(request_index);
                continue;                                                   // 次のリクエストへ
            }
            // 強制負けが予測される場合は深探索対象（対策確認用）
            if (evaluation.forced_loss) {
                deep_indexes.push_back(request_index);
                continue;                                                   // 次のリクエストへ
            }
            // スコアの絶対値が閾値以上なら深探索対象（重要な手を掘り下げる）
            if (std::abs(evaluation.reward) >= config.deep_score_threshold) {
                deep_indexes.push_back(request_index);                      // 深探索候補に追加
            }
        }

        // 深探索候補をスコア絶対値で大きい順にソート（重要な手を優先）
        std::stable_sort(deep_indexes.begin(), deep_indexes.end(), [&](int lhs, int rhs) {
            const double lhs_abs = std::abs(evaluations[static_cast<std::size_t>(lhs)].reward);  // 左のスコア絶対値
            const double rhs_abs = std::abs(evaluations[static_cast<std::size_t>(rhs)].reward);  // 右のスコア絶対値
            if (lhs_abs != rhs_abs) {
                return lhs_abs > rhs_abs;                                   // 絶対値が大きいほう優先
            }
            return lhs < rhs;                                               // 同じスコアならインデックス小さい方優先
        });
        // 2段階探索で、深探索対象数が上限を超える場合はトリミング
        if (config.staged && config.deep_top_k >= 0 && static_cast<int>(deep_indexes.size()) > config.deep_top_k) {
            deep_indexes.resize(static_cast<std::size_t>(config.deep_top_k));  // 上位K件に絞る
        }

        // 選別されたリクエストに対して深いTSS探索を実行（並列処理あり）
        evaluate_deep_indexes(
            deep_indexes,                                                   // 深探索対象のインデックス
            parsed_boards,                                                  // キャッシュした盤面
            parsed_players,                                                 // キャッシュしたプレイヤー情報
            moves,                                                          // 手の情報
            max_depth,                                                      // TSS探索最大深さ
            candidate_limit,                                                // 候補手上限
            config,                                                         // 報酬設定
            parallel_threads,                                               // 並列スレッド数
            evaluations                                                     // 評価結果を上書き
        );

        // C++側の評価結果をPython向けのTssRewardResult構造体に変換
        for (int request_index = 0; request_index < num_requests; ++request_index) {
            const RewardEvaluationResult& evaluation = evaluations[static_cast<std::size_t>(request_index)];
            // 評価結果をPython側が理解できる形に詰め込む
            results[request_index] = {
                evaluation.reward,                                          // 計算された報酬値
                evaluation.tss_score,                                       // TSS寄与度
                evaluation.forced_win ? 1 : 0,                              // 強制勝ちフラグ（bool→int）
                evaluation.forced_loss ? 1 : 0,                             // 強制負けフラグ（bool→int）
                evaluation.win_depth,                                       // 強制勝ちまでの深さ
                evaluation.loss_depth,                                      // 強制負けまでの深さ
                evaluation.illegal ? 1 : 0,                                 // 違反手フラグ（bool→int）
                evaluation.terminal ? 1 : 0,                                // 終了状態フラグ（bool→int）
            };
        }
        // 統計情報が要求された場合は計算して返す
        if (stats != nullptr) {
            const auto end_time = std::chrono::steady_clock::now();         // 処理終了時刻
            const std::chrono::duration<double, std::milli> elapsed = end_time - start_time;  // 経過時間
            stats->deep_count = static_cast<int>(deep_indexes.size());      // 深探索した件数
            stats->thread_count = resolve_thread_count(
                parallel_threads,                                           // 要求スレッド数
                static_cast<int>(deep_indexes.size())                       // 実作業件数
            );                                                              // 実際に使用したスレッド数
            stats->elapsed_ms = elapsed.count();                            // 経過ミリ秒を記録
        }
    } catch (const std::exception&) {
        return -5;                                                          // 例外発生
    }

    return 0;                                                               // 正常終了
}

// 統計情報なしでバッチ報酬評価を実行する（シンプルなラッパー）
int tss_evaluate_reward_batch(
    const int* boards,
    const int* players,
    const int* moves,
    int num_requests,
    int max_depth,
    int candidate_limit,
    const TssRewardConfig* reward_config,
    TssRewardResult* results
) {
    // with_stats版を呼び出す（並列スレッド数=1、統計情報=nullptr）
    return tss_evaluate_reward_batch_with_stats(
        boards,
        players,
        moves,
        num_requests,
        max_depth,
        candidate_limit,
        reward_config,
        results,
        1,                                                                  // 並列スレッド数（ここでは1固定）
        nullptr                                                             // 統計情報は不要
    );
}

int tss_is_forbidden_for_black(
    const int* board_array,
    int move
) {
    if (board_array == nullptr) {
        return 1;
    }
    if (move < 0 || move >= BOARD_CELLS) {
        return 1;
    }

    try {
        Board board {};
        for (int cell_index = 0; cell_index < BOARD_CELLS; ++cell_index) {
            const int cell = board_array[cell_index];
            if (cell < EMPTY || cell > WHITE) {
                return 1;
            }
            board[static_cast<std::size_t>(cell_index)] = cell;
        }
        return is_forbidden_for_black(board, move) ? 1 : 0;
    } catch (const std::exception&) {
        return 1;
    }
}

int tss_legal_move_mask(
    const int* board_array,
    int player,
    int* mask_out
) {
    if (board_array == nullptr || mask_out == nullptr) {
        return -1;
    }

    try {
        Board board {};
        for (int cell_index = 0; cell_index < BOARD_CELLS; ++cell_index) {
            const int cell = board_array[cell_index];
            if (cell < EMPTY || cell > WHITE) {
                return -3;
            }
            board[static_cast<std::size_t>(cell_index)] = cell;
        }

        int resolved_player = player;
        if (resolved_player != BLACK && resolved_player != WHITE) {
            resolved_player = infer_player(board);
        }

        for (int move = 0; move < BOARD_CELLS; ++move) {
            mask_out[move] = is_legal_move_for(board, move, resolved_player) ? 1 : 0;
        }
    } catch (const std::exception&) {
        return -5;
    }

    return 0;
}

int tss_legal_move_mask_batch(
    const int* boards,
    const int* players,
    int num_boards,
    int* masks_out
) {
    if (boards == nullptr || masks_out == nullptr) {
        return -1;
    }
    if (num_boards < 0) {
        return -2;
    }

    try {
        for (int board_index = 0; board_index < num_boards; ++board_index) {
            Board board {};
            const int board_offset = board_index * BOARD_CELLS;
            for (int cell_index = 0; cell_index < BOARD_CELLS; ++cell_index) {
                const int cell = boards[board_offset + cell_index];
                if (cell < EMPTY || cell > WHITE) {
                    return -3;
                }
                board[static_cast<std::size_t>(cell_index)] = cell;
            }

            int player = players == nullptr ? 0 : players[board_index];
            if (player != BLACK && player != WHITE) {
                player = infer_player(board);
            }

            const int mask_offset = board_index * BOARD_CELLS;
            for (int move = 0; move < BOARD_CELLS; ++move) {
                masks_out[mask_offset + move] = is_legal_move_for(board, move, player) ? 1 : 0;
            }
        }
    } catch (const std::exception&) {
        return -5;
    }

    return 0;
}

}  // extern "C"

// CLIエントリーポイント：標準入力からJSONを受け取りTSS評価を実行して結果をJSON形式で出力
int main(int argc, char* argv[]) {
    try {
        // ヘルプオプション（--help / -h）をチェック
        if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            print_usage();                                                  // 使用方法を表示
            return 0;                                                       // 正常終了
        }

        // 標準入力からJSON文字列を全て読み込んでリクエストをパース
        const Request request = parse_request(read_stdin());
        // TSS評価を実行（盤面・プレイヤー・手・探索パラメータを指定）
        const EvaluationResult evaluation = evaluate_tss(
            request.board,
            request.player,
            request.move,
            request.max_depth,
            request.candidate_limit
        );

        // JSON形式で結果を組み立てて標準出力に出力
        std::cout << "{"
                  << "\"score\":" << evaluation.score << ","                // 静的評価スコア
                  << "\"forced_win\":" << (evaluation.forced_win ? "true" : "false") << ","  // 強制勝ちあり
                  << "\"forced_loss\":" << (evaluation.forced_loss ? "true" : "false") << ","  // 強制負けあり
                  << "\"win_depth\":";                                      // 勝ちまでの深さフィールド開始
        if (evaluation.forced_win) {
            std::cout << evaluation.win_depth;                              // 深さが有効なら出力
        } else {
            std::cout << "null";                                            // 強制勝ちなしならnull
        }
        std::cout << ",\"loss_depth\":";                                    // 負けまでの深さフィールド開始
        if (evaluation.forced_loss) {
            std::cout << evaluation.loss_depth;                             // 深さが有効なら出力
        } else {
            std::cout << "null";                                            // 強制負けなしならnull
        }
        std::cout << "}\n";                                                 // JSONオブジェクトを閉じて改行
        return 0;                                                           // 正常終了
    } catch (const std::exception& ex) {
        // パースエラーやその他の例外をキャッチして標準エラー出力に出力
        std::cerr << "error: " << ex.what() << "\n";
        return 1;                                                           // エラーコード1で異常終了
    }
}
