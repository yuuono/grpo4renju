# 強化学習から PPO / GRPO まで

この資料は，強化学習の基本から，Policy Gradient，PPO，GRPO へつなげるためである．最初は一般的な強化学習として説明し，後半で RenjuTransformer にどう対応するかを説明する．

強化学習は，ニューラルネットワークを前提とする手法ではない．
状態数と行動数が小さい問題では，各状態や行動の価値を表として保持し，経験に基づいて更新できる．
代表的な手法として，On-policy の SARSA や Off-policy の Q-Learning がある．
一方，盤面や画像のように状態空間が大きい問題では，すべての状態や行動の価値を表として保持することは難しい．
そこで，価値関数や方策を関数として近似し，その近似器にニューラルネットワークを用いる．
本資料の後半で扱う PPO，GRPO，RenjuTransformer は，このような深層強化学習の文脈に位置づけられる．

## 目次

- [記法](#notation)
- [1. 強化学習の問題設定](#rl-problem)
- [2. 収益と割引率](#return-and-discount)
- [3. 価値関数と advantage](#value-and-advantage)
- [4. ベルマン方程式](#bellman-equation)
- [5. On-policy と Off-policy](#on-policy-and-off-policy)
- [6. 価値ベースと方策ベース](#value-based-and-policy-based)
  - [価値ベース](#value-based)
  - [方策ベース](#policy-based)
- [7. Policy Gradient](#policy-gradient)
  - [コードセクション: REINFORCE を最小構成で実装する](#reinforce-code)
- [8. Generalized Advantage Estimation (GAE)](#gae)
  - [この最小実装を GAE へ拡張するには](#gae-extension)
- [9. TRPO](#trpo)
- [10. PPO](#ppo)
  - [コードセクション: LineWorld を PPO で学習する](#ppo-code)
- [11. GRPO](#grpo)
  - [コードセクション: Renju の trajectory-group GRPO](#grpo-code)
- [14. まとめ](#summary)

<a id="notation"></a>
## 記法

この文書では，対応する確率変数と実現値がある場合，原則として確率変数を大文字，その実現値を対応する小文字で表す．例えば，状態を表す確率変数を $S_t$，実際に観測された状態を $s_t$ と書く．ただし，価値関数 $V_\pi$，行動価値関数 $Q_\pi$，Advantage $A_\pi$，収益 $G_t$ などについては，強化学習で一般的な表記を優先する．同じ文字の場合は，添字，引数，アクセントで意味を区別する．

| 記号                                    | 意味                                     |
| ------------------------------------- | -------------------------------------- |
| $t$                                   | エピソード内の時刻                              |
| $T$                                   | エピソードの終端時刻                             |
| $k$                                   | 将来の時刻または項を数える添字                        |
| $S_t$                                 | 時刻 $t$ の状態を表す確率変数                      |
| $s$                                   | 任意の状態                                  |
| $s'$                                  | 状態 $s$ から遷移した次の状態                      |
| $A_t$                                 | 時刻 $t$ の行動を表す確率変数                      |
| $a$                                   | 任意の行動                                  |
| $R_{t+1}$                             | 時刻 $t$ の行動後に得られる報酬を表す確率変数              |
| $r(s,a,s')$                           | 状態 $s$ で行動 $a$ を取り，状態 $s'$ に遷移したときの報酬  |
| $G_t$                                 | 時刻 $t$ から終端までの割引リターン                   |
| $\tau$                                | 方策と環境によって生成された1本の軌跡                    |
| $\gamma$                              | 将来の報酬に対する割引率                           |
| $\lambda$                             | GAE におけるバイアスと分散の調整係数                   |
| $\epsilon$                            | PPO および GRPO における確率比のクリップ幅             |
| $\beta$                               | KL正則化の係数                               |
| $\delta_t$                            | 時刻 $t$ の Temporal Difference 誤差        |
| $\pi(a\mid s)$                        | 状態 $s$ が与えられたときの行動 $a$に対する方策             |
| $\pi_\theta(a\mid s)$                 | パラメータ $\theta$ を持つ学習対象の方策              |
| $\pi_{\theta_{\rm old}}(a\mid s)$     | データを生成した更新前の方策                         |
| $\pi_{\mathrm{ref}}(a\mid s)$         | KL正則化の基準となる固定方策                        |
| $p(s'\mid s,a)$                       | 状態 $s$ で行動 $a$ を取ったときに状態 $s'$ への遷移確率   |
| $V_\pi(s)$                            | 方策 $\pi$ における状態 $s$ の状態価値関数            |
| $Q_\pi(s,a)$                          | 方策 $\pi$ における状態 $s$，行動 $a$ の行動価値関数     |
| $A_\pi(s,a)$                          | 方策 $\pi$ における状態 $s$，行動 $a$ の advantage |
| $\hat{A}_i$                           | GRPO の候補 $i$ に対する相対 advantage          |
| $\hat{A}_\pi^{\mathrm{MC}}(s_t,a_t)$  | モンテカルロ法による advantage の推定値              |
| $\hat{A}_\pi^{\mathrm{GAE}}(s_t,a_t)$ | GAE による advantage の推定値                 |
| $b(S_t)$                              | 方策勾配の分散を低減する baseline関数                |
| $J_\pi(\theta)$                       | 方策 $\pi_\theta$ の期待リターンを表す目的関数         |
| $L_\pi(\theta)$                       | 方策勾配法で用いる方策損失                          |
| $\rho_\theta(s,a)$                    | TRPO および PPO における新旧方策の確率比              |
| $r_i(\theta)$                         | GRPO の候補 $i$ に対する新旧方策の確率比              |
| $D_{\mathrm{KL}}(p\mathbin{\|}q)$     | 確率分布 $p$ と $q$ の KLダイバージェンス            |

<a id="rl-problem"></a>
## 1. 強化学習の問題設定

強化学習では，エージェントが環境に対して行動し，その結果として次の状態と報酬を受け取る．エージェントの目的は，将来得られる報酬の合計が大きくなるような行動の選び方を学習することである．基本的な流れは次のようになる．
```math
\begin{gathered}
状態 S_t を観測する \\
 \downarrow \\
方策 \pi に従って行動 A_t を選ぶ \\
 \downarrow \\
環境が次の状態 S_{t+1} と報酬 R_{t+1}を返す \\
 \downarrow \\
これを終端 まで繰り返す \\
\end{gathered}
```
このように，次の状態や報酬が現在の状態と行動のみで決定される時系列のことを，マルコフ決定過程（Markov Decision Process, MDP）と呼ぶ．また方策 $\pi(a | s)$ は、状態 $s$ が与えられたときの行動 $a$ に対する条件付き確率分布とする．
```math
\begin{align}
S_t&: 現在の状態 \\
A_t&: 現在の行動 \\
R_{t+1}&: 行動後に得られる報酬 \\
S_{t+1}&: 次の状態 \\
\pi(A_t=a | S_t=s)&: 状態 s で行動 a を選ぶ方策 \\
\end{align}
```
このようにして，軌跡(trajectory)が
```math
\tau = \left( S_0​,A_0​,R_1​,S_{1​},A_{1}​,R_2​​,\ldots,S_t​,A_t,R_{t+1}​​,\ldots,R_{T},S_{T}​\right)
```
生成される．この書き方がプログラムに馴染む定義であり，こちらを採用する．
エージェントは現在の状態 $s$ を観測し，方策 $\pi(a \mid s)$によって，行動 $a$ を選択する．そして，MDPを仮定すると，状態遷移確率 $p(s' \mid s, a)$ によって次の状態 $s'$ へと遷移し，報酬関数 $r\left(s,a,s'\right)$ により報酬が得られる．

<a id="return-and-discount"></a>
## 2. 収益 と割引率

終端時刻 $T$ までの軌跡 $\tau$ に対して，ある時刻 $t$ から将来得られる報酬の合計をリターンまたは収益と呼び，
```math
G_{t}​​ = \gamma^0 R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1} R_{T}
```
と書く．和の記号で書くと
```math
\begin{align}
G_{t} & = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \\
\end{align}
```
であり，初期状態 $(t=0)$ から将来得られる報酬の合計
```math
\begin{align}
G_{0} & = \sum_{k=0}^{T-1} \gamma^k R_{k+1}
\end{align}
```
となり，再帰的には
```math
G_{t} = R_{t+1} + \gamma\ G_{t+1}
```
とかける．ここで $\gamma$ は割引率である．
- $\gamma = 1.0$: 将来の報酬を弱めない
- $\gamma = 0.99$: 遠い将来の報酬を少し弱く見る
- $\gamma = 0$: 次の報酬だけを見る

割引率を使う理由は，遠い未来の報酬ほど不確実であり，現在の行動との因果関係も弱くなりやすいからである．

<a id="value-and-advantage"></a>
## 3. 価値関数 と advantage 

強化学習では，「ある状態がどれくらい良いか」を表す関数を価値関数と呼ぶ．
状態価値関数 $V_\pi(s)$は，状態 $s$ から方策 $\pi$ に従って行動した時のリターンの期待値である．
```math
\begin{align}
V_\pi(s) & := \mathbb{E}_\pi\left[G_t\mid S_t=s\right]　\\
&= \mathbb{E}_\pi\left[\sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \mid S_t=s \right]
\end{align}
```
一方で，行動価値関数 $Q_\pi(s, a)$ は，状態 $s$ で特定の行動 $a$ を取った時のリターンの期待値である．
```math
\begin{align}
Q_\pi(s, a) & := \mathbb{E}_\pi\left[G_t \mid S_t=s, A_t=a \right] \\
& = \mathbb{E}_\pi\left[\sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \mid S_t=s, A_t=a \right] \\
& = \mathbb{E}[R_{t+1} \mid S_t=s,A_t=a]+\gamma\ \mathbb{E}_\pi[G_{t+1} \mid S_t=s, A_t=a] \\
\end{align}
```
ここで， $Q_{\pi(s, a)}$ の行動 $a$ は，方策 $\pi$ には関係ないことに注意する．
$Q_\pi(s, a)$ の行動 $a$ は自由に決定することができ，その行動の後は，方策 $\pi$ に従って行動する．
最終行では，報酬 $R_{t+1}$ の期待値は方策 $\pi$ と無関係となるため除いた．

この二つの価値関数の違いは
- 状態価値関数 $V_\pi(s)$: この状態 $s$ はどれくらい良いか
- 行動価値関数 $Q_\pi(s, a)$: この状態 $s$ でこの行動 $a$ はどれくらい良いか
であり，
状態価値関数 $V_\pi(s)$では，行動 $a$ は方策 $\pi$ に従って選ばれ，
行動価値関数 $Q_\pi(s,a)$ では，行動は自由に選ぶことができる点が相違点となる．
このため，行動価値関数 $Q_\pi(s, a)$ の行動を方策 $\pi$ に従って選ぶとしたら，
行動価値関数 $Q_\pi(s, a)$ と状態価値関数 $V_\pi(s)$ は同じになる．そのため，状態価値関数 $V_\pi$ は，その状態で取り得る行動価値関数 $Q_\pi(s,a)$ を方策で平均したものという次の式が成り立つ．
```math
V_\pi(s) = \sum_a \pi(a \mid s) Q_\pi(s, a)
```
この二つの価値関数より，advantage関数 $A_\pi(s,a)$ を
```math
A_{\pi}\left(s, a\right) = Q_\pi(s, a) - V_\pi(s)
```
と定義する．この式は，特定の行動 $a$ を選んだ場合の価値から方策 $\pi$が選ぶ行動の平均的な良さを引いた値を示す．そのため，advantage関数は，特定の行動 $a$ が平均的な行動よりどれくらい良かったかを表す量となる．
$A_{\pi} > 0$ なら，その行動は平均より良い．
$A_{\pi} < 0$ なら，その行動は平均より悪いということになる．advantage を使うことで，「絶対的に報酬が高いか」ではなく，「その状態の中で相対的に良い行動か」を見やすくなる．単純に 収益 $G_t$ だけを使うと，報酬のばらつきが大きくなりやすい．

<a id="bellman-equation"></a>
## 4. ベルマン方程式

状態価値関数は以下のように，「即時報酬」と「次の状態の価値」に分解でき，分解後の式をベルマン方程式と呼ぶ．
```math
\begin{align}
V_\pi(s)
& = \sum_a \pi(a \mid s) Q_\pi(s, a)\\
& = \sum_a \pi(a \mid s)\ \mathbb{E}_\pi[R_{t+1} + \gamma\ G_{t+1} \mid S_t=s, A_t=a] \\
& = \sum_a \pi(a \mid s)\ \mathbb{E}[R_{t+1}\mid S_t=s, A_t=a]  + \gamma \sum_a \pi(a \mid s)\ \mathbb{E}_\pi [G_{t+1} \mid S_t=s, A_t=a] \\
& = \sum_a \pi(a \mid s)
\sum_{s'} p(s' \mid s, a)
\left(r(s, a, s') + \gamma V_\pi(s')\right)
\end{align}
```
この導出は，前の2式を組み合わせることで可能で，
状態価値関数におけるベルマン方程式は「状態 $s$ の価値関数」と「その次に取りうる $s'$ の価値関数」との関係を示し，
```math
\begin{align}
状態 s の価値 = & その状態で各行動を選ぶ確率 \times \\
&   その行動で次の状態に移る確率 \times \\
&   「即時報酬 + 次の状態の価値」 \quad の合計
\end{align}
```

```math
V_\pi(s)
=
\overbrace{\sum_a
\underbrace{\pi(a|s)}_{\text{方策の確率}}
\underbrace{
\sum_{s'}
\underbrace{p(s'|s,a)}_{\text{次の状態の確率}}
\left(
\underbrace{r(s,a,s')}_{\text{即時報酬}}
+
\overbrace{\gamma}^{\text{割引率}}
\underbrace{V_\pi(s')}_{\text{次の状態の価値}}
\right)
}_{\text{行動価値 } Q_\pi(s,a)}
}^{\text{状態価値} V_\pi(s)}
```
となる．行動価値関数 $Q(s,a)$におけるベルマン方程式も以下のように書ける．
```math
Q_\pi(s, a)
= \sum_{s'} p(s' \mid s, a)
\left(r(s, a, s') + \gamma V_\pi(s')\right)
```
これら二つのベルマン方程式から最適ベルマン方程式を導出し，最適行動または価値関数から，最適方策を求めるのが価値ベースの手法である

<a id="on-policy-and-off-policy"></a>
## 5. On-policy と Off-policy

強化学習では，「学習で更新する方策（target policy）」と「学習用データを集める時の行動方策（behavior policy）」という二つの方策が必要となる．強化学習は，方策を更新する際にどの方策から集めたデータを使うかで On-policy （方策オン）と Off-policy（方策オフ）の二つに分けられる．On-policy は，target policy と behavior policy が同じである訓練手法である．つまり，
以下の図のように，現在の方策で集めたデータを使って，方策を更新することであり，自分の経験から自分を更新する．On-policyの手法は，データ効率が悪くなりやすいが，方策の更新方向が現在の方策に対応しているため，目的関数の意味は比較的明確である．
```math
\begin{gathered}
現在の方策で行動をサンプル \\
\downarrow \\
報酬を得る \\
\downarrow \\
そのデータで同じ方策を更新 \\
\end{gathered}
```

一方で，Off-policy は，target policy と behavior policy が異なる訓練手法である．つまり，現在の方策とは違う方策で集めたデータで訓練を行い，他人の経験から自分を更新する．
このため，過去の経験を保存し，訓練に再利用可能でデータ効率が良い．

<a id="value-based-and-policy-based"></a>
## 6. 価値ベースと方策ベース

強化学習の方法は大きく分けると，価値ベースと方策ベースに分けられる．
<a id="value-based"></a>
### 価値ベース
価値ベースでは，「行動そのもの」ではなく「その行動を取ったときの良さ・価値」，状態or行動価値関数を学習し，価値が最大になるように行動を選んでいく．
代表的なものとして，状態 $s$ で行動 $a$ を取ったときの価値関数 $Q(s, a)$ を学習する手法が挙げられる．
つまり，「この状態でこの行動を取ると，将来的にどれくらい報酬が得られそうか」を数値として学習していく．
そして，過去に経験した軌跡 $(s, a, r, s')$ を使用し，価値関数が更新される．
そのため，価値関数が更新されると，同じデータでも違う結果となり，同じデータでも再び学習使用可能となる．また，価値ベースは，最適行動を求めるために，全ての可能な行動を確認する必要があり
```math
a^* = \arg \underset{a}{\max} Q_\pi(s, a)
```
を計算する．このため，行動候補が離散的で，最善手がはっきりしている問題に強い．しかしながら，行動空間が非常に大きい場合や連続値の場合に扱いづらい．例えば行動空間が $-1 \le a \le 1$ だと，行動 $0.999$ の価値， $0.998 $ の価値を計算する必要があり，困難である．さらに，最適行動決定の際，価値最大化を行うために，価値の過大評価が起きやすく，価値の誤差が行動選択に直接影響してしまう．シンプルな設定では理論的な収束性を議論しやすいという利点もある．代表例は Q-Learning，DQN，SARSA である．

<a id="policy-based"></a>
### 方策ベース
方策ベースでは，行動を選ぶ方策 $\pi_{\theta}$ ，「状態 $s$ で，どの行動 $a$ をどれくらいの確率で選ぶか」を学習している．これにより，最終的に欲しい方策を直接学習できる点， 確率的な方策や連続的な行動を自然に扱える点が優れている点となる．
一方で，現在の方策 $\pi_{\theta}$ で行動し，得られた報酬で更新を行うため，現在の方策で集めたデータが重要となり，On-policyになりやすく，データ効率が悪くなりやすい．例えば，報酬が大きかったり，小さかったりすると，勾配が振り幅が大きくなるため，学習が不安定となる．そして，
方策ベースの手法では，基本的に，将来報酬の期待値を最大化するため，どの行動が本当に良かったのかという credit assignment が難しいという点もある．代表例は REINFORCE，actor-critic，PPO，GRPO である．

<a id="policy-gradient"></a>
## 7. Policy Gradient

方策ベースの基本は，方策 $\pi_\theta(a \mid s)$ のパラメータ $\theta$ を直接更新することである．
目的は，方策 $\pi_\theta$ に従って生成された軌跡 $\tau$ のリターンの期待値の最大化 $\max J_\pi(\theta)$ 

```math
\begin{align}
J_\pi(\theta) & = \mathbb{E}_{\tau \sim \pi_\theta}[G_0] \\
& = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{k=0}^{T} \gamma^k R_{k+1}
\right]
\end{align}
```
であり，方策勾配定理より，この目的関数における方策勾配は， 
```math
\nabla_\theta J_\pi(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ G_0
\right]
```
書くことができる．詳細な導出に関しては，他の文献を参照してください．ここで，
```math
\nabla_\theta \log \pi_\theta(a_t \mid s_t) = \dfrac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}
```
となる．
ここで $\nabla_\theta \pi_\theta(a_t \mid s_t)$ は，方策の確率 $\pi_\theta(a_t \mid s_t)$ を大きくする方向となる．そのため，方策勾配とは，「良かった行動の確率を上げ，悪かった行動の確率を下げる」方向となる．
収益 $G_0$ が正なら $\log\pi_\theta(a_t \mid s_t)$ を大きくする方向に更新し，収益 $G_0$ が負なら小さくする方向に更新する．そのため，報酬の分散が大きいと，それが直接更新方向に影響を及ぼすため，学習が不安定になりやすい．そこで，報酬の分散を減らすようにしたのが，REINFORCEやbaseline導入やactor-critic という手法である．
REINFORCEでは，目的関数の一部である $G_0$ を $G_{t}$ に変更した手法である．
収益 $G_0$ は初期状態から将来 $(t=T)$ までの報酬の合計期待値となる．しかし，時刻 $t$ の現在状態において，初期状態から現在時刻までの報酬は，エージェントが取りうる行動の良し悪しとは関係ない．現在から将来までの報酬の合計期待値 $G_{t}$ を利用しても，方策勾配
```math
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ G_0
\right]
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ G_t\right]
```
が等価となることを証明した．時刻 $t$ について，軌跡全体の報酬を
```math
G_0 = \sum_{k=0}^{T-1} \gamma^k R_{k+1}= \sum_{k=0}^{t-1} \gamma^k R_{k+1} + \sum_{k=t}^{T-1} \gamma^k R_{k+1}
```
と分解することより，
```math
\begin{align}
\nabla_\theta J(\theta)
& = \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ G_0
\right]\\
& = \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ \sum_{k=0}^{t-1} \gamma^k R_{k-1} 
\right] + \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ \sum_{k=t}^{T} \gamma^k R_{k-1} 
\right]\\
& = 0 + \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ G_t\right]
\end{align}
```
となる．ここで，第1項が0となるのは Expected Grad-Log-Prob (EGLP) lemma より[こちら](https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof1.html)を参照．
$G_t$ は，現在からの報酬となるので，the **reward-to-go** from that pointと呼ばれる．
さらに，(EGLP) lemmaより，行動 $a$に依存しない関数であればどのような値を引いても方策勾配は変化しないことにより，以下のようにbaseline関数 $b(S_t)$ を導入したもの
```math
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ 
\left( G_t- b\left(S_t\right) \right)
\right]
```
がある．ここで，baseline関数に状態価値関数 $V_\pi$ を用いると，方策勾配
```math
\begin{align}
\nabla_\theta J(\theta)
& = \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ 
\left( G_t- V_\pi\left(S_t\right) \right)
\right] \\
\end{align}
```
として，現在の状態に対する報酬を引いて，選択した行動の価値のみを方策勾配に反映できる．
他の方策勾配としては，行動価値関数 $Q_\pi$を用いた
```math
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ Q_\pi\left(s_t, a_t\right)\right]
```
ものや， advantage 関数を用いた
```math
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\ A_\pi\left(s_t, a_t\right)
\right]
```
ものがある．また，方策を勾配によって急激に変更させると，学習が崩壊するため，勾配を調整して，進ませることが重要である．一方で，少しずつしか更新できないため，局所最適解に陥りやすいという欠点もある．これをプログラムで実装する際には，方策損失（policy loss）として
```math
L_\pi(\theta) = - \mathbb{E}_{\tau \sim \pi_{\theta_{\rm old}}}
\left[ \log \pi_\theta(a_t \mid s_t)\ A_\pi\left(s_t, a_t\right)
\right]
```
とすることで，自動微分によって方策勾配を計算する仕組みとなっている．訓練上で注意すべきは，方策勾配法でのロス関数は，通常の教師あり学習のロスとはかなり意味が異なる．強化学習において本当に最大化したいものは 収益期待値 $J_\pi(\theta)$ である．訓練実装上，上記のようなadvantage関数を用いた方策勾配で更新する時は，方策損失を
```math
L_\pi(\theta) = - \mathbb{E}_{\tau \sim \pi_{\theta_{\rm old}}}
\left[ \log \pi_\theta(a_t \mid s_t)\ A_\pi\left(s_t, a_t\right)
\right]
```
と定義して，最小化する．
この損失関数は通常の損失関数とは異なる．異なる点は，データ分布がパラメータに依存することと，この損失がモデル性能を計測していないことの2点である．
まず，強化学習において，方策 $\pi_\theta$が更新されると，エージェントの行動が変化し，訪れる状態も変わる．従って，収集されるデータそのものが変化する，データ分布がパラメータに依存しているということになる．現在の方策で集めたデータを使い，現在のパラメータで評価したときのみ，その勾配が方策損失を改善する方向を教えてくれるものであり，モデル性能を評価しているものではないということに注意するべき．そして，この方策損失はいくらでも小さくできるかもしれないが，その結果として実際の方策性能は崩壊することがある．
このように，方策勾配法はOn-policy で，学習が不安定になりやすいことから，安定した更新手法の研究が進んだ．他には，Natural policy gradientという方策の自然勾配を使用する手法や，Trust Region Policy Optimization (TRPO)がある．

<a id="reinforce-code"></a>
### コードセクション: REINFORCE　を最小構成で実装する

ここでは，方策勾配法を自分で実装できるようになることを目的として，外部の強化学習環境に依存しない小さな例を紹介すする．
扱う環境は `LineWorld` とする．エージェントは一次元のマス上におり，左端へ到達すると $-1$，右端へ到達すると $+1$ の報酬を得る．

| 状態 | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 位置 | 左端 |  | start |  | 右端 |
| 報酬 | -1 | 0 | 0 | 0 | +1 |

```text
行動0: 左へ移動
行動1: 右へ移動
開始位置: 状態2（start）
ゴール条件: 状態0または状態4に到達したら終了、など
報酬設計: 左端（状態0）で-1、右端（状態4）で+1
```

実装する処理は次の7段階である．

1. 方策 $\pi_\theta(a\mid s)$ から行動をサンプリングする
2. 環境を1ステップ進めて報酬を得る
3. 終端まで繰り返してtrajectoryを保存する
4. 各時刻のreward-to-go $G_t$ を計算する
5. baselineを引いてadvantageを作る
6. $-\log\pi_\theta(a_t\mid s_t)A_t$ を平均する
7. `backward()` と `optimizer.step()` で方策を更新する

以下は，そのまま1ファイルとして実行できるPyTorchコードである．

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


torch.manual_seed(0)


class LineWorld:
    """0が負け、4が勝ちとなる5マスの環境。"""

    def __init__(self, max_steps: int = 8) -> None:
        self.max_steps = max_steps
        self.position = 2
        self.steps = 0

    def reset(self) -> int:
        self.position = 2
        self.steps = 0
        return self.position

    def step(self, action: int) -> tuple[int, float, bool]:
        # 環境の状態遷移には勾配を流さない。
        # 行動0なら左へ移動, 1なら右へ移動
        self.position += -1 if action == 0 else 1
        self.steps += 1

        # 状態0または状態4に到達したら終了
        if self.position == 4:
            return self.position, 1.0, True
        if self.position == 0:
            return self.position, -1.0, True

        truncated = self.steps >= self.max_steps
        return self.position, -0.01, truncated


class Policy(nn.Module):
    """状態sを、左右2行動のlogitsへ変換する方策 π_θ(a|s)。"""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Embedding(num_embeddings=5, embedding_dim=16),
            nn.Linear(16, 2),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


@dataclass
class Episode:
    states: list[int]
    actions: list[int]
    rewards: list[float]


def collect_episode(env: LineWorld, policy: Policy) -> Episode:
    """更新前の現在方策を使って、trajectoryを1本収集する。"""
    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []

    state = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor([state], dtype=torch.long)

        # trajectory収集中にはパラメータ更新を行わない。
        with torch.no_grad():
            logits = policy(state_tensor)
            # logitsから、各選択肢のカテゴリカル分布（離散確率分布）作成
            # .sample()	確率に従って選択肢をランダムに選ぶ
            # .log_prob(x)	選んだ選択肢の対数確率を返す
            distribution = Categorical(logits=logits)
            action = int(distribution.sample().item())

        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    return Episode(states=states, actions=actions, rewards=rewards)


def reward_to_go(rewards: list[float], gamma: float) -> torch.Tensor:
    """G_t = R_{t+1} + γ G_{t+1} を後ろから計算する。"""
    returns = [0.0] * len(rewards)
    running_return = 0.0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return torch.tensor(returns, dtype=torch.float32)


def train_step(
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    episodes: list[Episode],
    gamma: float,
) -> dict[str, float]:
    """収集済みtrajectoryから、方策を1回更新する。"""
    all_states: list[int] = []
    all_actions: list[int] = []
    all_returns: list[torch.Tensor] = []

    for episode in episodes:
        all_states.extend(episode.states)
        all_actions.extend(episode.actions)
        all_returns.append(reward_to_go(episode.rewards, gamma))

    states = torch.tensor(all_states, dtype=torch.long)
    actions = torch.tensor(all_actions, dtype=torch.long)
    returns = torch.cat(all_returns)

    # b(S_t) の簡単な例として、バッチ内リターンの平均を使う。
    baseline = returns.mean()
    advantages = returns - baseline

    # advantageは報酬から作った学習ターゲットであり、
    # 方策側から勾配を流してはいけない。
    advantages = advantages.detach()

    # 現在の状況から方策
    logits = policy(states)
    distribution = Categorical(logits=logits)

    # log π_θ(a_t|s_t): 実際に選択した行動の対数確率だけを取り出す。
    selected_log_probs = distribution.log_prob(actions)

    # L_π(θ) = -E[log π_θ(a_t|s_t) A_t]
    # 最大化を行いたいので、マイナスをつけて最小化を行う。
    policy_loss = -(selected_log_probs * advantages).mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # エピソード単位での報酬の合計
    mean_episode_return = sum(sum(ep.rewards) for ep in episodes) / len(episodes)
    return {
        "policy_loss": float(policy_loss.item()),
        "mean_episode_return": mean_episode_return,
    }


def action_probabilities(policy: Policy, state: int = 2) -> list[float]:
    with torch.no_grad():
        # Batch_size, 2（出力次元、右か左の行動ロジット）
        logits = policy(torch.tensor([state], dtype=torch.long))
        # squeeze(0)（バッチ次元を削除） tolist()（Pythonのリストへ）
        return torch.softmax(logits, dim=-1).squeeze(0).tolist()


def main() -> None:
    env = LineWorld()
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-2)

    for iteration in range(300):
        # 更新するたびに、更新後の方策で新しいデータを集め直す。
        episodes = [collect_episode(env, policy) for _ in range(32)]
        metrics = train_step(policy, optimizer, episodes, gamma=0.99)

        if iteration % 50 == 0:
            left_prob, right_prob = action_probabilities(policy)
            print(
                f"iteration={iteration:3d} "
                f"loss={metrics['policy_loss']:+.4f} "
                f"return={metrics['mean_episode_return']:+.3f} "
                f"P(left)={left_prob:.3f} "
                f"P(right)={right_prob:.3f}"
            )


if __name__ == "__main__":
    main()
```

このコードでは，数式と変数が次のように対応している．

| 数式の要素 | コード | 実装上の役割 |
| --- | --- | --- |
| $\theta$ | `policy.parameters()` | 最適化する方策のパラメータ |
| $s_t$ | `episode.states[t]` | 行動を選ぶ直前の状態 |
| $\pi_\theta(a\mid s)$ | `Categorical(logits=policy(states))` | 状態ごとの行動確率分布 |
| $a_t$ | `episode.actions[t]` | 方策から実際にサンプルした行動 |
| $R_{t+1}$ | `episode.rewards[t]` | 行動後に環境が返した報酬 |
| $G_t$ | `reward_to_go(...)` | 時刻 $t$ 以降の割引リターン |
| $b(S_t)$ | `returns.mean()` | 分散を抑えるためのbaseline |
| $\hat A_t$ | `returns - baseline` | 今回の更新で使うadvantage推定値 |
| $\log\pi_\theta(a_t\mid s_t)$ | `distribution.log_prob(actions)` | 選択行動の対数確率 |
| $\mathbb{E}[\cdot]$ | `.mean()` | サンプル平均による期待値の近似 |
| $-\mathbb{E}[\log\pi_\theta A]$ | `policy_loss` | 最小化する方策損失 |
| $\nabla_\theta L$ | `policy_loss.backward()` | 自動微分による勾配計算 |
| パラメータ更新 | `optimizer.step()` | 方策を勾配方向へ更新 |

この例では， $M$ 本のtrajectoryに含まれる総着手数を

```math
N=\sum_{i=1}^{M}T_i
```

として，理論上の期待値を次の有限サンプル平均で近似している．

```math
\hat L_\pi(\theta)
=-\frac{1}{N}
\sum_{i=1}^{M}\sum_{t=0}^{T_i-1}
\log\pi_\theta(a_{i,t}\mid s_{i,t})
\left(G_{i,t}-\bar G\right)
```

コードでは各episodeの `states`，`actions`，`returns` を平坦化してから `.mean()` しているため，この式と一致する．ここで $\bar G$ はバッチ内の全リターンの平均であり，状態に依存しないbaselineである．

<a id="reinforce-two-forward-passes"></a>
#### なぜ収集時と更新時に方策を2回計算するのか

`collect_episode()` では，状態・行動・報酬だけを保存し，計算グラフを保持しない．そのため，trajectory収集時の推論は `torch.no_grad()` で実行している．更新時の `train_step()` で保存した状態をもう一度方策へ入力し，勾配を持つ `selected_log_probs` を計算する．

```text
収集: policy(state) → actionをサンプル → state, action, rewardを保存
更新: policy(saved_states) → 選択行動のlog probabilityを再計算 → backward
```

収集と再計算の間に `optimizer.step()` を実行していないため，どちらも同じ方策 $\pi_\theta$ である．途中で方策を更新する場合は，収集時の方策と更新時の方策が異なってしまうため，vanillaなOn-policy方策勾配としては扱えなくなる．PPOでは，この違いを新旧方策の確率比によって明示的に補正する．

<a id="reinforce-no-argmax"></a>
#### なぜ行動のサンプリングに `argmax` を使わないのか

方策勾配法では，方策が与える確率分布から行動をサンプリングする必要がある．

```python
action = distribution.sample()
```

とすることで，確率が低い行動にも試行機会が残る．`argmax` を使うと常に同じ行動が選ばれやすくなり，未経験の行動の報酬を観測できない．

<a id="reinforce-no-env-backprop"></a>
#### 環境の状態遷移に逆伝播しなくてよい理由

`env.step(action)` は整数の状態を更新しており，微分可能ではない．しかし，方策勾配法では環境を通して逆伝播する必要はない．勾配を流す対象は，

```python
selected_log_probs = distribution.log_prob(actions)
```

で計算した方策の対数確率である．環境から得た報酬は `advantages` としてその対数確率に掛けられ，どの行動を強化または抑制するかを決める．

<a id="reinforce-detach-advantages"></a>
#### `advantages.detach()` が必要な理由

この例のadvantageは環境報酬だけから計算しているため，もともと勾配を持たない．それでも `detach()` を明示することで，「advantageは方策を更新するための固定ターゲットである」という実装上の境界を表している．Actor-Criticで価値モデルからadvantageを計算する場合は，この切り離しを忘れると方策損失から価値モデルへ意図しない勾配が流れる可能性がある．

<a id="reinforce-actor-critic"></a>
#### この最小実装をActor-Criticへ拡張するには

現在は，

```python
baseline = returns.mean()
```

という状態に依存しないbaselineを使用している．これを状態価値モデル，

```python
values = value_model(states)
advantages = returns - values.detach()
```

に置き換えるとActor-CriticのActor側になる．Valueモデルは別に，

```python
value_loss = torch.nn.functional.mse_loss(values, returns)
```

で学習する．このとき，

```python
total_loss = policy_loss + value_coef * value_loss
```

として同時に更新することも，optimizerを分けて更新することもできる．この拡張でも，環境の状態遷移そのものに勾配を流す必要はない．

<a id="reinforce-checklist"></a>
#### 実装時の確認項目

- trajectory収集中にoptimizerを更新していないか
- 更新後は新しい方策でtrajectoryを集め直しているか
- `log_prob`は全行動ではなく，実際に選択した行動について取得しているか
- reward-to-goを終端側から逆順に計算しているか
- advantageへ方策側の勾配が流れていないか
- 損失の先頭にマイナス符号があるか
- 学習中は`argmax`ではなく確率分布からサンプリングしているか

実行すると，乱数seedを固定したこの例では，おおむね次のように右行動の確率が増加する．

```text
iteration=  0 ... return=-0.495 P(left)=0.448 P(right)=0.552
iteration= 50 ... return=+0.990 P(left)=0.000 P(right)=1.000
```

この出力で見るべき値は `policy_loss` の減少ではなく，`mean_episode_return` と正解行動の確率 `P(right)` の増加である．方策損失は，異なるiteration間でモデル性能を直接比較する指標ではない．



<a id="gae"></a>
## 8. Generalized Advantage Estimation (GAE)

実際に，方策損失
```math
L_\pi(\theta) = - \mathbb{E}_{\tau \sim \pi_{\theta_{\rm old}}}
\left[ \log \pi_\theta(a_t \mid s_t)\ A_\pi\left(s_t, a_t\right)
\right]
```
を最小化する際，advantage 関数 $A_\pi\left(s_t, a_t\right)=Q_\pi(s, a) - V_\pi(s)$ を推定することが求められる．
この時のadvantage 関数の推定手法について説明する．
まず，Temporal Difference 誤差 (TD residual) 
```math
\delta_t = r_{t+1} + \gamma V_\pi(s_{t+1}) - V_\pi(s_t) 
```
を用いた手法である．なぜなら，TD 誤差がadvantage 関数の不偏推定量
```math
\begin{align}
\mathbb{E}_{s_{t+1} \sim p\left(\cdot | s_t, a_t\right)} \left[r_{t+1} + \gamma V_\pi(s_{t+1}) - V_\pi(s_t)\right] & = \mathbb{E}_{s_{t+1} \sim p\left(\cdot | s_t, a_t\right)} \left[Q_\pi(s_t, a_t) - V_\pi(s_t)\right]= A_\pi(s_t, a_t) \\
\end{align}
```
である．TD誤差とは，環境モデルを使用せず，行動を１つ行う度に価値関数を更新する手法であるTD法に基づく概念である．
TD法では，現在の状態価値関数 $V_\pi(s_t)$ を，現在の報酬 $r_t$ と次の状態価値関数を用いて，サンプル近似を行う．そして，この近似をTD誤差で評価し，状態価値関数を更新することで，適当な状態価値関数を計算し，方策の評価を行う．このように，推定値を使って別の推定値を更新することを ブートストラップ と言う．
このTD法は，現在の状態価値関数を，1ステップ先の行動で評価するものであり，
価値関数における分散が低い，すぐ次の報酬しか見ないので安定しやすいという利点と，価値関数の推定が間違っているとバイアスが大きくなるという欠点が存在する．そこで，これは2，3ステップのように，伸ばすことでより良くすることが可能になると考えられる．
一方で，エピソード終了までの報酬を全部使い，収益を推定する手法もある．これはモンテカルロ (MC) 推定と呼ばれる．MC法では，
```math
\begin{align}
\hat{A}^{MC}_\pi(s_t, a_t) & = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1} - V_\pi(s_t)\\
\end{align}
```
と推定を行い，実際に得られた将来報酬を使用し，長期的な結果をよく反映する．一方で，多くのランダムな報酬を合計するため分散が大きく，学習が不安定になりやすい．TD誤差は低分散だがバイアスが大きくなりやすく，MC推定は低バイアスだが高分散になりやすい．
この二つの間を取って，N-step advantage 推定という，Nステップまで実際の報酬を用い，Nステップ以降は，推定値を用いてブートストラップを行う手法もある．しかし，このようにバイアスと分散のトレードオフを調整する必要があり，Nステップを一つ決定する必要がある．
そこで提案されたのが， Generalized Advantage Estimation(GAE)で，
```math
\begin{align}
\hat{A}^{GAE}_\pi(s_t, a_t) & = \sum_{k=0}^{T-t-1} (\gamma\lambda)^k \delta_{t+k} \\
where\  \delta_t & = r_{t+1} + \gamma V_\pi(s_{t+1}) - V_\pi(s_t)\end{align}
```
のように，将来のTD誤差を遠いものほど小さい重みで，いろいろなNステップ advantage 推定を重み付き平均する．
ここで，重み係数 $\lambda$ を0から1まで変化させることで，バイアスと分散のトレードオフを調整可能．
重み係数 $\lambda$ が小さいと，TD法に近づき，低分散・高バイアスで
重み係数 $\lambda$ が大きいと，MC法に近づき，高分散・低バイアスとなる．

<a id="gae-extension"></a>
#### この最小実装をGAEへ拡張するには

前節のREINFORCE実装では，

```python
returns = reward_to_go(episode.rewards, gamma)
advantages = returns - baseline
```

として，終端までの実報酬からadvantageを作った．GAEへ拡張するには，状態価値モデル $V_\phi(s)$ を追加し，

```math
\begin{align}
\delta_t
&=r_{t+1}+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t),\\
\hat A_t^{\mathrm{GAE}}
&=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}^{\mathrm{GAE}}
\end{align}
```

をtrajectoryの終端側から計算する．ここで $d_t$ は行動後にepisodeが終了した場合に1，継続する場合に0となる終端フラグである．$(1-d_t)$ を掛けることで，終端状態の先に存在しない価値をbootstrapしないようにする．

まず，価値モデルを追加する．

```python
class ValueModel(nn.Module):
    """状態sから状態価値 V_φ(s) を出力するCritic。"""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Embedding(num_embeddings=5, embedding_dim=16),
            nn.Linear(16, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)
```

GAEでは $s_t$ だけでなく $s_{t+1}$ と $d_t$ が必要になる．したがって，`Episode`へ次の状態と終端フラグを追加する．

```python
@dataclass
class Episode:
    states: list[int]
    actions: list[int]
    rewards: list[float]
    next_states: list[int]
    dones: list[bool]
```

`collect_episode()` でも，環境から返された値を保存する．

```python
def collect_episode(env: LineWorld, policy: Policy) -> Episode:
    states: list[int] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[int] = []
    dones: list[bool] = []

    state = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor([state], dtype=torch.long)
        with torch.no_grad():
            distribution = Categorical(logits=policy(state_tensor))
            action = int(distribution.sample().item())

        next_state, reward, done = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        state = next_state

    return Episode(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )
```

次に，GAEを計算する関数を作る．

```python
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1本のtrajectoryについてGAEとValue学習ターゲットを返す。"""
    not_done = 1.0 - dones

    # δ_t = r_{t+1} + γ(1-d_t)V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * not_done * next_values - values

    advantages = torch.zeros_like(rewards)
    running_advantage = torch.tensor(0.0, dtype=rewards.dtype)

    # A_t = δ_t + γλ(1-d_t)A_{t+1} を終端側から計算する。
    for t in reversed(range(len(rewards))):
        running_advantage = (deltas[t] + gamma * gae_lambda * not_done[t] * running_advantage)
        advantages[t] = running_advantage

    # A_t = Q_t - V(s_t) より、Criticの教師信号は A_t + V(s_t)。
    value_targets = advantages + values
    return advantages, value_targets
```

数式とコードは次のように対応する．

| 数式 | コード |
| --- | --- |
| $r_{t+1}$ | `rewards[t]` |
| $V_\phi(s_t)$ | `values[t]` |
| $V_\phi(s_{t+1})$ | `next_values[t]` |
| $d_t$ | `dones[t]` |
| $\delta_t$ | `deltas[t]` |
| $\gamma\lambda$ | `gamma * gae_lambda` |
| $\hat A_t^{\mathrm{GAE}}$ | `advantages[t]` |
| $\hat A_t+V_\phi(s_t)$ | `value_targets[t]` |

最後に，REINFORCE版の `train_step()` をActorとCriticの同時更新へ置き換える．GAEの再帰計算はepisode境界をまたいではいけないため，最初にepisodeごとに計算し，その後でテンソルを連結する．

```python
def train_step_with_gae(
    policy: Policy,
    value_model: ValueModel,
    optimizer: torch.optim.Optimizer,
    episodes: list[Episode],
    gamma: float,
    gae_lambda: float,
    value_coef: float = 0.5,
) -> dict[str, float]:
    all_states: list[torch.Tensor] = []
    all_actions: list[torch.Tensor] = []
    all_advantages: list[torch.Tensor] = []
    all_value_targets: list[torch.Tensor] = []

    for episode in episodes:
        states = torch.tensor(episode.states, dtype=torch.long)
        actions = torch.tensor(episode.actions, dtype=torch.long)
        rewards = torch.tensor(episode.rewards, dtype=torch.float32)
        next_states = torch.tensor(episode.next_states, dtype=torch.long)
        dones = torch.tensor(episode.dones, dtype=torch.float32)

        # GAEは今回の更新で固定して使うターゲットなので勾配を持たせない。
        with torch.no_grad():
            values = value_model(states)
            next_values = value_model(next_states)
            advantages, value_targets = compute_gae(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

        all_states.append(states)
        all_actions.append(actions)
        all_advantages.append(advantages)
        all_value_targets.append(value_targets)

    states = torch.cat(all_states)
    actions = torch.cat(all_actions)
    advantages = torch.cat(all_advantages)
    value_targets = torch.cat(all_value_targets)

    # Advantageのスケールを揃えて方策更新を安定させる。
    # GAEで得られたAdvantageの平均を0、標準偏差を1に正規化して、方策勾配の大きさを安定させる処理
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    distribution = Categorical(logits=policy(states))
    selected_log_probs = distribution.log_prob(actions)
    predicted_values = value_model(states)

    # Actor: GAEが正の行動を強化し、負の行動を抑制する。
    policy_loss = -(selected_log_probs * advantages.detach()).mean()

    # Critic: GAEから作ったvalue targetへ状態価値を近づける。
    value_loss = torch.nn.functional.mse_loss(
        predicted_values,
        value_targets.detach(),
    )

    total_loss = policy_loss + value_coef * value_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }
```

PolicyとValueモデルを一つのoptimizerで更新する場合は，次のように両方のパラメータを渡す．

```python
policy = Policy()
value_model = ValueModel()
optimizer = torch.optim.Adam(
    list(policy.parameters()) + list(value_model.parameters()),
    lr=3e-2,
)

for iteration in range(300):
    episodes = [collect_episode(env, policy) for _ in range(32)]
    metrics = train_step_with_gae(
        policy=policy,
        value_model=value_model,
        optimizer=optimizer,
        episodes=episodes,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
    )
```

REINFORCE版との本質的な差分は次のとおりである．

```text
REINFORCE:
    実際の終端報酬からreward-to-goを計算
    G_t - baseline を advantage として使用

GAE:
    CriticでV(s_t)とV(s_{t+1})を計算
    各時刻のTD誤差δ_tを計算
    TD誤差を後ろからγλで累積
    ActorとCriticをそれぞれの損失で更新
```

実装時には，次を確認する必要がある．

- GAEをepisodeごとに計算し，異なるepisodeをつないでいないか
- 終端時に `next_value` が加算されないよう `1 - done` を掛けているか
- GAEとvalue targetを更新中の固定ターゲットとして扱っているか
- `policy_loss`では `advantages.detach()` を使用しているか
- `value_loss`では `value_targets.detach()` を使用しているか
- `gae_lambda=0` で1-step TDに近づくか
- `gae_lambda=1` でMonte Carlo型の推定に近づくか

この最小環境では時間切れも `done=True` として終端扱いしている．実用環境で `terminated` と `truncated` が区別される場合，真の終端である `terminated` ではbootstrapを止め，時間制限による `truncated` では通常 $V(s_{t+1})$ をbootstrapする．

<a id="trpo"></a>
## 9. TRPO
TRPOでは，ある方策 $\tilde{\pi}$と他の方策 $\pi$との収益期待値の差の公式
```math
\mathbb{E}_{\tau \sim \tilde\pi_\theta}[G_0] = \mathbb{E}_{\tau \sim \pi_\theta}[G_0] + \mathbb{E}_{\tau \sim \tilde\pi_\theta}\left[ \sum_{k=0}^{T} \gamma^k A_\pi(s_t,a_t) \right]
```
を元に，代理目的関数 (surrogate objective function) $L_{\mathrm{TRPO}}(\theta)$ として，
```math
L_{\mathrm{TRPO}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\rm old}}}\left[\frac{\pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_{\rm old}}(a_t\mid s_t)} A_{\pi_{\theta_{\rm old}}}(s_t,a_t)  \right]
```
を選ぶ．この代理目的関数の勾配は，初期状態では，元の収益期待値の勾配と等しく，1次精度近似となっている．具体的には，
```math
\begin{align}
\nabla_\theta L_{\mathrm{TRPO}}(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta_{\rm old}}}\left[  \frac{\nabla_\theta \pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_{\rm old}}(a_t\mid s_t)} A_{\pi_{\theta_{\rm old}}}(s_t,a_t)  \right] \\
& = \mathbb{E}_{\tau \sim \pi_{\theta_{\rm old}}}\left[\frac{\pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_{\rm old}}(a_t\mid s_t)} \nabla_\theta \log \pi_{\theta}(a\mid s) A_{\pi_{\theta_{\rm old}}}(s_t,a_t)  \right] \\
\end{align}
```
となる。ここで，probability ratio を
```math
\rho(\theta) = \frac{\pi_{\theta}(a\mid s)}{\pi_{\theta_{\rm old}}(a\mid s)}
```
とすると， $\rho(\theta_{\rm old})=1$ となる．従って，
```math
\begin{align}
\nabla_\theta L_{\mathrm{TRPO}}(\theta)\Big|_{\theta=\theta_{\rm old}}　
& = \mathbb{E}_{\tau \sim \pi_{\theta_{\rm old}}}\left[ \nabla_\theta \log \pi_{\theta}(a_t\mid s_t) A_{\pi_{\theta_{\rm old}}}(s_t,a_t)  \right]\Big|_{\theta=\theta_{\rm old}}　 \\
& = \nabla_\theta J(\theta)\Big|_{\theta=\theta_{\rm old}}　
\end{align}
```
となる．TRPO の代理目的関数は，古い方策のデータ上で「良かった行動の確率を上げ，悪かった行動の確率を下げる」ものである．実際の実装では，自動微分を考慮すると，方策損失は，
```math
L(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\rm old}}}\left[\frac{\pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_{\rm old}}(a_t\mid s_t)} A_{\pi_{\theta_{\rm old}}}(s_t,a_t)  \right]
```
となり，ここで， $\pi_{\theta}(a_t\mid s_t)$ は，ある特定の状態 $s_t$ 時にある特定の行動 $a_t$ が出る確率となる．
そのため，古い方策での確率と，新しい方策での確率の比が係数となる．
これにより，同じバッチを使いながら，現在の方策が古い方策からどれくらいズレたかを ratio で追跡可能となり，この行動の確率を古い方策に比べてどれくらい変えたかを管理可能となる．
古い方策で集めたデータを使って何回か更新しても，古い方策から離れすぎないようにしている．そのため，データ効率が通常の方策勾配法よりも良くなる．通常の方策勾配法では，すぐに，バッチ内のデータのみに最適化されてしまう．環境から軌跡を集めるのが高コストである強化学習では，よりデータ効率の良い手法が求められる．
さらに，TRPOでは，訓練における学習率だけではなく，古い方策と新しい方策の距離を KLダイバージェンス $D_{KL}\left( \pi_{\theta_{\rm old}} \| \pi_{\theta}\right)$ が一定以下になるように制約をかけながら，更新することも提案されている．しかしながら，このKL制約付き最適化問題を解くには，ヘッセ行列が必要となるため，計算コストが高いという問題がある．以降のPPOで損失関数に入れることで解決されていく．

<a id="ppo"></a>
## 10. PPO

このように，方策ベースの強化学習において，勾配方策をそのまま使うと，1回の更新で方策が大きく変わりすぎることがある．
そして，方策が急に変わると，集めた On-policy データがすぐ古くなり，学習が不安定になると点がある．TRPOの良いアイデアを引き継ぎながら，1次精度の最適化手法で提案されたのが Proximal Policy Optimization (PPO)である．
TRPOのアイデアである，更新前の方策 $\pi_{\theta_{\rm old}}$ と，更新中の方策 $\pi_\theta$ の確率比
```math
\rho_t(\theta)
= \frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\rm old}}(a_t \mid s_t)}
```
を用いて，更新を制限する．さらに，クリップ幅 $\epsilon$ を用いて， $\rho_t$ をクリップすることで，より更新する幅を制限している．PPO の clipped objective function
```math
L_{\mathrm{PPO}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta) A_{\pi_{\theta_{\rm old}}}(s_t,a_t),\,
\mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_{\pi_{\theta_{\rm old}}}(s_t,a_t)
\right)
- \beta D_{\mathrm{KL}}\left(\pi_{\theta_{\rm old}}(\cdot \mid s_t)\ \|\ \pi_\theta (\cdot \mid s_t)\right)
\right]
```
と書ける．PPO の直感は次の通りである．
- advantage が正の行動は，確率を上げたい
- advantage が負の行動は，確率を下げたい
- ただし，一度に大きく変えすぎると壊れるので，確率比を clip する
- 良くなりすぎる方向は打ち切る  （クリップすると $\theta$ 依存がないので勾配0）
- 悪くなる方向はそのまま罰する　（クリップしても $r_t(\theta)$ が悪すぎなら反映）
また，TRPOからの改善として，KLダイバージェンスをロスに入れる手法もここで提案された．
PPOは基本的に，actor-critic で用いられ，この場合，状態価値関数 $V_\pi$ がcritic となる．
PPOにおけるadvantage の推定には，Generalized Advantage Estimation (GAE)が用いられる．
- Entropy正則化を行うことで、方策が早い段階で特定の行動だけに偏ることを防ぎ、探索を維持することが可能である．

<a id="ppo-code"></a>
### コードセクション: LineWorldをPPOで学習する

ここでは，前節までに実装した `LineWorld`，`Policy`，`ValueModel`，`Episode`，`collect_episode()`，`compute_gae()` を使ってPPOへ拡張する．REINFORCEとの重要な違いは，trajectory収集時の対数確率を `old_log_probs` として保存し，同じtrajectoryを複数回更新することである．更新中の方策で再計算した `current_log_probs` との比を取ることで，更新前の方策からどれだけ変化したかを測定する．

まず，PPO更新に必要なテンソルをまとめる．

```python
@dataclass
class PPOBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    value_targets: torch.Tensor
```

収集したepisodeから，old log probability と GAE を計算する．この関数を呼んでからPPO更新が終わるまで，`old_log_probs`を再計算してはいけない．

```python
def build_ppo_batch(
    policy: Policy,
    value_model: ValueModel,
    episodes: list[Episode],
    gamma: float,
    gae_lambda: float,
) -> PPOBatch:
    all_states: list[torch.Tensor] = []
    all_actions: list[torch.Tensor] = []
    all_old_log_probs: list[torch.Tensor] = []
    all_advantages: list[torch.Tensor] = []
    all_value_targets: list[torch.Tensor] = []

    # π_old と V_oldから作る値は、PPO更新中はすべて固定する。
    with torch.no_grad():
        for episode in episodes:
            states = torch.tensor(episode.states, dtype=torch.long)
            actions = torch.tensor(episode.actions, dtype=torch.long)
            rewards = torch.tensor(episode.rewards, dtype=torch.float32)
            next_states = torch.tensor(episode.next_states, dtype=torch.long)
            dones = torch.tensor(episode.dones, dtype=torch.float32)

            old_distribution = Categorical(logits=policy(states))
            old_log_probs = old_distribution.log_prob(actions)

            values = value_model(states)
            next_values = value_model(next_states)
            advantages, value_targets = compute_gae(
                rewards=rewards,
                values=values,
                next_values=next_values,
                dones=dones,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )

            all_states.append(states)
            all_actions.append(actions)
            all_old_log_probs.append(old_log_probs)
            all_advantages.append(advantages)
            all_value_targets.append(value_targets)

    states = torch.cat(all_states)
    actions = torch.cat(all_actions)
    old_log_probs = torch.cat(all_old_log_probs)
    advantages = torch.cat(all_advantages)
    value_targets = torch.cat(all_value_targets)

    # GAEで得られたAdvantageの平均を0、標準偏差を1に正規化して、方策勾配の大きさを安定させる処理
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    return PPOBatch(
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        value_targets=value_targets,
    )
```

次に，PPOのclipped objectiveを実装する．

```python
def ppo_update(
    policy: Policy,
    value_model: ValueModel,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    clip_eps: float,
    update_epochs: int,
    minibatch_size: int,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> dict[str, float]:
    num_steps = len(batch.states)
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_clip_fraction = 0.0

    # 同じ pi_old のデータを複数epoch使う。
    for _ in range(update_epochs):
        permutation = torch.randperm(num_steps)

        for offset in range(0, num_steps, minibatch_size):
            indices = permutation[offset : offset + minibatch_size]

            states = batch.states[indices]
            actions = batch.actions[indices]
            old_log_probs = batch.old_log_probs[indices]
            advantages = batch.advantages[indices]
            value_targets = batch.value_targets[indices]

            current_distribution = Categorical(logits=policy(states))
            current_log_probs = current_distribution.log_prob(actions)

            # ρ_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
            # 確率の除算ではなく、log probabilityの差をexpする。
            ratios = torch.exp(current_log_probs - old_log_probs)

            unclipped_objective = ratios * advantages
            clipped_objective = torch.clamp(
                ratios,
                1.0 - clip_eps,
                1.0 + clip_eps,
            ) * advantages

            # PPOは目的関数を最大化するため、損失では負号を付ける。
            policy_loss = -torch.min(
                unclipped_objective,
                clipped_objective,
            ).mean()

            # 価値モデルの誤差最適化
            predicted_values = value_model(states)
            value_loss = torch.nn.functional.mse_loss(
                predicted_values,
                value_targets,
            )

            # 方策が早い段階で特定の行動だけに偏ることを防ぎ、探索を維持するために
            # Entropyを最大化したいので、最小化するlossでは負号を付ける。
            entropy = current_distribution.entropy().mean()
            total_loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            with torch.no_grad():
                clip_fraction = (
                    torch.abs(ratios - 1.0) > clip_eps
                ).float().mean()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_clip_fraction = float(clip_fraction.item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "clip_fraction": last_clip_fraction,
    }
```

PPO全体の学習ループは次のようになる．

```python
def train_ppo() -> None:
    env = LineWorld()
    policy = Policy()
    value_model = ValueModel()
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_model.parameters()),
        lr=3e-3,
    )

    for iteration in range(300):
        # 1. 現在方策π_oldでon-policyデータを収集する。
        episodes = [collect_episode(env, policy) for _ in range(32)]

        # 2. old log probabilityとGAEを一度だけ計算して固定する。
        batch = build_ppo_batch(
            policy=policy,
            value_model=value_model,
            episodes=episodes,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # 3. 同じデータを使って方策を複数epoch更新する。
        metrics = ppo_update(
            policy=policy,
            value_model=value_model,
            optimizer=optimizer,
            batch=batch,
            clip_eps=0.2,
            update_epochs=4,
            minibatch_size=64,
        )

        if iteration % 50 == 0:
            mean_return = (
                sum(sum(ep.rewards) for ep in episodes) / len(episodes)
            )
            left_prob, right_prob = action_probabilities(policy)
            print(
                f"iteration={iteration:3d} "
                f"return={mean_return:+.3f} "
                f"policy_loss={metrics['policy_loss']:+.4f} "
                f"clip_fraction={metrics['clip_fraction']:.3f} "
                f"P(right)={right_prob:.3f}"
            )
```

数式とコードの対応は次のとおりである．

| 数式 | コード |
| --- | --- |
| $\pi_{\theta_{\rm old}}$ | `build_ppo_batch()`を呼んだ時点の`policy` |
| $\log\pi_{\theta_{\rm old}}(a_t\mid s_t)$ | `batch.old_log_probs` |
| $\pi_\theta$ | `ppo_update()`内で更新される`policy` |
| $\log\pi_\theta(a_t\mid s_t)$ | `current_log_probs` |
| $\rho_t(\theta)$ | `torch.exp(current_log_probs - old_log_probs)` |
| $\hat A_t^{\mathrm{GAE}}$ | `batch.advantages` |
| $\rho_t\hat A_t$ | `unclipped_objective` |
| $\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t$ | `clipped_objective` |
| $-\mathbb E[\min(\cdot,\cdot)]$ | `policy_loss` |
| $\left(V_\phi(s_t)-\hat V_t\right)^2$ | `value_loss` |

最初のPPO epochでは，更新前なので

```math
\log\pi_\theta(a_t\mid s_t)
=\log\pi_{\theta_{\rm old}}(a_t\mid s_t)
```

であり，`ratios`は1となる．1回目の`optimizer.step()`以降はcurrent policyだけが変化し，`old_log_probs`は固定されているため，ratioが1から離れ始める．この状態で同じデータを複数回使うことにより，clippingが意味を持つ．

実装時には，次を確認する必要がある．

- trajectory収集後，更新前に `old_log_probs` を保存しているか
- PPOの各epochで `old_log_probs` を再計算していないか
- 確率比を `exp(current_log_prob - old_log_prob)` で計算しているか
- Actorの目的関数に `torch.min()` を使っているか
- `advantages`と`value_targets`を固定ターゲットとして扱っているか
- 複数epoch更新後は，新しい方策でtrajectoryを集め直しているか

PPO schematic view
 ```mermaid
%%{init: {"flowchart": {"nodeSpacing": 12, "rankSpacing": 10, "htmlLabels": true}, "themeVariables": {"fontSize": "12px"}}}%%
flowchart LR
    Q["s"] --> P["Policy"]
    P --> O["$a$"]

    O --> M["Reference & Reward"]
    O --> V["Value"]

    M --> R["r & KL"]
    V --> VV["$V$"]

    R --> G["GAE"]
    VV --> G
    G --> A["A"]
    A --> P

    classDef trained fill:#fff0bd,stroke:#34495e;
    classDef frozen fill:#dcecff,stroke:#34495e;

    class P,V trained;
    class M frozen;

 ```


<a id="grpo"></a>
## 11. GRPO

GRPO はdeepseek における強化学習で提案された手法である．GRPO は PPO に近い方策更新を行うが，Critic による価値推定の代わりに，グループ内の相対評価で advantage を作る．同じ入力や同じ状態から $G$ 個の出力をサンプルする．
```math
a_1, a_2, \dots, a_G
\sim \pi_{\theta_{\rm old}}(\cdot \mid s)
```
それぞれ行動 $a_i$ に報酬 $r_i$ を付け，グループ内の平均と標準偏差より，相対advantage 
```math
\hat{A}_i=
\frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}
{\mathrm{std}(r_1, \dots, r_G) + \delta}
```
を計算する．ここで $\delta$はゼロ割を防ぐためである．
GRPO の目的関数は PPO と同じく clipped objective を使う．
```math
L_{\mathrm{GRPO}}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min
\left(\rho_{i,t}(\theta) \hat{A}_i,\,\mathrm{clip}(\rho_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right)
- \beta D_{\mathrm{KL}}\left(\pi_\theta \| \pi_{\mathrm{ref}}\right)
\right]
```
ここで $\pi_{\mathrm{ref}}$ は固定された reference model であり，KLダイバージェンス $D_{\mathrm{KL}}$ は現在の方策が reference からどれくらい離れたかを表す．
パラメータ $\beta$ は KL正則化の強さである．
GRPO の中心は，Critic を使わずに group 内比較で advantage を作る点にある．同じ状態から複数の候補を出し，その中で報酬が高いものを押し上げ，低いものを押し下げる．
KLダイバージェンスに関しては，サンプル近似を行う際に，負となる可能性があるために，少し改良を加えたもの
```math
D_{\mathrm{KL}}\left(\pi_\theta \| \pi_{\mathrm{ref}}\right)=\dfrac{\pi_{\mathrm{ref}}(a\mid s)}{\pi_\theta(a\mid s)} -\log \dfrac{\pi_{\mathrm{ref}}(a\mid s)}{\pi_\theta(a\mid s)} -1
```
を使用する．これは，KLの不偏推定量でかつ，非負となる．

GRPO schematic view
```mermaid
%%{init: {"flowchart": {"nodeSpacing": 12, "rankSpacing": 10, "htmlLabels": true}, "themeVariables": {"fontSize": "12px"}}}%%
flowchart LR
    Q["s"] --> P["Policy"]
    P["Policy"] --> O["$$a_1,\ldots,a_G$$"]

    O --> M["Reference & Reward"]
    M --> R["$$r_1,\ldots,r_G$$ & KL"]

    R --> A["$$A_1,\ldots,A_G$$"]
    A --> P

    classDef trained fill:#fff0bd,stroke:#34495e;
    classDef frozen fill:#dcecff,stroke:#34495e;

    class P trained;
    class M frozen;
 ```

<a id="grpo-code"></a>
### コードセクション: Renjuのtrajectory-group GRPO

ここでは，`src/renju_transformer/grpo.py` に実装されているtrajectory-group GRPOを，実際の処理順に沿って説明する．

この実装では，同じ開始盤面かつ同じpolicy担当色から複数の対局を生成する．
各trajectoryの報酬は，policyが打った手の局所報酬の合計と終局勝敗bonusから作る．
同じgroup内でtrajectory報酬を標準化し，そのtrajectoryでpolicyが打った全着手へ同じadvantageを割り当てる．

<a id="grpo-trajectory-values"></a>
#### 1. trajectoryに保存する値

実装で使用するデータクラスは次の形である．

```python
@dataclass(slots=True)
class TrajectoryStep:
    board: list[int]
    action: int
    actor: int
    old_log_prob: float
    local_reward: float
    group_index: int
    chosen: bool
    learn: bool


@dataclass(slots=True)
class Trajectory:
    steps: list[TrajectoryStep]
    winner: int | None
    final_board: list[int]
    total_reward: float
    actual_plies: int
    policy_player: int | None = None
```

各フィールドの意味は次のとおりである．

| フィールド | 内容 |
| --- | --- |
| `board` | 着手前の盤面 $s_t$ |
| `action` | 候補または実際に選ばれた着手 $a_t$ |
| `actor` | その手を打つ黒または白 |
| `old_log_prob` | trajectory生成時の行動対数確率 |
| `local_reward` | TSS・即勝ち・即負け・形などから得た局所報酬 |
| `group_index` | trajectory内の手数index |
| `chosen` | 候補のうち実際に盤面へ反映した手か |
| `learn` | policyの更新対象となる手か |
| `total_reward` | trajectory単位で比較する合計報酬 |
| `policy_player` | このtrajectoryでpolicyが担当する色 |

trajectory-groupでは1局面につき1手だけサンプリングするため，保存されるstepの`chosen`は常に`True`になる．
`TrajectoryStep`はstep-group objectiveとも共用しているため，候補比較用のフィールドも持っている．

Referenceのlog probabilityはtrajectoryには保存しない．固定reference modelから，GRPO lossを計算するたびに同じ状態・同じ行動について再計算する．
一方，新旧policyの比率に必要な`old_log_prob`はtrajectory生成時に保存し，更新中は固定する．

<a id="grpo-policy-side"></a>
#### 2. policy担当色と対戦相手を決める

`resolve_policy_players()`は，設定からtrajectory groupを作るpolicy担当色を返す．

```python
def resolve_policy_players(cfg, rollout_index=None):
    learning_player = str(
        cfg.grpo.step_group.get("learning_player", "both")
    )

    if learning_player == "black":
        return [BLACK]
    if learning_player == "white":
        return [WHITE]
    if learning_player == "both" and rollout_index is None:
        return [BLACK, WHITE]
    ...
```

デフォルトの`learning_player=both`では，同じ開始盤面に対して次の2 groupを別々に生成する．

```text
group 1: policyが黒，referenceが白
group 2: policyが白，referenceが黒
```

各groupはそれぞれ`grpo.trajectory_group.group_size`本のtrajectoryを持つ．黒policyのtrajectoryと白policyのtrajectoryを同じgroupとして正規化することはない．

`grpo.step_group.opponent=reference`では，policy担当色以外の手を固定reference modelが打つ．`opponent=self`では両方の色をpolicy modelが打つが，`learn=True`になるのは指定したpolicy担当色の手だけである．

<a id="grpo-sample-action"></a>
#### 3. 合法手から1手をサンプリングする

着手のサンプリングでは，非合法手のlogitを`-inf`へ置き換えてからsoftmaxを計算する．

```python
@torch.no_grad()
def sample_actions_from_logits(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_logits = logits.masked_fill(
        ~legal_masks,
        float("-inf"),
    )
    probabilities = torch.softmax(masked_logits, dim=-1)

    sampled_actions = []
    for row_index in range(probabilities.size(0)):
        legal_count = int(legal_masks[row_index].sum().item())
        replacement = legal_count < sample_count
        sampled_actions.append(
            torch.multinomial(
                probabilities[row_index],
                num_samples=sample_count,
                replacement=replacement,
            )
        )

    actions = torch.stack(sampled_actions, dim=0)
    log_probs = torch.log(
        probabilities.gather(dim=-1, index=actions).clamp_min(1e-12)
    )
    return actions, log_probs
```

trajectory-groupからは`sample_count=1`で呼ばれる．したがって各手番でpolicyまたはreferenceから1手をサンプリングし，その手をそのまま盤面へ反映する．

<a id="grpo-generate-trajectory"></a>
#### 4. 1本のtrajectoryを生成する

1本の対局生成は`rollout_policy_episode()`が担当する．
関数全体に`@torch.no_grad()`が付いているため，trajectory収集中に勾配は作られない．呼び出し側ではpolicyとreferenceを`eval()`へ切り替えてからrolloutする．

処理の中心は次のとおりである．

```python
@torch.no_grad()
def rollout_policy_episode(
    start_board,
    policy_player,
    policy_model,
    reference_model,
    tokenizer,
    reward_evaluator,
    cfg,
    device,
    *,
    sample_count,
    default_opponent,
    reward_mode,
):
    board = start_board.copy()
    steps = []
    winner = current_winner(board)
    actual_plies = 0
    max_plies = int(cfg.grpo.step_group.max_plies)
    opponent = str(
        cfg.grpo.step_group.get("opponent", default_opponent)
    )

    while (
        winner is None
        and not board_is_full(board)
        and actual_plies < max_plies
    ):
        actor = infer_player(board)
        learn = policy_player is None or actor == policy_player

        acting_model = (
            policy_model
            if learn or opponent == "self"
            else reference_model
        )

        candidate_actions, old_log_probs = sample_policy_actions(
            acting_model,
            tokenizer,
            board,
            sample_count=sample_count,
            device=device,
        )

        local_rewards = reward_evaluator.evaluate_batch(
            [board],
            [candidate_actions],
        )[0]

        chosen_index = 0
        chosen_action = candidate_actions[chosen_index]

        steps.append(
            TrajectoryStep(
                board=board.copy(),
                action=chosen_action,
                actor=actor,
                old_log_prob=old_log_probs[chosen_index],
                local_reward=(
                    float(local_rewards[chosen_index])
                    * float(cfg.grpo.step_group.tss_weight)
                ),
                group_index=actual_plies,
                chosen=True,
                learn=learn,
            )
        )

        board = board_with_move(board, chosen_action, actor)
        winner = winner_after_move(board, chosen_action, actor)
        actual_plies += 1
```

`board.copy()`によって着手前の盤面を保存してから，`board_with_move()`で次状態へ進める．勝敗が決まる，盤面が埋まる，合法手がなくなる，または`max_plies`へ到達するとrolloutを終了する．

各手の`local_reward`は`GrpoRewardEvaluator`が計算する．設定に応じて，合法性，即勝ち・即負け，相手の即勝ちのブロック，四や開三，TSSによる強制勝ち・強制負けなどが含まれる．

<a id="grpo-trajectory-reward"></a>
#### 5. trajectory報酬を作る

trajectory-groupの比較に使う報酬は，policy担当色が実際に打った手の局所報酬だけを合計し，最後に終局結果を1回加えたものである．

```python
if reward_mode == "trajectory_group":
    total_reward = sum(
        step.local_reward
        for step in steps
        if step.learn
    )
    total_reward += final_reward_for_actor(
        winner,
        policy_player,
        cfg,
    )
```

数式で表すと，

```math
R(\tau_i)
=
\sum_{t:\,\mathrm{learn}_{i,t}=\mathrm{True}}
r_{i,t}^{\mathrm{local}}
+
r_i^{\mathrm{terminal}}
```

となる．終局報酬は開始手番視点ではなく，そのtrajectoryの`policy_player`視点で決まる．

```python
def final_reward_for_actor(winner, actor, cfg):
    if winner is None:
        return float(cfg.grpo.step_group.draw_reward)
    if winner == actor:
        return float(cfg.grpo.step_group.final_result_weight)
    return -float(cfg.grpo.step_group.final_result_weight)
```

したがってデフォルト設定では，

```text
policy担当色が勝利: +1
policy担当色が敗北: -1
引き分け・未決着:   0
```

が局所報酬の合計へ加わる．reference担当色が打った手の局所報酬はstepには保存されるが，trajectoryの`total_reward`には含めない．

<a id="grpo-generate-group"></a>
#### 6. 同じ開始盤面からG本を生成する

学習ループでは，policyとreferenceを評価modeにしてから，同じ開始盤面・同じpolicy担当色で`rollout_policy_episode()`をG回呼び出す．

```python
policy_model.eval()
reference_model.eval()

for policy_player in policy_players:
    trajectories = [
        rollout_policy_episode(
            prompt,
            policy_player,
            policy_model,
            reference_model,
            tokenizer,
            reward_evaluator,
            cfg,
            device,
            sample_count=1,
            default_opponent="reference",
            reward_mode="trajectory_group",
        )
        for _ in range(trajectory_group_size)
    ]
```

各rolloutの先頭で`start_board.copy()`を行うため，G本は同じ内容の開始盤面から独立して分岐する．現在の実装はG本を手数ごとのGPU batchとして同時進行させるのではなく，1本ずつ直列に生成する．ただし，G本すべてを生成し終わるまでpolicyのoptimizer更新は行わない．

1回のoptimizer更新では，`grpo.step_group.prompts_per_step`個の開始盤面を処理する．`learning_player=both`なら，各開始盤面について黒policy groupと白policy groupの両方を生成する．

<a id="grpo-group-advantage"></a>
#### 7. group-relative advantageを計算する

同じ開始盤面かつ同じpolicy担当色のG本について，`total_reward`を標準化する．

```python
def normalize_group_advantages(
    rewards: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    )
    return (rewards - mean) / (std + epsilon)
```

trajectory $i$のadvantageは，

```math
\hat A_i
=
\frac{
R(\tau_i)-\mu_R
}{
\sigma_R+\delta
}
```

である．G本の報酬がすべて同じ場合，標準偏差は0になり，全trajectoryのadvantageも0になる．このgroupからpolicy gradientは得られないが，KL項とentropy項は計算される．

<a id="grpo-assign-advantage"></a>
#### 8. policyの着手へadvantageを割り当てる

`flatten_trajectories()`は各trajectoryのadvantageを，そのtrajectoryで`learn=True`となっている全stepへ同じ値で割り当てる．

```python
for trajectory, advantage in zip(
    learnable_trajectories,
    advantages.tolist(),
    strict=True,
):
    for step in trajectory.steps:
        if not step.learn:
            continue

        boards.append(step.board)
        actions.append(step.action)
        old_log_probs.append(step.old_log_prob)
        all_advantages.append(float(advantage))
```

referenceが打ったstepは盤面を進めるためだけに使い，GRPO更新用のflat batchには入れない．そのため，相手番のadvantageを符号反転する処理はない．黒policy groupでは黒の着手だけ，白policy groupでは白の着手だけを学習する．

また，trajectory長による重みの補正は行っていない．全学習対象stepを後段のlossで単純平均するため，policy着手数の多い長いtrajectoryほど，optimizer更新に含まれる項数が多くなる．

<a id="grpo-loss"></a>
#### 9. GRPO lossを計算する

更新時には，全stepの盤面をまとめてmodelへ入力する．current policyと固定referenceのlog probabilityは，同じ合法手maskと変換していない生logitsから計算する．

```python
logits = policy_model(input_ids)
log_probs = masked_log_probs(logits, legal_masks)
selected_log_probs = log_probs.gather(
    dim=-1,
    index=actions.reshape(-1, 1),
).squeeze(-1)

with torch.no_grad():
    reference_logits = reference_model(input_ids)
    reference_log_probs = masked_log_probs(
        reference_logits,
        legal_masks,
    )
    selected_reference_log_probs = reference_log_probs.gather(
        dim=-1,
        index=actions.reshape(-1, 1),
    ).squeeze(-1)
```

新旧policyの確率比とclipped objectiveは，

```math
\rho_t(\theta)
=
\exp\left(
\log\pi_\theta(a_t\mid s_t)
-
\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
\right)
```

```python
ratio = torch.exp(selected_log_probs - old_log_probs)
clipped_ratio = ratio.clamp(
    1.0 - cfg.grpo.clip_epsilon,
    1.0 + cfg.grpo.clip_epsilon,
)

policy_loss = -torch.minimum(
    ratio * advantages,
    clipped_ratio * advantages,
).mean()
```

で計算する．

ReferenceとのKL近似には，

```math
x-\log x-1,
\qquad
x=\frac{\pi_{\mathrm{ref}}(a\mid s)}
{\pi_\theta(a\mid s)}
```

を使う．

```python
log_ratio_to_ref = (
    selected_reference_log_probs
    - selected_log_probs
)
kl = (
    torch.exp(log_ratio_to_ref)
    - log_ratio_to_ref
    - 1.0
).mean()
```

さらに合法手分布のentropyを計算し，最終lossを，

```math
L
=
L_{\mathrm{policy}}
+
\beta D_{\mathrm{KL}}
-
c_{\mathrm{entropy}}H(\pi_\theta)
```

として合成する．

```python
loss = (
    policy_loss
    + float(cfg.grpo.kl_beta) * kl
    - float(cfg.grpo.entropy_coef) * entropy
)
```

<a id="grpo-multiple-epochs"></a>
#### 10. 同じ収集データを複数epoch更新する

rollout後，policyを`train()`へ戻し，複数prompt・複数groupから集めた全学習対象stepを1つのflat batchへまとめる．現在の実装はminibatchへ分割せず，flat batch全体を各update epochで使用する．

```python
policy_model.train()

for _ in range(int(cfg.grpo.update_epochs)):
    optimizer.zero_grad(set_to_none=True)

    loss, update_metrics = compute_flat_grpo_loss(
        policy_model,
        reference_model,
        input_ids,
        legal_masks,
        actions,
        old_log_probs,
        advantages,
        cfg,
    )

    loss.backward()

    if cfg.train.gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(),
            cfg.train.gradient_clip_norm,
        )

    optimizer.step()
```

`old_log_probs`とadvantageは全update epochで固定する．reference modelは同じ教師ありcheckpointから初期化し，`requires_grad=False`としてoptimizer更新の対象外にする．

モデルのdropout既定値は0であり，GRPOモデルをcheckpointから再構築するときも現在の`cfg.model.dropout`を使う．デフォルト設定では，rollout時の`eval()`と更新時の`train()`でdropoutによる確率変動は生じない．そのため，最初のoptimizer更新前は，収集時と更新時のpolicyが同じなら確率比は1になる．

<a id="grpo-overview"></a>
#### 11. 処理全体の対応

| GRPOの処理 | 実装 |
| --- | --- |
| 開始盤面を作る | `build_start_prompts()` |
| policy担当色を決める | `resolve_policy_players()` |
| 1本の対局を生成する | `rollout_policy_episode()` |
| 同一起点・同一policy色からG本作る | `train_trajectory_group_grpo_loop()` |
| trajectory報酬を標準化する | `normalize_group_advantages()` |
| policy着手へadvantageを配る | `flatten_trajectories()` |
| PPO型clip・KL・entropyを計算する | `compute_flat_grpo_loss()` |
| referenceを固定する | `load_policy_and_reference()` |

処理の流れをまとめると次のようになる．

```text
開始盤面を選ぶ
  ├─ policy=黒 のtrajectoryをG本直列生成
  │    ├─ 黒はpolicyからサンプル
  │    ├─ 白はreferenceからサンプル
  │    └─ 黒の局所報酬合計 + 黒視点の終局報酬
  └─ policy=白 のtrajectoryをG本直列生成
       ├─ 白はpolicyからサンプル
       ├─ 黒はreferenceからサンプル
       └─ 白の局所報酬合計 + 白視点の終局報酬

各group内でtrajectory報酬を標準化
  ↓
各trajectoryのpolicy着手だけをflat batchへ追加
  ↓
current policy / old policyの比率を計算
  ↓
clipped policy loss + KL - entropyでpolicyを更新
```

<a id="summary"></a>
## 14. まとめ

強化学習では，将来得られる報酬の合計を最大化するように方策を学習する．価値ベースは価値関数を学習し，方策ベースは行動を選ぶ確率分布そのものを学習する．

Policy Gradient は，良い行動の確率を上げ，悪い行動の確率を下げる基本的な方策最適化である．PPO は，確率比を clip することで，方策が一度に変わりすぎることを防ぐ．GRPO は，PPO に近い更新を使いつつ，Critic の代わりに グループ内比較で advantage を作る．

RenjuTransformer では，同じ局面から複数候補手を出し，TSS やルール報酬で比較できるため，GRPO と相性が良い．勝敗だけでなく，TSS による中間報酬を使うことで，疎な報酬を緩和しながら policy を改善できる
