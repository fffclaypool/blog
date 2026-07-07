---
layout: post
title: "Transformer アーキテクチャの数理 — Attention・FFN・マスクを式で読む"
description: "Transformer を ℝ^{n×d} → ℝ^{n×d} の写像として書き下し、√d_k で割る理由、O(n²d) の出どころ、Encoder と Decoder を分けるマスクの正体を数式から導く。"
date: 2026-07-07 12:00:00
lang: ja
ref: transformer-architecture-math
permalink: /ja/transformer-architecture-math/
translation: /en/transformer-architecture-math/
translation_label: "English"
---

Transformer のブロック図は誰もが見たことがある。しかし各矢印が「どんな行列演算か」「なぜその形か」まで言えるだろうか。本記事では、埋め込みから Attention、FFN、LayerNorm、マスクまでを行列の言葉で一つずつ書き下し、**$$\sqrt{d_k}$$ で割る理由や計算量 $$O(n^2 d)$$ の出どころといった「式を読めばわかる事実」**を拾い上げていく。

## 全体像 — Transformer は何を計算する関数か

数理的に見た Transformer ブロックは、驚くほど単純な型を持つ。**長さ $$n$$ のトークン列を $$d$$ 次元ベクトルの列として受け取り、同じ形の列を返す写像**である。

$$
X \in \mathbb{R}^{n \times d} \;\longmapsto\; \mathrm{TransformerBlock}(X) \in \mathbb{R}^{n \times d}
$$

行が1トークン、列が特徴次元。この形が保存されるからこそブロックを何層でも積める。1ブロックの中身は、2つの部品の交互適用にすぎない。

- **Self-Attention**：トークン**間**で情報を混ぜる。行方向（系列方向）の演算。
- **FFN（Feed-Forward Network）**：各トークン**内**で特徴を変換する。列方向（特徴方向）の演算で、全トークンに同じ MLP を独立適用する。

![Transformerブロックの構造。系列方向に混ぜるAttentionと、特徴方向に変換するFFNを、残差接続とLayerNormを挟んで交互に適用する]({{ '/assets/img/transformer-block.svg' | relative_url }})

*1ブロックの分業。系列方向の情報伝達は Attention が独占し、FFN は位置ごとに閉じている。この対比が計算量の議論にもそのまま効いてくる。*

それぞれに残差接続と LayerNorm が付く。つまり Transformer とは「**系列方向に混ぜる層と特徴方向に変換する層を交互に $$L$$ 回積んだもの**」であり、系列の情報が他のトークンへ渡る経路は Attention だけである。この分業構造を頭に入れておくと、以降の式がすべて「どちらの方向の演算か」で整理できる。

## 入力の数理 — 埋め込みと位置符号化

トークン ID の列を行列 $$X$$ に変える最初のステップが埋め込みである。語彙サイズを $$\lvert V \rvert$$ とすると、埋め込み行列 $$E \in \mathbb{R}^{\lvert V \rvert \times d}$$ の「行の引き当て」がその実体だ。数式上は one-hot ベクトルとの積として書ける。

$$
x_i = \mathrm{onehot}(t_i)^\top E \in \mathbb{R}^d
$$

ここで問題がひとつある。次節で見る Attention は**置換等変（permutation-equivariant）**、つまり入力の行を並べ替えると出力も同じ並べ替えを受けるだけで、**語順の情報を一切持たない**。「犬が猫を追う」と「猫が犬を追う」が区別できないのだ。そこで位置情報をベクトルとして加算する。原論文の sinusoidal 符号化は次で定義される。

$$
\mathrm{PE}(\mathrm{pos},\, 2i) = \sin\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right), \qquad
\mathrm{PE}(\mathrm{pos},\, 2i+1) = \cos\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right)
$$

次元ペアごとに波長の異なる正弦波で、入力は $$X + \mathrm{PE}$$ として与えられる。この設計の数理的な妙は、**任意の固定オフセット $$k$$ に対して $$\mathrm{PE}(\mathrm{pos}+k)$$ が $$\mathrm{PE}(\mathrm{pos})$$ の線形変換で書ける**ことにある。回転行列の加法定理がそのまま効いており、モデルが「相対位置」を線形演算で扱える余地を作っている。この発想を推し進め、位置情報を加算ではなく Q と K の**回転**として埋め込むのが、現代の LLM で主流の RoPE（Rotary Position Embedding）である。内積 $$q \cdot k$$ が回転角の差、すなわち相対位置だけに依存するようになる。

## Attention — 加重平均としての読み解き

Self-Attention の計算は3行で書ける。入力 $$X$$ から3つの行列を線形変換で作り、

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

（$$W_Q, W_K \in \mathbb{R}^{d \times d_k}$$、$$W_V \in \mathbb{R}^{d \times d_v}$$。）スコア行列を計算して行ごとにソフトマックスをかけ、$$V$$ の加重和を取る。

$$
A = \mathrm{softmax}_{\mathrm{row}}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}, \qquad
\mathrm{Output} = A V \in \mathbb{R}^{n \times d_v}
$$

$$A$$ の $$(i, j)$$ 成分は「トークン $$i$$ がトークン $$j$$ をどれだけ参照するか」で、各行は非負で合計1になる。

![Attentionの行列演算。QとK転置の積でn×nの類似度行列を作り、行ソフトマックスで確率化してVを混ぜる]({{ '/assets/img/attention-matmul.svg' | relative_url }})

*行列演算としての Attention。中央の $$n \times n$$ 行列が Attention 重みで、第 $$i$$ 行がトークン $$i$$ の「どこを見るか」の確率分布になっている。系列長の2乗の行列がここで生まれる。*

この式の最良の読み方は、1トークン分を取り出して**加重平均**として見ることだ。トークン $$i$$ の出力は、

$$
\mathrm{output}_i = \sum_{j} a_{ij}\, v_j, \qquad
a_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d_k})}
$$

つまり Attention とは、**内積の大きさで重みを決めた、Value ベクトルの凸結合**である。Query は「いま何を探しているか」、Key は「自分が何の情報を持っているかの見出し」、Value は「実際に渡す中身」——データベースのキー・バリュー検索を、ハードな一致ではなくソフトな（微分可能な）内積類似度で行っている、という比喩がそのまま式に対応している。

出力が常に凸結合（重み非負・合計1）であることには含意がある。各トークンの新しい表現は「他のトークンの Value が張る凸包の内部」から選ばれる。Attention 自体は新しい特徴を作らず**混ぜるだけ**で、非線形な特徴生成は FFN の仕事——という分業がここでも確認できる。

## なぜ √d_k で割るのか — 分散の計算

「scaled」の由来である $$\sqrt{d_k}$$ は、飾りではなく短い計算から必然的に出てくる。$$q$$ と $$k$$ の各成分が独立に平均0・分散1で分布すると仮定して、内積の分散を計算してみる。

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i, \qquad
\mathbb{E}[q \cdot k] = 0, \qquad
\mathrm{Var}[q \cdot k] = \sum_{i=1}^{d_k} \mathrm{Var}[q_i k_i] = d_k
$$

独立な積 $$q_i k_i$$ の分散は $$1 \times 1 = 1$$。それが $$d_k$$ 個足されるので、内積の標準偏差は $$\sqrt{d_k}$$ のオーダーで次元とともに育つ。$$d_k = 64$$ なら $$\pm 8$$ 程度、$$d_k = 512$$ なら $$\pm 23$$ 程度のスコアが普通に出る。これをそのままソフトマックスに入れると何が起きるか。

- 指数関数がスコア差を増幅し、ソフトマックスが**ほぼ one-hot に飽和**する。
- 飽和領域ではソフトマックスの勾配（ヤコビアンの成分 $$a_i(\delta_{ij} - a_j)$$）がほぼ0になり、**学習信号が消える**。

$$\sqrt{d_k}$$ で割れば内積の分散は次元によらず1に正規化され、初期化直後のソフトマックスが適度になだらかな分布から学習を始められる。**$$\sqrt{d_k}$$ は「ソフトマックスを勾配の生きている領域に保つための温度」**だと言ってよい。

## Multi-Head Attention — 部分空間への分割

1つの Attention は1枚の $$n \times n$$ 重み地図しか持てない。「構文的な依存」と「照応関係」と「近傍の連続性」を**同時に**見たければ、地図が複数欲しい。そこで表現空間を $$h$$ 個の部分空間に射影し、それぞれで独立に Attention を走らせる。

$$
\mathrm{head}_i = \mathrm{Attention}(X W_Q^{(i)},\, X W_K^{(i)},\, X W_V^{(i)}), \qquad
\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\, W_O
$$

通常 $$d_k = d_v = d/h$$ と取る（例：$$d = 512$$、$$h = 8$$ → 各ヘッド64次元）。重要なのは、$$d$$ を $$h$$ 分割するため**パラメータ数と計算量は単一ヘッドとほぼ同じ**という点だ。タダで $$h$$ 枚の注意地図が手に入る設計になっている。各ヘッドは $$d/h$$ 次元という低ランクの「視点」しか持たないが、その代わり $$h$$ 通りの異なる関係性を並列に符号化できる。学習済みモデルの分析では、特定の構文関係や位置パターンに特化したヘッドが自然に分化することが繰り返し観察されている。

## FFN・残差接続・LayerNorm

### FFN — 位置ごとの2層 MLP

Attention で混ざった表現を、各トークン独立に非線形変換するのが FFN である。

$$
\mathrm{FFN}(x) = \varphi(x W_1 + b_1)\, W_2 + b_2
$$

$$W_1 \in \mathbb{R}^{d \times d_{\mathrm{ff}}}$$、$$W_2 \in \mathbb{R}^{d_{\mathrm{ff}} \times d}$$ で、中間次元は慣習的に $$d_{\mathrm{ff}} = 4d$$。$$\varphi$$ は ReLU / GELU、近年は GLU 系（SwiGLU など）が主流である。一度 $$4d$$ 次元へ広げてから戻すこの形は、後述の通り Transformer のパラメータの過半を占める。近年の解釈研究では、FFN は **key-value 型の連想記憶**——$$W_1$$ の行がパターン検出器、$$W_2$$ の行が対応する出力パターン——として振る舞うという見方が有力で、モデルの「知識」の多くがここに格納されていると考えられている（Geva et al., 2021）。

### 残差接続 — 恒等写像を基準にする

$$
x \leftarrow x + \mathrm{Sublayer}(x)
$$

残差接続の数理的な効用は2つある。第一に、逆伝播のヤコビアンが $$I + \partial \mathrm{Sublayer} / \partial x$$ の形になり、恒等成分 $$I$$ を通じて**勾配が減衰せずに深い層まで届く**。第二に、ネットワーク全体が「恒等写像＋各層の小さな更新の和」として書けるため、各層は表現をゼロから作り直すのではなく**差分を書き込む**だけでよい。この見方（residual stream への読み書き）は、現代の機構的解釈可能性研究の基本言語にもなっている（Elhage et al., 2021）。

### LayerNorm — 特徴方向の正規化

$$
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad
\mu = \frac{1}{d} \sum_{i} x_i, \qquad
\sigma^2 = \frac{1}{d} \sum_{i} (x_i - \mu)^2
$$

1トークンの $$d$$ 次元ベクトル内で平均・分散を取る（BatchNorm と違い、バッチにも系列長にも依存しない）。配置には2流派ある。原論文の **Post-LN**（$$\mathrm{LN}(x + \mathrm{Sublayer}(x))$$）は残差経路そのものを正規化してしまうため深いモデルで学習が不安定になりやすく、warmup が必須になる。現代の標準は **Pre-LN**（$$x + \mathrm{Sublayer}(\mathrm{LN}(x))$$）で、残差経路を素通しに保つことで深くても安定に学習できる（Xiong et al., 2020）。GPT-2 以降の LLM はほぼすべて Pre-LN 系（さらに平均項を省いた RMSNorm が主流）である。

## 因果マスクと Decoder

ここまでの式は、Encoder と Decoder で共通である。両者を分けるのは本質的にただ一点、**Attention 行列にどんなマスクをかけるか**だ。Decoder では「未来を見ない」制約を課すため、スコア行列の上三角部分をソフトマックスの**前に** $$-\infty$$ に置き換える。

$$
A = \mathrm{softmax}_{\mathrm{row}}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right), \qquad
M_{ij} =
\begin{cases}
0 & (j \le i) \\
-\infty & (j > i)
\end{cases}
$$

$$\exp(-\infty) = 0$$ なので、ソフトマックス後に未来位置の重みが厳密に0になる。0を「後から」掛けるのではなく、ソフトマックスの分母からも除外される点が重要だ。

![双方向Attention（Encoder）と因果マスク付きAttention（Decoder）の比較。Encoderは全トークンが全トークンを参照でき、Decoderは下三角のみ]({{ '/assets/img/causal-mask.svg' | relative_url }})

*マスクの違いが Encoder / Decoder の違いの核心。BERT は左、GPT は右の形の Attention を全層で使う。*

このマスクひとつで、モデルの性格が決まる。

- **Encoder（BERT型）**：マスクなしの双方向 Attention。全文を見た文脈表現が作れるため理解系タスクに強いが、そのままでは自己回帰生成に使えない。
- **Decoder（GPT型）**：因果マスク付き。位置 $$i$$ の表現が未来に依存しないため、学習時は**全位置の次トークン予測を1回の行列演算で並列に**計算でき、推論時は KV キャッシュ（過去の $$K, V$$ を保存し、新トークン分だけ計算する）で逐次生成が効率化できる。
- **Encoder–Decoder（原論文 / T5 型）**：Decoder 側に第3の Attention として **Cross-Attention** が入る。$$Q$$ は Decoder の表現から、$$K, V$$ は Encoder の出力から作る——つまり「生成中の系列が、入力系列を検索する」演算である。

## 計算量とパラメータ数の勘定

式が揃ったので、コストを数えられる。1ブロックあたりの主な行列積は次の通りである。

| 演算 | 行列の形 | 計算量のオーダー |
| --- | --- | --- |
| Q, K, V, 出力射影 | (n×d)(d×d) ×4 | O(n d²) |
| QKᵀ（スコア計算） | (n×d)(d×n) | O(n² d) |
| AV（加重和） | (n×n)(n×d) | O(n² d) |
| FFN | (n×d)(d×4d) ×2 | O(n d²)（係数8） |

ここから2つの重要な事実が読める。第一に、**系列長 $$n$$ に対して2乗で効くのは Attention のスコア計算と加重和だけ**である。$$n$$ が $$d$$ に比べて小さい領域では FFN の $$O(n d^2)$$ が支配的で、$$n$$ が大きくなるほど $$O(n^2 d)$$ が牙をむく。長文コンテキストの研究（FlashAttention、線形 Attention、スパース Attention 等）がすべてこの $$n^2$$ 項を標的にしているのは、この勘定から必然である。なお FlashAttention は計算量のオーダー自体は変えず、$$n \times n$$ 行列を HBM に実体化させないメモリアクセスの再編成によって高速化している——ボトルネックが演算でなくメモリ帯域にあることを突いた設計だ（Dao et al., 2022）。

第二に、パラメータ数を数えると **FFN が過半を占める**。1ブロックあたり、Attention 系が $$4d^2$$（$$W_Q, W_K, W_V, W_O$$）、FFN が $$8d^2$$（$$d \times 4d$$ が2枚）で、比率はちょうど 1 : 2 になる。「Transformer の知識は FFN に宿る」という解釈研究の示唆は、単純にパラメータの置き場所の比率とも整合している。

```python
# 1ブロックのパラメータ数の勘定（バイアス・LN項は省略）
d = 4096                      # 例: 7Bクラスのモデル次元
attn = 4 * d * d              # W_Q, W_K, W_V, W_O   → 67M
ffn  = 2 * d * (4 * d)        # W_1 (d→4d), W_2 (4d→d) → 134M
block = attn + ffn            # ≈ 201M / ブロック
# これを L=32 層積むと ≈ 6.4B。埋め込みを足すと「7B」の数字感になる
```

## 学習 — 何を最小化しているのか

最後に、この巨大な写像が何を目指して調整されるのかを式で確認する。Decoder 型言語モデルの目的関数は、系列の同時確率の連鎖律分解に対する負の対数尤度である。

$$
P(t_1, \ldots, t_n) = \prod_{i=1}^{n} P(t_i \mid t_{<i}), \qquad
\mathcal{L} = -\sum_{i} \log \mathrm{softmax}(h_i W_{\mathrm{LM}})_{t_i}
$$

$$h_i$$ は位置 $$i$$ の最終層表現、$$W_{\mathrm{LM}} \in \mathbb{R}^{d \times \lvert V \rvert}$$ は LM head（埋め込み $$E$$ と共有されることが多い）。因果マスクのおかげで全位置の損失が1パスで計算できる（teacher forcing）。

一方 Encoder 型（BERT）は系列の一部（約15%）を `[MASK]` に置換し、その位置だけで同じ形のクロスエントロピーを最小化する（Masked Language Modeling）。目的関数の形はどちらも「ソフトマックス分布と正解 one-hot のクロスエントロピー」で共通であり、違いは**条件付ける文脈が片側か両側か**に尽きる。

まとめよう。Transformer とは、**「内積類似度による微分可能な検索（Attention）」と「位置ごとの連想記憶（FFN）」を、残差ストリーム上で交互に $$L$$ 回適用する関数**である。$$\sqrt{d_k}$$ は勾配を生かす温度、マスクは Encoder / Decoder を分ける唯一のスイッチ、$$O(n^2 d)$$ はスコア行列の宿命——主要な設計判断はすべて、この短い式たちの中から読み取れる。

> この記事のソフトマックス出力を「確率」として扱えるか——キャリブレーションの観点からの解説は、別記事「[分類モデルの確信度 — その確率はどこから来て、どこまで信用できるのか]({{ '/ja/classification-confidence/' | relative_url }})」にまとめている。

## 参考文献

- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. *ICML 2020*.
- Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- Geva, M., et al. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2021*.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits.
- Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*.
