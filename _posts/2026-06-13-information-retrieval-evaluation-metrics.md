---
layout: post
title: "情報検索の評価指標 — 数理的解説"
description: "Precision / Recall から AP・MAP・nDCG・MRR まで、定義式で理解する"
date: 2026-06-13
lang: ja
ref: ir-metrics
permalink: /ja/information-retrieval-metrics/
translation: /en/information-retrieval-metrics/
translation_label: "English"
---

本稿は、情報検索（Information Retrieval, IR）システムの性能を測る代表的な評価指標を、定義式とその意味を中心に数理的に整理したものである。集合ベースの指標から順位（ランキング）を考慮した指標、複数クエリにわたる集約、そしてオフライン評価の前提までを順に扱う。

## 1. 前提と記法

あるクエリ $$q$$ に対して、検索対象となる文書集合（コーパス）を次のようにおく。

$$
D = \{d_1, d_2, \ldots, d_N\}
$$

各文書 $$d$$ には、クエリ $$q$$ に対する関連性（relevance）が定義されているとする。

- **二値関連度（binary relevance）**: $$\mathrm{rel}(d) \in \{0, 1\}$$。関連なら 1、非関連なら 0。
- **段階的関連度（graded relevance）**: $$\mathrm{rel}(d) \in \{0, 1, 2, \ldots, g_{\max}\}$$。例: 0=非関連, 1=やや関連, 2=関連, 3=完全一致。

検索システムは $$D$$ から（または順位付き上位 $$k$$ 件として）結果を返す。返された文書集合を $$R$$、真に関連する文書集合を $$\mathrm{Rel} = \{\, d : \mathrm{rel}(d) = 1 \,\}$$ と書く。ランキングを論じる場合は、システムが返した順序付きリスト $$\pi = (d_{(1)}, d_{(2)}, \ldots, d_{(n)})$$ を考え、$$d_{(i)}$$ を第 $$i$$ 位の文書とする。

## 2. 混同行列に基づく基本量

二値関連度のもとでは、検索結果を「返したか否か」と「関連か否か」の $$2 \times 2$$ で分類できる。

| | 関連 (Relevant) | 非関連 (Non-relevant) |
|---|---|---|
| 取得した (Retrieved) | TP（真陽性） | FP（偽陽性） |
| 取得しない (Not retrieved) | FN（偽陰性） | TN（真陰性） |

これらは集合を用いて次のように表せる。

$$
\mathit{TP} = \lvert R \cap \mathrm{Rel} \rvert,\quad
\mathit{FP} = \lvert R \setminus \mathrm{Rel} \rvert,\quad
\mathit{FN} = \lvert \mathrm{Rel} \setminus R \rvert,\quad
\mathit{TN} = \lvert D \setminus (R \cup \mathrm{Rel}) \rvert
$$

## 3. 集合ベースの指標

順序を無視し「返した集合」と「正解集合」だけを比べる古典的指標。上の混同行列から直接定義される。

### 適合率 Precision <span class="pill">取得結果の正確さ</span>

取得した文書のうち、実際に関連していた割合。

$$
\mathrm{Precision} = \frac{\mathit{TP}}{\mathit{TP} + \mathit{FP}} = \frac{\lvert R \cap \mathrm{Rel} \rvert}{\lvert R \rvert}
$$

「返したものは正しいか」を測る。FP を嫌う場面（少数を厳選する検索）で重視される。
{:.note}

### 再現率 Recall <span class="pill">取りこぼしの少なさ</span>

関連文書全体のうち、取得できた割合。

$$
\mathrm{Recall} = \frac{\mathit{TP}}{\mathit{TP} + \mathit{FN}} = \frac{\lvert R \cap \mathrm{Rel} \rvert}{\lvert \mathrm{Rel} \rvert}
$$

「関連文書を漏らさず拾えたか」を測る。FN を嫌う場面（網羅性が重要な調査・法務検索）で重視される。
{:.note}

> **適合率と再現率のトレードオフ:** 取得件数 $$\lvert R \rvert$$ を増やせば Recall は上がりやすいが Precision は下がりやすい。両者は一般に競合するため、単独では性能を表しきれない。これを 1 つにまとめるのが F 値である。

### F値 F-measure <span class="pill">P と R の調和平均</span>

適合率 $$P$$ と再現率 $$R$$ の調和平均。重み付き一般形は次のとおり。

$$
F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}
$$

最も一般的な $$\beta = 1$$ のとき、両者を等しく重み付けする **F1 スコア** となる。

$$
F_1 = \frac{2PR}{P + R} = \frac{2\,\mathit{TP}}{2\,\mathit{TP} + \mathit{FP} + \mathit{FN}}
$$

$$\beta > 1$$ は再現率を、$$\beta < 1$$ は適合率を重視する。調和平均ゆえ、どちらか一方が極端に小さいと $$F_\beta$$ も小さくなる（両立を要求する）。
{:.note}

## 4. 順位を考慮した指標

実際の検索結果は順序付きである。上位に関連文書を置くほど良いという直感を、集合ベースの指標は捉えられない。以下では順位 $$k$$ や順位そのものを取り込む。

### P@k <span class="pill">Precision at k</span>

上位 $$k$$ 件に含まれる関連文書の割合。

$$
\mathrm{P}@k = \frac{1}{k} \sum_{i=1}^{k} \mathrm{rel}_i, \qquad (\mathrm{rel}_i \in \{0, 1\})
$$

ユーザーは通常上位だけを見るため、$$k = 10$$ 程度がよく使われる。ただし関連文書の総数 $$\lvert \mathrm{Rel} \rvert$$ が $$k$$ 未満だと、理論上 1.0 に到達できない弱点がある。
{:.note}

### AP <span class="pill">Average Precision・平均適合率</span>

各「関連文書が出現した順位」での P@k を平均する。関連文書を上位に置くほど高くなる。

$$
\mathrm{AP} = \frac{1}{\lvert \mathrm{Rel} \rvert} \sum_{k=1}^{n} \mathrm{P}@k \cdot \mathrm{rel}_k
$$

ここで $$\mathrm{rel}_k$$ により、関連文書が現れた位置でのみ P@k が加算される。AP は Precision–Recall 曲線の下側面積（AUC-PR）の離散近似に相当する。

例: 5 件中 1・3 位が関連なら、$$\mathrm{AP} = \tfrac{1}{2}\!\left(\tfrac{1}{1} + \tfrac{2}{3}\right) \approx 0.833$$。
{:.note}

### MAP <span class="pill">Mean Average Precision</span>

複数クエリ集合 $$\mathcal{Q}$$ にわたって AP を平均した、最も標準的なランキング評価指標の一つ。

$$
\mathrm{MAP} = \frac{1}{\lvert \mathcal{Q} \rvert} \sum_{q \in \mathcal{Q}} \mathrm{AP}(q)
$$

クエリ間の難易度差を均し、システム全体の安定した順位性能を表す。
{:.note}

## 5. 段階的関連度を扱う指標

「やや関連」と「完全一致」を区別したい場合、二値では不十分である。DCG 系の指標は段階的関連度と順位割引を組み合わせる。

### DCG@k <span class="pill">Discounted Cumulative Gain</span>

各文書の利得（gain）を関連度から定め、下位ほど対数で割引いて累積する。代表的な定義は次の二つ。

$$
\mathrm{DCG}@k = \sum_{i=1}^{k} \frac{\mathrm{rel}_i}{\log_2(i + 1)} \qquad \text{(線形利得)}
$$

$$
\mathrm{DCG}@k = \sum_{i=1}^{k} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i + 1)} \qquad \text{(指数利得：高関連度を強調)}
$$

割引因子 $$1/\log_2(i+1)$$ は順位が下がるほど寄与を小さくする。$$i = 1$$ で割引なし（$$\log_2 2 = 1$$）。
{:.note}

### nDCG@k <span class="pill">Normalized DCG</span>

DCG はクエリごとに値域が変わるため、理想順位での DCG（**IDCG**、関連度降順に並べたときの DCG）で正規化し、$$[0, 1]$$ に収める。

$$
\mathrm{nDCG}@k = \frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}, \qquad
\mathrm{IDCG}@k = \sum_{i=1}^{\lvert \mathrm{REL}_k \rvert} \frac{2^{\mathrm{rel}^{*}_i} - 1}{\log_2(i + 1)}
$$

$$\mathrm{rel}^{*}_i$$ は関連度を降順に並べ替えた理想列、$$\mathrm{REL}_k$$ は上位 $$k$$ の理想集合。$$\mathrm{nDCG} = 1$$ は完全な理想順位を意味する。クエリ間・システム間の比較に適する。
{:.note}

## 6. 位置に基づく指標

### MRR <span class="pill">Mean Reciprocal Rank</span>

最初の関連文書が現れた順位 $$\mathrm{rank}_q$$ の逆数を、クエリ全体で平均する。「最初の正解にどれだけ早く到達するか」を測る。

$$
\mathrm{MRR} = \frac{1}{\lvert \mathcal{Q} \rvert} \sum_{q \in \mathcal{Q}} \frac{1}{\mathrm{rank}_q}
$$

1 位に正解があれば寄与 1、2 位なら 0.5、3 位なら 0.33…。1 つの正解で十分な質問応答・Known-item 検索で有用。
{:.note}

## 7. 指標の比較とまとめ

| 指標 | 順位考慮 | 関連度 | 正規化 | 主な用途 |
|---|:---:|:---:|:---:|---|
| Precision / Recall | × | 二値 | [0,1] | 基本的な取得性能 |
| F1 | × | 二値 | [0,1] | P・R のバランス |
| P@k | △(上位k) | 二値 | [0,1] | 上位の精度 |
| MAP | ○ | 二値 | [0,1] | 標準的なランキング評価 |
| nDCG@k | ○ | 段階的 | [0,1] | 段階関連度のランキング |
| MRR | ○ | 二値 | (0,1] | 最初の正解の早さ |

> **指標選択の指針:**
>
> - 網羅性が重要 → **Recall / MAP**
> - 上位数件の質が重要 → **P@k / nDCG@k**
> - 1 つの正解で十分（QA・既知項目検索）→ **MRR**
> - 段階的関連度を持つ Web 検索 → **nDCG**

> **評価の前提に関する注意:** これらの指標は、文書ごとの正解（relevance judgment）が与えられたオフライン評価を前提とする。実運用では全文書を判定しきれないため、プーリング（pooling）で一部のみを評価対象とすることが多く、未判定文書を非関連とみなす近似が入る。また、複数クエリの集約では平均だけでなく、統計的有意差検定（対応のある t 検定・ウィルコクソン検定など）を併せて報告するのが望ましい。
