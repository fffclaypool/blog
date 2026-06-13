---
layout: post
title: "Evaluation Metrics for Information Retrieval — A Mathematical Guide"
description: "From Precision/Recall to AP, MAP, nDCG, and MRR, understood through their definitions."
date: 2026-06-13
lang: en
ref: ir-metrics
permalink: /en/information-retrieval-metrics/
translation: /ja/information-retrieval-metrics/
translation_label: "日本語"
---

This article organizes the standard metrics used to evaluate information retrieval (IR) systems, focusing on their defining formulas and what they mean. We proceed from set-based metrics, to rank-aware metrics, to aggregation across multiple queries, and finally to the assumptions behind offline evaluation.

## 1. Preliminaries and Notation

For a query $$q$$, let the target document collection (corpus) be:

$$
D = \{d_1, d_2, \ldots, d_N\}
$$

Each document $$d$$ is assumed to have a defined relevance with respect to the query $$q$$.

- **Binary relevance**: $$\mathrm{rel}(d) \in \{0, 1\}$$. 1 if relevant, 0 if not.
- **Graded relevance**: $$\mathrm{rel}(d) \in \{0, 1, 2, \ldots, g_{\max}\}$$. e.g., 0 = non-relevant, 1 = marginally relevant, 2 = relevant, 3 = perfect match.

The system returns results from $$D$$ (or as a ranked top-$$k$$ list). We write the returned set as $$R$$ and the set of truly relevant documents as $$\mathrm{Rel} = \{\, d : \mathrm{rel}(d) = 1 \,\}$$. When discussing ranking, consider the ordered list returned by the system, $$\pi = (d_{(1)}, d_{(2)}, \ldots, d_{(n)})$$, where $$d_{(i)}$$ is the document at rank $$i$$.

> **Notation summary:** $$\lvert \cdot \rvert$$ denotes set cardinality, and $$\mathbf{1}[\cdot]$$ is the indicator function that returns 1 if its condition is true and 0 otherwise. $$\mathrm{rel}_i \equiv \mathrm{rel}(d_{(i)})$$ is the relevance of the document at rank $$i$$.

## 2. Foundational Quantities from the Confusion Matrix

Under binary relevance, retrieval results can be classified in a $$2 \times 2$$ scheme along "retrieved or not" and "relevant or not."

| | Relevant | Non-relevant |
|---|---|---|
| Retrieved | TP (true positive) | FP (false positive) |
| Not retrieved | FN (false negative) | TN (true negative) |

These can be expressed with sets as follows.

$$
\mathit{TP} = \lvert R \cap \mathrm{Rel} \rvert,\quad
\mathit{FP} = \lvert R \setminus \mathrm{Rel} \rvert,\quad
\mathit{FN} = \lvert \mathrm{Rel} \setminus R \rvert,\quad
\mathit{TN} = \lvert D \setminus (R \cup \mathrm{Rel}) \rvert
$$

## 3. Set-Based Metrics

Classical metrics that ignore order and compare only the "returned set" against the "answer set." They follow directly from the confusion matrix above.

### Precision <span class="pill">Correctness of results</span>

The fraction of retrieved documents that are actually relevant.

$$
\mathrm{Precision} = \frac{\mathit{TP}}{\mathit{TP} + \mathit{FP}} = \frac{\lvert R \cap \mathrm{Rel} \rvert}{\lvert R \rvert}
$$

Measures "are the returned items correct?" Emphasized when FP is costly (e.g., search that returns a carefully selected few).
{:.note}

### Recall <span class="pill">Few misses</span>

The fraction of all relevant documents that were retrieved.

$$
\mathrm{Recall} = \frac{\mathit{TP}}{\mathit{TP} + \mathit{FN}} = \frac{\lvert R \cap \mathrm{Rel} \rvert}{\lvert \mathrm{Rel} \rvert}
$$

Measures "did we retrieve relevant documents without missing any?" Emphasized when FN is costly (e.g., exhaustive surveys, legal search).
{:.note}

> **The precision–recall trade-off:** Increasing the number retrieved, $$\lvert R \rvert$$, tends to raise recall but lower precision. The two generally conflict, so neither alone captures performance. The F-measure unifies them.

### F-measure <span class="pill">Harmonic mean of P and R</span>

The harmonic mean of precision $$P$$ and recall $$R$$. Its weighted general form is:

$$
F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}
$$

The most common case, $$\beta = 1$$, weights both equally, giving the **F1 score**.

$$
F_1 = \frac{2PR}{P + R} = \frac{2\,\mathit{TP}}{2\,\mathit{TP} + \mathit{FP} + \mathit{FN}}
$$

$$\beta > 1$$ emphasizes recall; $$\beta < 1$$ emphasizes precision. Being a harmonic mean, $$F_\beta$$ becomes small whenever either value is extremely small (it demands both at once).
{:.note}

## 4. Rank-Aware Metrics

Real search results are ordered. The intuition that placing relevant documents higher is better cannot be captured by set-based metrics. Below, we incorporate the rank $$k$$ or the rank itself.

### P@k <span class="pill">Precision at k</span>

The fraction of relevant documents among the top $$k$$.

$$
\mathrm{P}@k = \frac{1}{k} \sum_{i=1}^{k} \mathrm{rel}_i, \qquad (\mathrm{rel}_i \in \{0, 1\})
$$

Since users typically view only the top results, $$k = 10$$ is common. However, if the total number of relevant documents $$\lvert \mathrm{Rel} \rvert$$ is smaller than $$k$$, the score cannot reach 1.0 even in theory — a known weakness.
{:.note}

### AP <span class="pill">Average Precision</span>

Averages P@k over the ranks at which relevant documents appear. Higher when relevant documents are placed near the top.

$$
\mathrm{AP} = \frac{1}{\lvert \mathrm{Rel} \rvert} \sum_{k=1}^{n} \mathrm{P}@k \cdot \mathrm{rel}_k
$$

Here $$\mathrm{rel}_k$$ ensures that P@k is added only at the positions where relevant documents appear. AP corresponds to a discrete approximation of the area under the precision–recall curve (AUC-PR).

Example: if relevant items appear at ranks 1 and 3 out of 5, then $$\mathrm{AP} = \tfrac{1}{2}\!\left(\tfrac{1}{1} + \tfrac{2}{3}\right) \approx 0.833$$.
{:.note}

### MAP <span class="pill">Mean Average Precision</span>

The mean of AP over a set of queries $$\mathcal{Q}$$; one of the most standard ranking metrics.

$$
\mathrm{MAP} = \frac{1}{\lvert \mathcal{Q} \rvert} \sum_{q \in \mathcal{Q}} \mathrm{AP}(q)
$$

It evens out difficulty differences across queries and represents the system's stable overall ranking performance.
{:.note}

## 5. Metrics for Graded Relevance

When you want to distinguish "marginally relevant" from "a perfect match," binary relevance is insufficient. DCG-family metrics combine graded relevance with rank discounting.

### DCG@k <span class="pill">Discounted Cumulative Gain</span>

Defines each document's gain from its relevance and accumulates it with a logarithmic discount for lower ranks. Two common definitions:

$$
\mathrm{DCG}@k = \sum_{i=1}^{k} \frac{\mathrm{rel}_i}{\log_2(i + 1)} \qquad \text{(linear gain)}
$$

$$
\mathrm{DCG}@k = \sum_{i=1}^{k} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i + 1)} \qquad \text{(exponential gain: emphasizes high relevance)}
$$

The discount factor $$1/\log_2(i+1)$$ shrinks the contribution as the rank drops. There is no discount at $$i = 1$$ (since $$\log_2 2 = 1$$).
{:.note}

### nDCG@k <span class="pill">Normalized DCG</span>

Because DCG's range varies per query, normalize by the DCG of the ideal ranking (**IDCG**, the DCG obtained when documents are sorted by descending relevance) to bound it to $$[0, 1]$$.

$$
\mathrm{nDCG}@k = \frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}, \qquad
\mathrm{IDCG}@k = \sum_{i=1}^{\lvert \mathrm{REL}_k \rvert} \frac{2^{\mathrm{rel}^{*}_i} - 1}{\log_2(i + 1)}
$$

$$\mathrm{rel}^{*}_i$$ is the ideal sequence sorted by descending relevance, and $$\mathrm{REL}_k$$ is the ideal top-$$k$$ set. $$\mathrm{nDCG} = 1$$ means a perfect ideal ranking. It is well suited for comparing across queries and systems.
{:.note}

## 6. Position-Based Metrics

### MRR <span class="pill">Mean Reciprocal Rank</span>

Averages, over all queries, the reciprocal of the rank $$\mathrm{rank}_q$$ at which the first relevant document appears. Measures "how quickly you reach the first correct answer."

$$
\mathrm{MRR} = \frac{1}{\lvert \mathcal{Q} \rvert} \sum_{q \in \mathcal{Q}} \frac{1}{\mathrm{rank}_q}
$$

A correct answer at rank 1 contributes 1, at rank 2 gives 0.5, at rank 3 gives 0.33…. Useful for question answering or known-item search, where a single correct answer suffices.
{:.note}

## 7. Comparison and Summary

| Metric | Rank-aware | Relevance | Normalized | Main use |
|---|:---:|:---:|:---:|---|
| Precision / Recall | ✗ | Binary | [0,1] | Basic retrieval performance |
| F1 | ✗ | Binary | [0,1] | Balance of P and R |
| P@k | △ (top-k) | Binary | [0,1] | Top-of-list precision |
| MAP | ✓ | Binary | [0,1] | Standard ranking evaluation |
| nDCG@k | ✓ | Graded | [0,1] | Graded-relevance ranking |
| MRR | ✓ | Binary | (0,1] | Speed to first correct answer |

> **A guide to choosing metrics:**
>
> - Coverage matters most → **Recall / MAP**
> - Quality of the top few matters → **P@k / nDCG@k**
> - A single correct answer suffices (QA, known-item) → **MRR**
> - Web search with graded relevance → **nDCG**

> **A note on assumptions:** These metrics assume offline evaluation with per-document relevance judgments. In practice, not all documents can be judged, so pooling is often used to evaluate only a subset — introducing the approximation of treating unjudged documents as non-relevant. Moreover, when aggregating across queries, it is advisable to report not just the mean but also statistical significance tests (e.g., a paired t-test or the Wilcoxon signed-rank test).
