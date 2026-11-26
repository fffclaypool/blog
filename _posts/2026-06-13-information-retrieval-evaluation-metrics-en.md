---
layout: post
title: "Evaluation Metrics for Information Retrieval"
description: "The standard evaluation metrics for information retrieval, from Precision and Recall to AP, MAP, nDCG and MRR, explained through their defining formulas."
date: 2026-06-13
lang: en
ref: ir-metrics
permalink: /en/information-retrieval-metrics/
translation: /ja/information-retrieval-metrics/
translation_label: "日本語"
---

This article goes through the standard metrics for evaluating information retrieval (IR) systems, focusing on their defining formulas and what they mean. It covers set-based metrics, rank-aware metrics, aggregation across queries, and the assumptions behind offline evaluation, in that order.

## 1. Preliminaries and Notation

For a query $$q$$, the target document collection (corpus) is written as follows.

$$
D = \{d_1, d_2, \ldots, d_N\}
$$

Each document $$d$$ is assumed to have a defined relevance with respect to the query $$q$$.

- Binary relevance: $$\mathrm{rel}(d) \in \{0, 1\}$$, which is 1 if the document is relevant and 0 if not.
- Graded relevance: $$\mathrm{rel}(d) \in \{0, 1, 2, \ldots, g_{\max}\}$$, for example 0 = non-relevant, 1 = marginally relevant, 2 = relevant, 3 = perfect match.

The system returns results from $$D$$, possibly as a ranked top-$$k$$ list. The returned set is written $$R$$, and the set of truly relevant documents is $$\mathrm{Rel} = \{\, d : \mathrm{rel}(d) = 1 \,\}$$. For ranking, the ordered list returned by the system is written $$\pi = (d_{(1)}, d_{(2)}, \ldots, d_{(n)})$$, where $$d_{(i)}$$ is the document at rank $$i$$.

## 2. Foundational Quantities from the Confusion Matrix

Under binary relevance, retrieval results can be classified in a $$2 \times 2$$ table by whether a document was retrieved and whether it is relevant.

| | Relevant | Non-relevant |
|---|---|---|
| Retrieved | TP (true positive) | FP (false positive) |
| Not retrieved | FN (false negative) | TN (true negative) |

These quantities can be expressed with sets as follows.

$$
\mathit{TP} = \lvert R \cap \mathrm{Rel} \rvert,\quad
\mathit{FP} = \lvert R \setminus \mathrm{Rel} \rvert,\quad
\mathit{FN} = \lvert \mathrm{Rel} \setminus R \rvert,\quad
\mathit{TN} = \lvert D \setminus (R \cup \mathrm{Rel}) \rvert
$$

## 3. Set-Based Metrics

These are classical metrics that ignore order and compare only the returned set with the set of relevant documents. They follow directly from the confusion matrix above.

### Precision <span class="pill">Correctness of results</span>

Precision is the fraction of retrieved documents that are actually relevant.

$$
\mathrm{Precision} = \frac{\mathit{TP}}{\mathit{TP} + \mathit{FP}} = \frac{\lvert R \cap \mathrm{Rel} \rvert}{\lvert R \rvert}
$$

It measures whether the returned items are correct, and it matters most when false positives are costly, as in a search that returns a small, carefully selected set.
{:.note}

### Recall <span class="pill">Few misses</span>

Recall is the fraction of all relevant documents that were retrieved.

$$
\mathrm{Recall} = \frac{\mathit{TP}}{\mathit{TP} + \mathit{FN}} = \frac{\lvert R \cap \mathrm{Rel} \rvert}{\lvert \mathrm{Rel} \rvert}
$$

It measures whether relevant documents were found without omissions, and it matters most when false negatives are costly, as in exhaustive surveys or legal search.
{:.note}

> **The precision–recall trade-off:** Increasing the number of retrieved documents $$\lvert R \rvert$$ tends to raise recall and lower precision. The two generally conflict, so neither one alone describes performance. The F-measure combines them into one value.

### F-measure <span class="pill">Harmonic mean of P and R</span>

The F-measure is the harmonic mean of precision $$P$$ and recall $$R$$. Its weighted general form is as follows.

$$
F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}
$$

The most common case, $$\beta = 1$$, weights the two equally and is called the F1 score.

$$
F_1 = \frac{2PR}{P + R} = \frac{2\,\mathit{TP}}{2\,\mathit{TP} + \mathit{FP} + \mathit{FN}}
$$

$$\beta > 1$$ emphasizes recall and $$\beta < 1$$ emphasizes precision. Because it is a harmonic mean, $$F_\beta$$ is small whenever either value is very small, so both have to be high for a high score.
{:.note}

## 4. Rank-Aware Metrics

Real search results are ordered. Set-based metrics cannot express the intuition that placing relevant documents higher is better. The metrics below take the cutoff $$k$$ or the rank itself into account.

### P@k <span class="pill">Precision at k</span>

P@k is the fraction of relevant documents among the top $$k$$.

$$
\mathrm{P}@k = \frac{1}{k} \sum_{i=1}^{k} \mathrm{rel}_i, \qquad (\mathrm{rel}_i \in \{0, 1\})
$$

Users usually look only at the top results, so $$k = 10$$ is a common choice. Its weakness is that the score cannot reach 1.0, even in theory, when the total number of relevant documents $$\lvert \mathrm{Rel} \rvert$$ is smaller than $$k$$.
{:.note}

### AP <span class="pill">Average Precision</span>

AP averages P@k over the ranks at which relevant documents appear. It is higher when relevant documents are placed near the top.

$$
\mathrm{AP} = \frac{1}{\lvert \mathrm{Rel} \rvert} \sum_{k=1}^{n} \mathrm{P}@k \cdot \mathrm{rel}_k
$$

The factor $$\mathrm{rel}_k$$ means that P@k is added only at positions where a relevant document appears. AP is a discrete approximation of the area under the precision–recall curve (AUC-PR).

Example: if relevant items appear at ranks 1 and 3 out of 5, then $$\mathrm{AP} = \tfrac{1}{2}\!\left(\tfrac{1}{1} + \tfrac{2}{3}\right) \approx 0.833$$.
{:.note}

### MAP <span class="pill">Mean Average Precision</span>

MAP is the mean of AP over a set of queries $$\mathcal{Q}$$ and is one of the most widely used ranking metrics.

$$
\mathrm{MAP} = \frac{1}{\lvert \mathcal{Q} \rvert} \sum_{q \in \mathcal{Q}} \mathrm{AP}(q)
$$

It evens out differences in difficulty between queries and represents the system's stable overall ranking performance.
{:.note}

## 5. Metrics for Graded Relevance

Binary relevance is not enough when "marginally relevant" has to be distinguished from "a perfect match". DCG-family metrics combine graded relevance with a discount by rank.

### DCG@k <span class="pill">Discounted Cumulative Gain</span>

DCG defines each document's gain from its relevance and accumulates the gains with a logarithmic discount for lower ranks. There are two common definitions.

$$
\mathrm{DCG}@k = \sum_{i=1}^{k} \frac{\mathrm{rel}_i}{\log_2(i + 1)} \qquad \text{(linear gain)}
$$

$$
\mathrm{DCG}@k = \sum_{i=1}^{k} \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i + 1)} \qquad \text{(exponential gain: emphasizes high relevance)}
$$

The discount factor $$1/\log_2(i+1)$$ reduces the contribution as the rank goes down. There is no discount at $$i = 1$$, since $$\log_2 2 = 1$$.
{:.note}

### nDCG@k <span class="pill">Normalized DCG</span>

The range of DCG varies from query to query, so it is normalized by the DCG of the ideal ranking (IDCG, the DCG obtained when documents are sorted by descending relevance), which bounds it to $$[0, 1]$$.

$$
\mathrm{nDCG}@k = \frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}, \qquad
\mathrm{IDCG}@k = \sum_{i=1}^{\lvert \mathrm{REL}_k \rvert} \frac{2^{\mathrm{rel}^{*}_i} - 1}{\log_2(i + 1)}
$$

$$\mathrm{rel}^{*}_i$$ is the ideal sequence sorted by descending relevance, and $$\mathrm{REL}_k$$ is the ideal top-$$k$$ set. $$\mathrm{nDCG} = 1$$ means the ranking matches the ideal one. The metric is well suited to comparisons across queries and systems.
{:.note}

## 6. Position-Based Metrics

### MRR <span class="pill">Mean Reciprocal Rank</span>

MRR averages, over all queries, the reciprocal of the rank $$\mathrm{rank}_q$$ at which the first relevant document appears. It measures how quickly the first correct answer is reached.

$$
\mathrm{MRR} = \frac{1}{\lvert \mathcal{Q} \rvert} \sum_{q \in \mathcal{Q}} \frac{1}{\mathrm{rank}_q}
$$

A correct answer at rank 1 contributes 1, at rank 2 it contributes 0.5, and at rank 3 about 0.33. MRR is useful for question answering and known-item search, where one correct answer is enough.
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
> - When coverage matters most, use Recall / MAP
> - When the quality of the top few results matters, use P@k / nDCG@k
> - When a single correct answer is enough (QA, known-item search), use MRR
> - For web search with graded relevance, use nDCG

> **A note on assumptions:** These metrics assume offline evaluation with per-document relevance judgments. In practice not every document can be judged, so pooling is often used to judge only a subset, which introduces the approximation of treating unjudged documents as non-relevant. When aggregating across queries, it is also advisable to report statistical significance tests (for example a paired t-test or the Wilcoxon signed-rank test) along with the mean.
