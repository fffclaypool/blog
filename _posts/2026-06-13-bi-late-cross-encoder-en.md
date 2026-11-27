---
layout: post
title: "Bi-Encoder, Late Interaction, Cross-Encoder: Comparing Accuracy and Latency"
description: "This article compares the accuracy and latency of Bi-Encoder, Late Interaction and Cross-Encoder models by where the query and the document interact, that is, by whether the score function factorizes."
date: 2026-06-13 12:00:00
lang: en
ref: bi-late-cross
permalink: /en/bi-late-cross-encoder/
translation: /ja/bi-late-cross-encoder/
translation_label: "日本語"
---

In neural search and reranking, a model assigns a relevance score $$s(q, d) \in \mathbb{R}$$ to a query $$q$$ and a document $$d$$. Bi-Encoder, Late Interaction and Cross-Encoder models all compute this score, but they differ considerably in accuracy and latency. This article shows with equations that those differences follow from where the query and the document interact.

## Notation and setup

- Let the query be a token sequence $$q = (q_1, \ldots, q_n)$$ and the document a token sequence $$d = (d_1, \ldots, d_m)$$.
- Write an encoder (for example a Transformer) as $$\mathrm{Enc}(\cdot)$$.
- The goal is to compute a relevance score $$s(q, d) \in \mathbb{R}$$ and rank documents by it.

The comparison uses two criteria.

- Accuracy: how finely the score function $$s$$ can express the relationship between $$q$$ and $$d$$ (its expressiveness).
- Latency: how much computation is needed between the arrival of a query and the output of its score.

Both are determined by whether $$s$$ factorizes into the form

$$
s(q, d) = F\big(\phi(q),\ \psi(d)\big).
$$

If the document-side representation $$\psi(d)$$ does not depend on $$q$$, it can be computed before any query arrives. The work left at query time is encoding the query and applying a cheap combiner $$F$$, so latency is low. If $$s$$ does not factorize, the full model has to run for every $$(q, d)$$ pair, and latency is high. Where the interaction is placed determines both accuracy and latency.

## Bi-Encoder

The query and the document are encoded independently, each into a single vector.

$$
u = \mathrm{Enc}_q(q) \in \mathbb{R}^k, \qquad v = \mathrm{Enc}_d(d) \in \mathbb{R}^k
$$

The score is their inner product (or cosine similarity).

$$
s_{\mathrm{bi}}(q, d) = \langle u, v \rangle = u^\top v
$$

Vector search, which encodes queries and documents into dense vectors and retrieves by vector closeness, is this approach.

This is the simplest factorized form, with $$\phi(q) = u$$, $$\psi(d) = v$$ and $$F = \langle \cdot, \cdot \rangle$$. Because $$v$$ does not depend on $$q$$, the document vectors can be prepared in advance. What remains at query time is one query encoding and one inner product ($$O(k)$$), which gives the lowest latency of the three.

The cost is accuracy. The interaction between $$q$$ and $$d$$ is compressed into the inner product of two pooled vectors, so token-level correspondences are lost. This limits expressiveness and is known as the representation bottleneck.

## Cross-Encoder

The query and the document are concatenated and passed through a single encoder.

$$
s_{\mathrm{cross}}(q, d) = \mathbf{w}^\top\, \mathrm{Enc}\big(\,[\,q \,;\, \mathrm{[SEP]} \,;\, d\,]\,\big)_{\mathrm{[CLS]}}
$$

The formula can be read step by step. First, the query and document tokens are joined into one sequence with special tokens around them.

$$
x = (\,\mathrm{[CLS]},\ q_1, \ldots, q_n,\ \mathrm{[SEP]},\ d_1, \ldots, d_m,\ \mathrm{[SEP]}\,)
$$

The sequence has $$n + m + 3$$ tokens in total (query $$n$$ + document $$m$$ + three special tokens: one `[CLS]` and two `[SEP]`). Passing it through the encoder yields a contextualized output $$\mathrm{Enc}(x) \in \mathbb{R}^{(n+m+3) \times k}$$, with one $$k$$-dimensional vector per token. The output vector at the leading $$\mathrm{[CLS]}$$ position, $$\mathrm{Enc}(x)_{\mathrm{[CLS]}} \in \mathbb{R}^k$$, can be regarded as a summary of the whole pair $$(q, d)$$. Its inner product with a learned weight vector $$\mathbf{w} \in \mathbb{R}^k$$ gives a single score.

Self-attention over all tokens lets every query token and every document token interact directly inside the network. The representation of a document token therefore changes with the query, so the score is a non-separable function of the joined input $$[\,q ; d\,]$$ and cannot be split into $$\phi(q)$$ and $$\psi(d)$$.

This has direct consequences for latency.

- The document side cannot be precomputed. The full model has to run once for every $$(q, d)$$ pair to be scored.
- Self-attention costs $$O\big((n + m)^2\big)$$ per pair.
- The per-pair latency is therefore the highest of the three.

In return, a Cross-Encoder expresses the interaction in the most detail, and its accuracy is typically the highest.

## Late Interaction (ColBERT)

Late Interaction encodes the query and the document independently, as a Bi-Encoder does, but keeps one vector per token instead of pooling them into a single vector.

$$
U = (u_1, \ldots, u_n), \qquad V = (v_1, \ldots, v_m), \qquad u_i, v_j \in \mathbb{R}^k
$$

Each vector is assumed to be normalized. The score is computed with MaxSim.

$$
s_{\mathrm{late}}(q, d) = \sum_{i=1}^{n} \max_{1 \le j \le m} \langle u_i, v_j \rangle
$$

For each query token, MaxSim takes the similarity to its most similar document token, and these values are summed over the query.

<figure>
  <img src="{{ '/assets/img/maxsim-en.svg' | relative_url }}" alt="MaxSim: each query token selects the document token with the highest similarity (the row-wise max), and those maxima are summed into the score">
  <figcaption>MaxSim: for each query token, take the similarity to its most similar document token (the row-wise max), and sum them into the score.</figcaption>
</figure>

Encoding is independent ($$v_j$$ does not depend on $$q$$), so the document token vectors can be precomputed. The interaction, MaxSim, is a cheap $$\max$$-and-sum operation performed after encoding. Query-time work is about $$O(n m k)$$, which is heavier than a Bi-Encoder's inner product and far lighter than a Cross-Encoder's self-attention. Latency falls between the two.

Accuracy is also in between. Keeping token-level correspondences makes it more expressive than a Bi-Encoder's single-vector inner product, but it does not reach a Cross-Encoder, which interacts in every layer.

<figure>
  <img src="{{ '/assets/img/bi-late-cross-en.svg' | relative_url }}" alt="Structural comparison of the three methods. Bi and Late keep two independent encoder towers and interact at the end, while Cross feeds the concatenated input through a single encoder">
  <figcaption>The three architectures. Bi and Late keep two independent encoder towers and place the interaction at the end, so the document side can be precomputed and query-time work is light. Cross concatenates the inputs from the start and runs one encoder, so the full computation is repeated every time.</figcaption>
</figure>

## Where the difference comes from: the locus of interaction

The three methods differ in where the information from $$q$$ and $$d$$ is mixed.

- Bi: mixed once at the very end, in the inner product of two independently pooled vectors. $$s = \phi(q)^\top \psi(d)$$ (bilinear, separable).
- Late: also mixed at the end, but at token granularity through $$\max$$. $$s = \sum_i \max_j \langle \phi_i(q), \psi_j(d) \rangle$$ (the encoders are separable, and the interaction is a fixed operator).
- Cross: mixed in every layer of the network through attention. The score can only be written as $$s = h(q, d)$$ and is non-separable.

The place where they are mixed determines both latency and accuracy.

Latency is determined by factorizability. Bi and Late are separable, so the document side $$\psi(d)$$ can be precomputed and query-time work is light. Cross is not separable, so the full model runs every time and is heavy.

Accuracy is determined by expressiveness. A Bi-Encoder is a special case of Late Interaction (with $$n = m = 1$$, a single pooled token), and Late Interaction can be seen as a restricted Cross-Encoder (no cross-attention within layers and a fixed MaxSim at the end). As function classes, this gives

$$
\mathcal{F}_{\mathrm{bi}} \subseteq \mathcal{F}_{\mathrm{late}} \subseteq \mathcal{F}_{\mathrm{cross}}.
$$

Accuracy and latency both increase in the same order (Bi → Late → Cross). More interaction means more expressiveness, and because that interaction is computed at query time, it also means more latency. Choosing a method therefore comes down to one question: how much accuracy can be obtained within the latency that can be tolerated.

## Accuracy and latency at a glance

| Method | Precompute (offline) | Query-time compute | Latency | Accuracy |
|---|---|---|---|---|
| Bi-Encoder | document → 1 vector | encode + inner product, O(k) | lowest | low–mid |
| Late Interaction | document → per-token vectors | MaxSim, O(n·m·k) | mid | mid–high |
| Cross-Encoder | none | concatenate + self-attention, O((n+m)²) | highest | highest |

## Using them in practice

In practice the trade-off is usually handled with a two-stage setup.

- Stage 1 (retrieval): low latency is essential, so the candidates are narrowed with a precomputable Bi-Encoder (or Late Interaction).
- Stage 2 (reranking): the candidates are down to the top $$K$$, so the high per-item latency of a Cross-Encoder is affordable, and it raises the final accuracy.
- Late Interaction sits between the two and can also be used on its own as a retriever that is more accurate than Bi and lower in latency than Cross.

While there are still many candidates, a fast and moderately accurate method reduces their number. Once they are few enough, a slow and highly accurate method is applied.

> For the mathematical definitions of the evaluation metrics (Recall, nDCG, MAP and so on), see the companion article "[Evaluation Metrics for Information Retrieval](/blog/en/information-retrieval-metrics/)".
