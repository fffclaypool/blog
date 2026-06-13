---
layout: post
title: "Bi-Encoder / Late Interaction / Cross-Encoder: The Mathematics of Where Interaction Happens"
description: "Where do the query and document mix? Through the lens of score-function factorizability, we read off the accuracy–latency trade-off of the three approaches."
date: 2026-06-13 12:00:00
lang: en
ref: bi-late-cross
permalink: /en/bi-late-cross-encoder/
translation: /ja/bi-late-cross-encoder/
translation_label: "日本語"
---

In neural search and reranking, a model assigns a relevance score $$s(q, d) \in \mathbb{R}$$ to a query $$q$$ and a document $$d$$. Bi-Encoder, Late Interaction, and Cross-Encoder all compute this score, yet they differ sharply in **accuracy** and **latency**. This article shows, with the math, that every one of those differences follows from a single question: **where do the query and the document interact?**

## Notation and setup

- Let the query be a token sequence $$q = (q_1, \ldots, q_n)$$ and the document a token sequence $$d = (d_1, \ldots, d_m)$$.
- Write an encoder (a Transformer, say) as $$\mathrm{Enc}(\cdot)$$.
- The goal is to compute a relevance score $$s(q, d) \in \mathbb{R}$$ and rank documents by it.

We focus on two quantities.

- **Accuracy**: how richly the score function $$s$$ can express the relationship between $$q$$ and $$d$$ (its expressiveness).
- **Latency**: how much computation happens between a query arriving and its score coming out.

The key that separates the two is whether $$s$$ **factorizes** into the form

$$
s(q, d) = F\big(\phi(q),\ \psi(d)\big).
$$

If the document-side representation $$\psi(d)$$ does not depend on $$q$$, then $$\psi(d)$$ **can be computed before any query arrives**. The only work left at query time is "encode the query + a cheap combiner $$F$$," which keeps latency low. If it does not factorize, the model must run in full for every $$(q, d)$$ pair, and latency rises. Where you place the interaction decides accuracy and latency at the same time.

## Bi-Encoder

Encode the query and the document independently, each into a single vector.

$$
u = \mathrm{Enc}_q(q) \in \mathbb{R}^k, \qquad v = \mathrm{Enc}_d(d) \in \mathbb{R}^k
$$

The score is their inner product (or cosine similarity).

$$
s_{\mathrm{bi}}(q, d) = \langle u, v \rangle = u^\top v
$$

Encoding the query and document into dense vectors and searching by vector closeness is exactly what is known as **vector search**.

This is the simplest factorized form: $$\phi(q) = u$$, $$\psi(d) = v$$, $$F = \langle \cdot, \cdot \rangle$$. Since $$v$$ does not depend on $$q$$, the document vectors can be prepared in advance. All that remains at query time is "one query encoding + one inner product ($$O(k)$$)," giving the **lowest latency of the three**.

The cost is accuracy. The interaction between $$q$$ and $$d$$ is compressed into nothing but the inner product of two pooled vectors, so token-level correspondences are lost. That caps expressiveness — the representation bottleneck.

## Cross-Encoder

**Concatenate the query and document and pass them through a single encoder.**

$$
s_{\mathrm{cross}}(q, d) = \mathbf{w}^\top\, \mathrm{Enc}\big(\,[\,q \,;\, \mathrm{[SEP]} \,;\, d\,]\,\big)_{\mathrm{[CLS]}}
$$

Read the formula like this. First, the query and document tokens are joined into one sequence, separated by special tokens.

$$
x = (\,\mathrm{[CLS]},\ q_1, \ldots, q_n,\ \mathrm{[SEP]},\ d_1, \ldots, d_m,\ \mathrm{[SEP]}\,)
$$

This is $$n + m + 3$$ tokens in total (query $$n$$ + document $$m$$ + three special tokens: one `[CLS]` and two `[SEP]`). Passing it through the encoder yields a contextualized output $$\mathrm{Enc}(x) \in \mathbb{R}^{(n+m+3) \times k}$$ — one $$k$$-dimensional vector per token. The output vector at the leading $$\mathrm{[CLS]}$$ position, $$\mathrm{Enc}(x)_{\mathrm{[CLS]}} \in \mathbb{R}^k$$, can be taken as a summary of the whole pair $$(q, d)$$; an inner product with a learned weight vector $$\mathbf{w} \in \mathbb{R}^k$$ collapses it to a single score.

Through self-attention over all tokens, every query token and every document token **interact directly inside the network**. As a result, even a document token's representation changes depending on the query, so the score is a non-separable function of the joined input $$[\,q ; d\,]$$ and cannot be split into $$\phi(q)$$ and $$\psi(d)$$.

This inability to split feeds straight back into latency.

- The document side **cannot be precomputed**. For every $$(q, d)$$ pair you want to score, the model must run in full once.
- Self-attention costs $$O\big((n + m)^2\big)$$ per pair.
- So the per-pair latency is the **highest of the three**.

In return, it expresses interaction the most richly, so accuracy is typically the highest. It is the method that buys accuracy at the price of latency.

## Late Interaction (ColBERT)

Encoding the query and document independently is the same as in the Bi-Encoder, but instead of collapsing to a single vector, it keeps a **per-token set of vectors**.

$$
U = (u_1, \ldots, u_n), \qquad V = (v_1, \ldots, v_m), \qquad u_i, v_j \in \mathbb{R}^k
$$

(Each vector is assumed normalized.) The score is given by **MaxSim**.

$$
s_{\mathrm{late}}(q, d) = \sum_{i=1}^{n} \max_{1 \le j \le m} \langle u_i, v_j \rangle
$$

That is: for each query token, take the similarity to its most similar document token, and sum over the query.

<figure>
  <img src="{{ '/assets/img/maxsim-en.svg' | relative_url }}" alt="MaxSim: each query token selects the document token with the highest similarity (the row-wise max), and those maxima are summed into the score">
  <figcaption>MaxSim: for each query token, take the similarity to its most similar document token (the row-wise max), and sum them into the score.</figcaption>
</figure>

Encoding is still independent ($$v_j$$ does not depend on $$q$$), so the document token vectors can be precomputed. The interaction, MaxSim, is a cheap $$\max$$-and-sum performed **after** encoding. Query-time work is about $$O(n m k)$$ — heavier than the Bi-Encoder's inner product, but far lighter than the Cross-Encoder's self-attention. Latency lands **in between**.

Accuracy is in between too. Keeping token-level correspondences makes it more expressive than the Bi-Encoder's single-vector inner product, yet it falls short of the Cross-Encoder's full cross-layer interaction.

<figure>
  <img src="{{ '/assets/img/bi-late-cross-en.svg' | relative_url }}" alt="Structural comparison of the three methods. Bi and Late keep two independent encoder towers and interact at the end, while Cross feeds the concatenated input through a single encoder">
  <figcaption>The three architectures. Bi and Late keep two independent encoder towers and place the interaction at the end (→ the document side can be precomputed; query time is light). Cross concatenates from the start and runs one encoder (→ full computation every time).</figcaption>
</figure>

## Where the difference comes from: the locus of interaction

In the end, the three methods differ only in **where the information from $$q$$ and $$d$$ mixes.**

- **Bi**: mixing happens once, at the very end, in the inner product of two independently pooled vectors. $$s = \phi(q)^\top \psi(d)$$ (bilinear, separable).
- **Late**: mixing also happens at the end, but at token granularity through $$\max$$. $$s = \sum_i \max_j \langle \phi_i(q), \psi_j(d) \rangle$$ (the encoders are separable; the interaction is a fixed operator).
- **Cross**: mixing happens in **every layer** of the network, through attention. There is no choice but to write $$s = h(q, d)$$ — non-separable.

This place where they mix decides latency and accuracy **together**.

**Latency** is set by factorizability. The separable Bi and Late can precompute the document side $$\psi(d)$$, so query-time work is light. The non-separable Cross runs in full each time, and is heavy.

**Accuracy** is set by expressiveness. The Bi-Encoder is a special case of Late Interaction (with $$n = m = 1$$, a single pooled token), and Late Interaction is a restricted Cross-Encoder (no cross-attention within layers, just a fixed MaxSim at the end). So as function classes,

$$
\mathcal{F}_{\mathrm{bi}} \subseteq \mathcal{F}_{\mathrm{late}} \subseteq \mathcal{F}_{\mathrm{cross}}.
$$

In short, **accuracy and latency both grow in the same order (Bi → Late → Cross).** The thicker the interaction, the higher the expressiveness — but the more of that interaction must be computed at query time, the higher the latency. So choosing a method comes down to one axis: within the latency you can tolerate, how much accuracy can you squeeze out.

## Accuracy and latency at a glance

| Method | Precompute (offline) | Query-time compute | Latency | Accuracy |
|---|---|---|---|---|
| Bi-Encoder | document → 1 vector | encode + inner product, O(k) | lowest | low–mid |
| Late Interaction | document → per-token vectors | MaxSim, O(n·m·k) | mid | mid–high |
| Cross-Encoder | none | concatenate + self-attention, O((n+m)²) | highest | highest |

## Using them in practice

In practice, this accuracy–latency trade-off is usually handled with a two-stage setup.

- **Stage 1 (retrieval)**: low latency is essential, so narrow the candidates with a precomputable Bi-Encoder (or Late Interaction).
- **Stage 2 (reranking)**: the candidates are now down to the top $$K$$, so the high per-item latency of a Cross-Encoder becomes affordable, and it lifts the final accuracy.
- Late Interaction sits in the middle and can serve on its own as a retriever that is "more accurate than Bi, lower-latency than Cross."

The standard recipe: while there are still many candidates, narrow them with a fast but moderately accurate method; once they are narrowed enough, apply a slow but highly accurate one.

> For the mathematical definitions of the evaluation metrics (Recall, nDCG, MAP, and so on), see the companion article "[Evaluation Metrics for Information Retrieval](/blog/en/information-retrieval-metrics/)."
