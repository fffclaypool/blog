---
layout: post
title: "The Mathematics of the Transformer Architecture — Reading Attention, FFN, and Masks Through Equations"
description: "Writing the Transformer down as a map ℝ^{n×d} → ℝ^{n×d}: why we divide by √d_k, where O(n²d) comes from, and how a single mask separates Encoders from Decoders."
date: 2026-07-07 12:00:00
lang: en
ref: transformer-architecture-math
permalink: /en/transformer-architecture-math/
translation: /ja/transformer-architecture-math/
translation_label: "日本語"
---

Everyone has seen the Transformer block diagram. But can you say, for each arrow, *which matrix operation it is* and *why it has that shape*? In this article we write everything down in the language of matrices — from embeddings through Attention, the FFN, LayerNorm, and masking — and collect the **facts you can read directly off the equations**: why we divide by $$\sqrt{d_k}$$, where the $$O(n^2 d)$$ cost comes from, and more.

## The Big Picture — What Function Does a Transformer Compute?

Viewed mathematically, a Transformer block has a remarkably simple type signature: it is **a map that takes a sequence of $$n$$ tokens, represented as $$d$$-dimensional vectors, and returns a sequence of the same shape**.

$$
X \in \mathbb{R}^{n \times d} \;\longmapsto\; \mathrm{TransformerBlock}(X) \in \mathbb{R}^{n \times d}
$$

Each row is one token; each column is a feature dimension. Because this shape is preserved, blocks can be stacked to any depth. Inside one block there are just two components, applied in alternation.

- **Self-Attention**: mixes information *across* tokens — an operation along the sequence (row) direction.
- **FFN (Feed-Forward Network)**: transforms features *within* each token — an operation along the feature (column) direction, applying the same MLP independently to every position.

![Structure of a Transformer block: Attention mixes along the sequence direction and the FFN transforms along the feature direction, with residual connections and LayerNorm in between]({{ '/assets/img/transformer-block-en.svg' | relative_url }})

*The division of labor within a block. Information transfer along the sequence is monopolized by Attention, while the FFN is closed within each position. This contrast carries over directly to the complexity analysis below.*

Each component comes with a residual connection and LayerNorm. In other words, a Transformer is "**a layer that mixes along the sequence and a layer that transforms along the features, stacked alternately $$L$$ times**" — and Attention is the *only* path by which information moves between tokens. Keeping this division of labor in mind, every equation that follows can be sorted by asking "which direction does this operation act in?"

## The Mathematics of the Input — Embeddings and Positional Encoding

The first step, turning a sequence of token IDs into the matrix $$X$$, is the embedding. With vocabulary size $$\lvert V \rvert$$, the operation is a row lookup into the embedding matrix $$E \in \mathbb{R}^{\lvert V \rvert \times d}$$, which can be written as a product with a one-hot vector:

$$
x_i = \mathrm{onehot}(t_i)^\top E \in \mathbb{R}^d
$$

There is one problem. The Attention operation of the next section is **permutation-equivariant**: permuting the rows of the input merely permutes the rows of the output, so the model **carries no information about word order whatsoever**. "The dog chases the cat" and "the cat chases the dog" would be indistinguishable. We therefore add positional information as vectors. The sinusoidal encoding of the original paper is defined as

$$
\mathrm{PE}(\mathrm{pos},\, 2i) = \sin\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right), \qquad
\mathrm{PE}(\mathrm{pos},\, 2i+1) = \cos\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right)
$$

Each pair of dimensions carries a sine wave of a different wavelength, and the input is given as $$X + \mathrm{PE}$$. The mathematical elegance of this design is that **for any fixed offset $$k$$, $$\mathrm{PE}(\mathrm{pos}+k)$$ can be written as a linear transformation of $$\mathrm{PE}(\mathrm{pos})$$** — the angle-addition identities for sine and cosine at work — giving the model room to handle *relative* positions with linear operations. Pushing this idea further, RoPE (Rotary Position Embedding), the de facto standard in modern LLMs, encodes position not by addition but as a **rotation of Q and K**, so that the inner product $$q \cdot k$$ depends only on the difference of rotation angles, i.e., on relative position alone.

## Attention — Read It as a Weighted Average

Self-Attention can be written in three lines. From the input $$X$$, form three matrices by linear projection:

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

(where $$W_Q, W_K \in \mathbb{R}^{d \times d_k}$$ and $$W_V \in \mathbb{R}^{d \times d_v}$$). Then compute the score matrix, apply a row-wise softmax, and take the weighted sum of $$V$$:

$$
A = \mathrm{softmax}_{\mathrm{row}}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}, \qquad
\mathrm{Output} = A V \in \mathbb{R}^{n \times d_v}
$$

Entry $$(i, j)$$ of $$A$$ is "how much token $$i$$ attends to token $$j$$"; each row is non-negative and sums to 1.

![Attention as matrix multiplication: Q times K-transpose produces an n×n similarity matrix, which is normalized row-wise by softmax and used to mix V]({{ '/assets/img/attention-matmul-en.svg' | relative_url }})

*Attention as matrix arithmetic. The central $$n \times n$$ matrix holds the attention weights; row $$i$$ is token $$i$$'s probability distribution over "where to look." This is where the matrix that is quadratic in sequence length is born.*

The best way to read this equation is to pull out a single token and view the computation as a **weighted average**. The output for token $$i$$ is

$$
\mathrm{output}_i = \sum_{j} a_{ij}\, v_j, \qquad
a_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d_k})}
$$

That is, Attention is a **convex combination of Value vectors, with weights determined by inner-product similarity**. The Query is "what am I looking for right now," the Key is "the label of what I have to offer," and the Value is "the content actually delivered" — the key–value retrieval metaphor from databases maps onto the equation exactly, with hard matching replaced by soft (differentiable) inner-product similarity.

The fact that the output is always a convex combination (non-negative weights summing to 1) has a consequence: each token's new representation is chosen from *inside the convex hull spanned by the other tokens' Values*. Attention creates no new features — it **only mixes** — and nonlinear feature creation is the FFN's job. The division of labor shows up here as well.

## Why Divide by √d_k — A Variance Calculation

The $$\sqrt{d_k}$$ that gives "scaled" dot-product attention its name is not decoration; it falls out of a short calculation. Assume the components of $$q$$ and $$k$$ are independent with mean 0 and variance 1, and compute the variance of the inner product:

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i, \qquad
\mathbb{E}[q \cdot k] = 0, \qquad
\mathrm{Var}[q \cdot k] = \sum_{i=1}^{d_k} \mathrm{Var}[q_i k_i] = d_k
$$

The variance of each independent product $$q_i k_i$$ is $$1 \times 1 = 1$$, and $$d_k$$ of them are summed, so the standard deviation of the inner product grows like $$\sqrt{d_k}$$ with the dimension. With $$d_k = 64$$ scores of roughly $$\pm 8$$ are routine; with $$d_k = 512$$, roughly $$\pm 23$$. Feed these raw scores into a softmax and two things happen:

- The exponential amplifies score differences and the softmax **saturates to nearly one-hot**.
- In the saturated regime, the softmax gradient (Jacobian entries $$a_i(\delta_{ij} - a_j)$$) collapses to nearly zero, and **the learning signal vanishes**.

Dividing by $$\sqrt{d_k}$$ normalizes the inner-product variance to 1 regardless of dimension, so that at initialization the softmax starts from a suitably smooth distribution. It is fair to say that **$$\sqrt{d_k}$$ is a temperature that keeps the softmax in the region where gradients are alive**.

## Multi-Head Attention — Splitting into Subspaces

A single Attention can hold only one $$n \times n$$ attention map. If you want to track syntactic dependencies, coreference, and local continuity *simultaneously*, you want several maps. So the representation space is projected into $$h$$ subspaces, and Attention runs independently in each:

$$
\mathrm{head}_i = \mathrm{Attention}(X W_Q^{(i)},\, X W_K^{(i)},\, X W_V^{(i)}), \qquad
\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\, W_O
$$

Conventionally $$d_k = d_v = d/h$$ (e.g., $$d = 512$$, $$h = 8$$ → 64 dimensions per head). The crucial point is that because $$d$$ is split $$h$$ ways, **the parameter count and compute are essentially the same as a single head** — the design hands you $$h$$ attention maps for free. Each head has only a low-rank "viewpoint" of $$d/h$$ dimensions, but in exchange, $$h$$ different kinds of relationships can be encoded in parallel. Analyses of trained models repeatedly find heads that specialize in particular syntactic relations or positional patterns emerging naturally.

## FFN, Residual Connections, and LayerNorm

### FFN — A Two-Layer MLP per Position

The FFN applies a nonlinear transformation to each token independently, after Attention has done the mixing:

$$
\mathrm{FFN}(x) = \varphi(x W_1 + b_1)\, W_2 + b_2
$$

with $$W_1 \in \mathbb{R}^{d \times d_{\mathrm{ff}}}$$, $$W_2 \in \mathbb{R}^{d_{\mathrm{ff}} \times d}$$, and, by convention, $$d_{\mathrm{ff}} = 4d$$. The activation $$\varphi$$ is ReLU / GELU, with GLU variants (e.g., SwiGLU) dominant in recent models. This expand-then-contract shape accounts for the majority of a Transformer's parameters, as we count below. In recent interpretability work, a prominent view is that the FFN behaves as a **key–value associative memory** — the rows of $$W_1$$ act as pattern detectors and the rows of $$W_2$$ as the corresponding output patterns — and much of the model's "knowledge" is thought to be stored here (Geva et al., 2021).

### Residual Connections — Making Identity the Baseline

$$
x \leftarrow x + \mathrm{Sublayer}(x)
$$

The residual connection has two mathematical benefits. First, the backward-pass Jacobian takes the form $$I + \partial \mathrm{Sublayer} / \partial x$$, and the identity component $$I$$ lets **gradients reach deep layers without decaying**. Second, the whole network can be written as "the identity map plus a sum of small per-layer updates," so each layer need only *write a delta* rather than rebuild the representation from scratch. This view — layers reading from and writing to a residual stream — has become the basic vocabulary of modern mechanistic interpretability (Elhage et al., 2021).

### LayerNorm — Normalizing Along the Feature Direction

$$
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad
\mu = \frac{1}{d} \sum_{i} x_i, \qquad
\sigma^2 = \frac{1}{d} \sum_{i} (x_i - \mu)^2
$$

The mean and variance are taken within the $$d$$-dimensional vector of a single token (unlike BatchNorm, there is no dependence on the batch or the sequence length). There are two schools of placement. The original paper's **Post-LN** ($$\mathrm{LN}(x + \mathrm{Sublayer}(x))$$) normalizes the residual path itself, which destabilizes training in deep models and makes warmup mandatory. The modern standard is **Pre-LN** ($$x + \mathrm{Sublayer}(\mathrm{LN}(x))$$), which keeps the residual path untouched and trains stably even at depth (Xiong et al., 2020). Virtually all LLMs since GPT-2 are Pre-LN variants (with RMSNorm, which drops the mean term, now dominant).

## The Causal Mask and the Decoder

Everything up to this point is common to Encoders and Decoders. What separates the two is essentially a single point: **what mask is applied to the Attention matrix**. The Decoder imposes the "no looking at the future" constraint by replacing the upper-triangular part of the score matrix with $$-\infty$$ *before* the softmax:

$$
A = \mathrm{softmax}_{\mathrm{row}}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right), \qquad
M_{ij} =
\begin{cases}
0 & (j \le i) \\
-\infty & (j > i)
\end{cases}
$$

Since $$\exp(-\infty) = 0$$, the weights on future positions are exactly zero after the softmax. The important detail is that the masked entries are excluded from the softmax *denominator* as well — this is not the same as multiplying by zero afterwards.

![Comparison of bidirectional Attention (Encoder) and causally masked Attention (Decoder): the Encoder lets every token attend to every token, while the Decoder allows only the lower triangle]({{ '/assets/img/causal-mask-en.svg' | relative_url }})

*The difference in masks is the heart of the Encoder/Decoder distinction. BERT uses the left pattern and GPT the right pattern in every layer.*

This one mask determines the model's character.

- **Encoder (BERT-style)**: unmasked, bidirectional Attention. It builds contextual representations that see the whole input, making it strong for understanding tasks — but unusable as-is for autoregressive generation.
- **Decoder (GPT-style)**: causally masked. Because the representation at position $$i$$ never depends on the future, training can compute **next-token predictions for all positions in parallel with a single matrix operation**, and inference is accelerated by the KV cache (store past $$K, V$$ and compute only the new token's share).
- **Encoder–Decoder (original paper / T5-style)**: the Decoder gains a third attention, **Cross-Attention**, in which $$Q$$ comes from the Decoder's representations while $$K, V$$ come from the Encoder's output — an operation in which "the sequence being generated searches the input sequence."

## Counting Compute and Parameters

With the equations assembled, we can count costs. The main matrix products per block are:

| Operation | Matrix shapes | Complexity |
| --- | --- | --- |
| Q, K, V, output projection | (n×d)(d×d) ×4 | O(n d²) |
| QKᵀ (score computation) | (n×d)(d×n) | O(n² d) |
| AV (weighted sum) | (n×n)(n×d) | O(n² d) |
| FFN | (n×d)(d×4d) ×2 | O(n d²) (constant 8) |

Two important facts can be read off. First, **the only terms quadratic in the sequence length $$n$$ are Attention's score computation and weighted sum**. In the regime where $$n$$ is small relative to $$d$$, the FFN's $$O(n d^2)$$ dominates; as $$n$$ grows, $$O(n^2 d)$$ bares its teeth. It is no accident that the long-context literature (FlashAttention, linear attention, sparse attention, and so on) targets exactly this $$n^2$$ term. Note that FlashAttention does not change the asymptotic complexity at all — it speeds things up by reorganizing memory access so that the $$n \times n$$ matrix is never materialized in HBM, exploiting the fact that the bottleneck is memory bandwidth, not arithmetic (Dao et al., 2022).

Second, counting parameters shows that **the FFN holds the majority**. Per block, the Attention side has $$4d^2$$ ($$W_Q, W_K, W_V, W_O$$) and the FFN has $$8d^2$$ (two matrices of size $$d \times 4d$$), a ratio of exactly 1 : 2. The interpretability finding that "a Transformer's knowledge lives in the FFN" is consistent with the simple accounting of where the parameters sit.

```python
# Parameter count for one block (biases and LN terms omitted)
d = 4096                      # e.g., model dimension of a 7B-class model
attn = 4 * d * d              # W_Q, W_K, W_V, W_O    -> 67M
ffn  = 2 * d * (4 * d)        # W_1 (d->4d), W_2 (4d->d) -> 134M
block = attn + ffn            # ~ 201M per block
# Stack L=32 layers: ~ 6.4B. Add embeddings and you arrive at the "7B" figure.
```

## Training — What Is Being Minimized?

Finally, let us confirm in equations what this enormous map is tuned toward. The objective of a Decoder-style language model is the negative log-likelihood of the chain-rule factorization of the sequence's joint probability:

$$
P(t_1, \ldots, t_n) = \prod_{i=1}^{n} P(t_i \mid t_{<i}), \qquad
\mathcal{L} = -\sum_{i} \log \mathrm{softmax}(h_i W_{\mathrm{LM}})_{t_i}
$$

Here $$h_i$$ is the final-layer representation at position $$i$$, and $$W_{\mathrm{LM}} \in \mathbb{R}^{d \times \lvert V \rvert}$$ is the LM head (often weight-tied with the embedding $$E$$). Thanks to the causal mask, the loss at every position is computed in a single pass (teacher forcing).

An Encoder-style model (BERT), by contrast, replaces a fraction of the sequence (about 15%) with `[MASK]` and minimizes the same form of cross-entropy only at those positions (Masked Language Modeling). The objective has the same shape in both cases — cross-entropy between a softmax distribution and a one-hot target — and the difference comes down to **whether the conditioning context is one-sided or two-sided**.

To summarize: a Transformer is **a function that alternately applies "differentiable retrieval by inner-product similarity (Attention)" and "per-position associative memory (FFN)" on a residual stream, $$L$$ times**. $$\sqrt{d_k}$$ is the temperature that keeps gradients alive, the mask is the single switch separating Encoder from Decoder, and $$O(n^2 d)$$ is the destiny of the score matrix — every major design decision can be read out of these short equations.

> Whether the softmax outputs discussed here can be treated as "probabilities" — an analysis from the calibration perspective — is covered in a companion article (in Japanese): "[分類モデルの確信度 — その確率はどこから来て、どこまで信用できるのか]({{ '/ja/classification-confidence/' | relative_url }})".

## References

- Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Xiong, R., et al. (2020). On Layer Normalization in the Transformer Architecture. *ICML 2020*.
- Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- Geva, M., et al. (2021). Transformer Feed-Forward Layers Are Key-Value Memories. *EMNLP 2021*.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits.
- Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*.
