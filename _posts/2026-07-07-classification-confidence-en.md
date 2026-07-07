---
layout: post
title: "Confidence Scores in Classification Models — Where Do the Probabilities Come From, and How Far Can You Trust Them?"
description: "Is the '94%' returned by a softmax really a probability? Through the lens of calibration, we examine how the nature of confidence differs between Encoders (BERT-style) and Decoders (GPT-style LLMs)."
date: 2026-07-07 13:00:00
lang: en
ref: classification-confidence
permalink: /en/classification-confidence/
translation: /ja/classification-confidence/
translation_label: "日本語"
---

The confidence scores that classification models output are often treated as probabilities, as-is. But do those numbers really deserve the name? In this article we walk through the mechanics of the logit-to-softmax transformation and the evaluation axis of **calibration**, and then examine **how the origin and reliability of confidence differ between Encoders (BERT-style) and Decoders (GPT-style LLMs)**, with equations in hand.

## Where Confidence Comes From — Logits and Softmax

The final layer of a neural classifier outputs a real-valued score per class — the **logits** $$z \in \mathbb{R}^C$$ (where $$C$$ is the number of classes). Logits are arbitrary real numbers and carry no probabilistic meaning on their own. The softmax function is what converts them into something probability-shaped:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}
$$

$$T$$ is a temperature parameter, normally $$T = 1$$ during training. Raising $$T$$ flattens the distribution (less confident); lowering it sharpens the distribution (more confident).

![The logit-to-softmax transformation: the ranking and gaps between scores are preserved, but frequency-level correctness is not guaranteed]({{ '/assets/img/logits-softmax-en.svg' | relative_url }})

*Softmax is merely a projection of relative scores onto the probability simplex. Adding a constant to all of $$z$$ leaves $$p$$ unchanged (shift invariance): $$p$$ is determined solely by the differences between logits.*

Softmax does essentially two things: **(1) it maps every score to a positive number via the exponential, and (2) it normalizes them to sum to 1.** The output is "a set of non-negative numbers summing to 1," so it satisfies the formal requirements of a probability distribution — but that is a mathematical formality, and frequency-level meaning does not come attached automatically.

So why is it called a probability at all? When trained with cross-entropy loss (negative log-likelihood), the model's output is, in theory, **an estimator of the conditional distribution $$P(y \mid x)$$**. With infinite data, sufficient capacity, and perfect optimization, the softmax output converges to the true conditional probability. In other words, the probability interpretation *does* have theoretical grounding. The problem is that real training satisfies none of these ideal conditions: finite data, excessive parameters, early stopping and regularization, and the mismatch between the training distribution and the deployment distribution. All of these push the output away from the true probability.

## The Condition for Being a "Probability" — Calibration

The concept that measures whether confidence can be trusted is **calibration**. The definition is simple: a model is *well-calibrated* if, among all predictions made with confidence 0.9, roughly 90% are actually correct.

$$
P(\hat{y} = y \mid \mathrm{conf} = p) = p \quad \text{for all } p \in [0, 1]
$$

The standard visualization is the **reliability diagram**: bin the confidence values, then plot each bin's mean confidence against its actual accuracy. Landing on the diagonal is ideal; dipping below it means the model is **overconfident**. The weighted average of the per-bin gaps is the **ECE (Expected Calibration Error)**, the most common calibration metric.

![Reliability diagram: an overconfident model's curve sinks below the diagonal]({{ '/assets/img/reliability-diagram-en.svg' | relative_url }})

*A reliability diagram. The typical overconfidence pattern: the set of predictions made with "confidence 0.95" turns out to be only 0.82 accurate.*

An important empirical fact: **modern deep neural networks are systematically overconfident** (Guo et al., 2017). That work showed that calibration degrades as models grow larger and deeper — and that **temperature scaling**, which learns just a single temperature $$T$$ on a validation set, is a surprisingly effective correction.

Note that calibration is a property **independent of accuracy**. A model with 99% accuracy that always reports "100% confidence" is poorly calibrated; a model with 60% accuracy whose confidence is honest is well calibrated. "Being right" and "reporting your confidence honestly" are separate axes.

## Confidence in Encoders (BERT-style)

Classification with Encoder models — BERT, RoBERTa, DeBERTa — has a clean structure. The input is read bidirectionally, and a **classification head (a linear layer)** sits on top of the `[CLS]` token (or a pooled representation). The output dimension equals the number of classes $$C$$, and a softmax is applied.

- **The label space is closed.** The output is always exactly $$C$$ logits, and the probability is consistently defined as a distribution over $$C$$ candidates.
- **It is optimized for the task alone.** The classification head is trained solely on that task's cross-entropy, so the confidence directly reflects the training distribution.
- **The output is stable to interpret.** The same input deterministically yields the same probabilities. Problems like prompt phrasing or output-format drift simply do not exist.

Because the structure is so straightforward, the weaknesses are also well characterized. First, **fine-tuned models tend to be overconfident**. Adjusting a pre-trained model on a small dataset for a few epochs lets it nearly memorize the training data, and the gaps between logits blow up. The result is "confidence saturation," with values pinned near 0.99 (Desai & Durrett, 2020).

Second, **the model cannot stay silent on out-of-distribution (OOD) inputs**. Softmax forces normalization to a sum of 1, so even inputs far from the training distribution — say, a legal document fed to a sentiment classifier — often receive high confidence for *some* class. Structurally, there is no escape hatch labeled "none of the above."

Fortunately, Encoder confidence is **easy to correct**. With a fixed label space and easily constructed validation sets, post-hoc calibration methods — temperature scaling, Platt scaling, isotonic regression — apply directly.

```python
# Minimal temperature scaling (PyTorch)
# Optimize a single scalar T on validation logits and labels
import torch

def fit_temperature(logits_val, labels_val):
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    nll = torch.nn.CrossEntropyLoss()

    def step():
        opt.zero_grad()
        loss = nll(logits_val / T, labels_val)
        loss.backward()
        return loss

    opt.step(step)
    return T.item()   # at inference: softmax(logits / T)
```

If the learned $$T > 1$$, the correction reads "this model was overconfident, so flatten the distribution." Since $$T$$ never changes the ranking of predictions, it calibrates confidence without affecting accuracy at all.

## Confidence in Decoders (GPT-style LLMs)

When classification is done with a GPT-style Decoder, the situation changes fundamentally. A Decoder has no "classification head" — all it has is **next-token prediction**. Classification is translated into the task of "generating the label as a continuation of the prompt." There are two main families of methods for extracting confidence.

### (1) Reading token probabilities (logprobs)

Read the probability of the label tokens from the distribution over the next token following the prompt. In principle this is closest to the Encoder's softmax, but it comes with its own set of problems.

- **Competition with the entire vocabulary.** An Encoder's probability is a distribution over $$C$$ classes; a Decoder's probability is a distribution over the entire vocabulary $$V$$ (tens to hundreds of thousands of tokens). Probability mass for label tokens is eaten by unrelated continuations like "Yes" or "The answer is." Extracting only the label candidates and renormalizing is practically mandatory.
- **Surface form competition.** A single label meaning splits across multiple surface forms — "positive," "Positive," "pos" — and the probability mass splits with it (Holtzman et al., 2021). Concepts with more surface variants are systematically disadvantaged.
- **The multi-token problem.** If a label cannot be expressed in one token, you must compute the probability of the whole sequence (a product of per-token conditional probabilities), which introduces a length bias against longer labels.
- **Prompt sensitivity.** Reordering few-shot examples or rewording the prompt can move the probability distribution substantially — an instability that Encoders simply do not have.

### (2) Verbalized confidence

Instruct the model to "state your answer and your confidence from 0–100%," and have it *say* a number. Popular because it works even with APIs that expose no logprobs — but note that this number is **not a readout of internal probability computation; it is generated text that reports confidence**. Empirically, a strong overconfidence bias is repeatedly observed, with values clustering around 80–95% regardless of whether the answer is right or wrong (Xiong et al., 2023). Given how rarely human-written text says "I am 52% confident," a skewed distribution of reports is a natural consequence of the training data.

### RLHF breaks calibration

There is one more phenomenon specific to Decoders. **Base models straight out of pre-training are surprisingly well calibrated — and RLHF degrades that calibration.** The GPT-4 technical report shows a comparison in which the pre-trained model's answer-choice probabilities lie almost on the diagonal, while calibration collapses after RLHF. The interpretation is that optimizing for the crisp, assertive responses humans prefer pushes the probability distribution toward sharpness. In short: **the more an LLM has been tuned for chat, the less its confidence can be taken at face value.**

## How Far Can You Trust It?

The bottom line: confidence is **very useful when calibrated and used as ranking information**, and **dangerous when the raw number is piped directly into decisions as a probability**. Concretely:

Uses you can trust:

- **Ranking and triage.** Even when calibration is broken, the *ordering* of confidence tends to be relatively preserved. Selective prediction — "route the lowest-confidence cases to human review" — works even when the raw values are distorted.
- **Thresholding, paired with post-hoc calibration.** Check the reliability diagram and ECE on a validation set, correct with temperature scaling or similar, and only then draw thresholds like "auto-process above 0.9, route to a human below." The threshold's meaning exists only after calibration.
- **Relative comparison within the same distribution.** Comparing samples under the same model and the same input distribution is far more reliable than comparisons across distribution shifts.

Uses you should not trust:

- **Feeding uncalibrated values into risk calculations.** Reading "confidence 0.94" as "6% error rate" has no basis unless calibration has been verified.
- **Substituting for OOD detection.** High confidence does not mean "this input is in-distribution." Models are frequently at their most confidently wrong on out-of-distribution inputs.
- **Comparing confidence across models or prompts.** Especially with LLMs, changing a single word in the prompt moves the distribution. Comparing absolute confidence values across different setups is close to meaningless.
- **Taking verbalized confidence at face value.** Treat verbalized confidence as an auxiliary signal; where possible, combine it with logprob-based measures or agreement rates across repeated samples (self-consistency).

## Encoder / Decoder Summary

| Aspect | Encoder (BERT-style + classification head) | Decoder (GPT-style LLM) |
| --- | --- | --- |
| Source of the probability | Classification-head logits → softmax over C classes | Next-token prediction → read labels from the vocabulary-wide distribution, or verbalize confidence |
| Label space | Closed (fixed number of classes) | Open (competes with the whole vocabulary; renormalization needed) |
| Output stability | Deterministic and stable | Sensitive to prompts, few-shot order, surface forms |
| Typical distortions | Overconfidence and saturation after small-data fine-tuning | Surface form competition, length bias, RLHF-induced miscalibration |
| Ease of calibration | High — temperature scaling etc. apply directly | Requires work — renormalize over candidates + temperature, self-consistency |
| Best suited for | Fixed labels, training data available, threshold-based operation | Fluid labels, few examples; keep confidence as an auxiliary signal |

To compress it into one sentence: **a softmax output is a "relative score shaped like a probability," and the right to treat it as a probability is earned only by verifying calibration.** With Encoders that verification and correction is straightforward; with Decoders, extracting the probability at all is riddled with pitfalls. In either case, drawing a single reliability diagram on your own data is the shortest ritual that makes the number worthy of trust.

> The structures assumed here — Attention, the classification head, the LM head — are derived from the equations up in the companion article "[The Mathematics of the Transformer Architecture — Reading Attention, FFN, and Masks Through Equations]({{ '/en/transformer-architecture-math/' | relative_url }})".

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
- Holtzman, A., West, P., et al. (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. *EMNLP 2021*.
- OpenAI (2023). GPT-4 Technical Report.
- Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know.
- Xiong, M., et al. (2023). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs.
- Desai, S., & Durrett, G. (2020). Calibration of Pre-trained Transformers. *EMNLP 2020*.
