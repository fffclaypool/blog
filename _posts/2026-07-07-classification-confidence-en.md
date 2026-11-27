---
layout: post
title: "Confidence Scores in Classification Models: Where the Probabilities Come From and How Far You Can Trust Them"
description: "Can the '94%' returned by a softmax be treated as a probability? This article uses calibration to answer that, and examines how confidence differs between Encoders (BERT-style) and Decoders (GPT-style LLMs)."
date: 2026-07-07 13:00:00
lang: en
ref: classification-confidence
permalink: /en/classification-confidence/
translation: /ja/classification-confidence/
translation_label: "日本語"
---

The confidence scores that classification models output are often treated as probabilities without further checks. Whether they deserve that name has to be verified. This article explains how logits are turned into a softmax output and what calibration measures, and then examines how the origin and reliability of confidence differ between Encoders (BERT-style) and Decoders (GPT-style LLMs), using equations.

## Where Confidence Comes From: Logits and Softmax

The final layer of a neural classifier outputs one real-valued score per class. These scores are called logits and written $$z \in \mathbb{R}^C$$, where $$C$$ is the number of classes. Logits are arbitrary real numbers and have no probabilistic meaning on their own. The softmax function converts them into the form of a probability distribution.

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}
$$

$$T$$ is a temperature parameter, normally set to $$T = 1$$ during training. Raising $$T$$ flattens the distribution (less confident), and lowering it sharpens the distribution (more confident).

<figure>
  <img src="{{ '/assets/img/logits-softmax-en.svg' | relative_url }}" alt="The logit-to-softmax transformation: the ranking and gaps between scores are preserved, but frequency-level correctness is not guaranteed">
  <figcaption>Softmax projects relative scores onto the probability simplex. Adding a constant to all of \(z\) leaves \(p\) unchanged (shift invariance), so \(p\) is determined only by the differences between logits.</figcaption>
</figure>

Softmax does two things: it maps every score to a positive number with the exponential, and it normalizes the results to sum to 1. The output is a set of non-negative numbers that sum to 1, so it meets the formal requirements of a probability distribution. That is a matter of form, though, and it does not give the numbers a frequency interpretation automatically.

There is still a reason it is called a probability. When a model is trained with cross-entropy loss (negative log-likelihood), its output is in theory an estimator of the conditional distribution $$P(y \mid x)$$. With infinite data, sufficient capacity and perfect optimization, the softmax output converges to the true conditional probability, so the probability interpretation does have theoretical grounding. Real training meets none of these conditions. Data is finite, the model is overparameterized, early stopping and regularization are applied, and the training distribution differs from the deployment distribution. Together these move the output away from the true probability.

## The Condition for Being a "Probability": Calibration

Calibration is the concept that measures whether confidence can be trusted. A model is called well-calibrated if, among all predictions made with confidence 0.9, roughly 90% are actually correct.

$$
P(\hat{y} = y \mid \mathrm{conf} = p) = p \quad \text{for all } p \in [0, 1]
$$

The standard visualization is the reliability diagram. Confidence values are binned, and each bin's mean confidence is plotted against its actual accuracy. Points on the diagonal are ideal, and points below it mean the model is overconfident. The weighted average of the per-bin gaps is the ECE (Expected Calibration Error), the most common calibration metric.

<figure>
  <img src="{{ '/assets/img/reliability-diagram-en.svg' | relative_url }}" alt="Reliability diagram: an overconfident model's curve sinks below the diagonal">
  <figcaption>A reliability diagram. In the typical overconfidence pattern, predictions made with confidence 0.95 turn out to be only 0.82 accurate.</figcaption>
</figure>

Modern deep neural networks are systematically overconfident (Guo et al., 2017). That work showed that calibration degrades as models grow larger and deeper. It also showed that temperature scaling, which learns a single temperature $$T$$ on a validation set, is an effective correction.

Calibration is independent of accuracy. A model with 99% accuracy that always reports 100% confidence is poorly calibrated, and a model with 60% accuracy whose confidence matches its actual accuracy is well calibrated. Being right and reporting confidence accurately are separate properties.

## Confidence in Encoders (BERT-style)

Classification with Encoder models such as BERT, RoBERTa and DeBERTa has a simple structure. The input is read bidirectionally, and a classification head (a linear layer) sits on top of the `[CLS]` token or a pooled representation. The output dimension equals the number of classes $$C$$, and a softmax is applied. This structure has the following properties.

- The label space is closed. The output is always exactly $$C$$ logits, and the probability is consistently defined as a distribution over $$C$$ candidates.
- The head is optimized for the task alone. It is trained only on that task's cross-entropy, so the confidence directly reflects the training distribution.
- The output is stable. The same input always yields the same probabilities, and issues such as prompt phrasing or output-format drift do not arise.

The structure has two known weaknesses. First, fine-tuned models tend to be overconfident. Tuning a pre-trained model on a small dataset for a few epochs lets it nearly memorize the training data, and the gaps between logits become extreme. The result is confidence saturation, with values stuck near 0.99 (Desai & Durrett, 2020).

Second, the model cannot say "I don't know" for out-of-distribution (OOD) inputs. Softmax normalizes to a sum of 1, so even an input far from the training distribution, such as a legal document given to a sentiment classifier, often receives high confidence for one of the classes. The structure has no way to output "none of the above".

Encoder confidence is, however, easy to correct. The label space is fixed and validation sets are easy to build, so post-hoc calibration methods such as temperature scaling, Platt scaling and isotonic regression apply directly.

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

If the learned temperature satisfies $$T > 1$$, the model was overconfident and the correction flattens the distribution. $$T$$ does not change the ranking of predictions, so it calibrates confidence without affecting accuracy.

## Confidence in Decoders (GPT-style LLMs)

Classification with a GPT-style Decoder works very differently. A Decoder has no classification head and only performs next-token prediction, so classification is recast as generating the label as a continuation of the prompt. There are two main ways to extract confidence.

### 1. Reading token probabilities (logprobs)

This method reads the probability of the label tokens from the distribution over the next token after the prompt. In principle it is the closest to an Encoder's softmax, but it has several problems of its own.

- The label competes with the entire vocabulary. An Encoder's probability is a distribution over $$C$$ classes, while a Decoder's is a distribution over the entire vocabulary $$V$$ (tens to hundreds of thousands of tokens). Some of the probability mass goes to unrelated continuations such as "Yes" or "The answer is". Extracting only the label candidates and renormalizing is practically required.
- Surface forms compete. One label meaning is split across several surface forms, such as "positive", "Positive" and "pos", and its probability mass is split with it (Holtzman et al., 2021). Concepts with more surface variants are at a disadvantage.
- Labels can span multiple tokens. If a label cannot be expressed in one token, the probability of the whole sequence (the product of per-token conditional probabilities) has to be computed, which lowers the probability of longer labels.
- The output is sensitive to the prompt. Reordering few-shot examples or rewording the prompt can change the probability distribution substantially. Encoders do not have this instability.

### 2. Verbalized confidence

This method instructs the model to "state your answer and your confidence from 0–100%" and has it say a number. It is widely used because it works with APIs that expose no logprobs. The number is not a readout of an internal probability computation, though. It is generated text that reports confidence. Studies repeatedly find a strong overconfidence bias, with values clustering around 80–95% whether the answer is right or wrong (Xiong et al., 2023). Human-written text rarely says "I am 52% confident", so a skewed distribution of reported values is a natural consequence of the training data.

### RLHF breaks calibration

Decoders show one more specific phenomenon. Base models directly after pre-training are well calibrated, and RLHF degrades that calibration. The GPT-4 technical report includes a comparison in which the pre-trained model's answer-choice probabilities lie almost on the diagonal, while calibration is much worse after RLHF. The usual interpretation is that optimizing for the assertive responses people prefer sharpens the probability distribution. The more an LLM has been tuned for chat, the less its confidence can be taken at face value.

## How Far Can You Trust It?

Confidence is useful when it is calibrated and used as ranking information, and it is risky when the raw number is used directly in decisions as a probability.

The following uses are reliable.

- Ranking and triage. Even when calibration is poor, the ordering of confidence values is usually preserved fairly well. Selective prediction, where the lowest-confidence cases are sent to human review, works even with distorted raw values.
- Thresholds combined with post-hoc calibration. Check the reliability diagram and ECE on a validation set, correct with temperature scaling or a similar method, and then set thresholds such as "process automatically above 0.9, send to a human below". A threshold has meaning only after calibration.
- Relative comparison within one distribution. Comparing samples under the same model and the same input distribution is far more reliable than comparing across a distribution shift.

The following uses should be avoided.

- Using uncalibrated values in risk calculations. Reading "confidence 0.94" as "6% error rate" has no basis unless calibration has been verified.
- Using confidence in place of OOD detection. High confidence does not mean the input is in-distribution. Models are often wrong with high confidence on out-of-distribution inputs.
- Comparing confidence across models or prompts. With LLMs in particular, changing a single word in the prompt changes the distribution. Comparing absolute confidence values between different setups means very little.
- Taking verbalized confidence at face value. Treat it as an auxiliary signal, and where possible combine it with logprob-based measures or agreement rates across repeated samples (self-consistency).

## Encoder / Decoder Summary

| Aspect | Encoder (BERT-style + classification head) | Decoder (GPT-style LLM) |
| --- | --- | --- |
| Source of the probability | Classification-head logits → softmax over C classes | Next-token prediction → read labels from the vocabulary-wide distribution, or verbalize confidence |
| Label space | Closed (fixed number of classes) | Open (competes with the whole vocabulary; renormalization needed) |
| Output stability | Deterministic and stable | Sensitive to prompts, few-shot order, surface forms |
| Typical distortions | Overconfidence and saturation after small-data fine-tuning | Surface form competition, length bias, RLHF-induced miscalibration |
| Ease of calibration | High; temperature scaling etc. apply directly | Requires work: renormalize over candidates + temperature, self-consistency |
| Best suited for | Fixed labels, training data available, threshold-based operation | Fluid labels, few examples; keep confidence as an auxiliary signal |

A softmax output is a relative score in the form of a probability, and whether it can be treated as a probability can be judged only after calibration has been verified. With Encoders the verification and correction are easy to do. With Decoders, extracting the probability itself has several pitfalls. In both cases, the first step is to draw a reliability diagram on your own data.

## References

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
- Holtzman, A., West, P., et al. (2021). Surface Form Competition: Why the Highest Probability Answer Isn't Always Right. *EMNLP 2021*.
- OpenAI (2023). GPT-4 Technical Report.
- Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know.
- Xiong, M., et al. (2023). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs.
- Desai, S., & Durrett, G. (2020). Calibration of Pre-trained Transformers. *EMNLP 2020*.
