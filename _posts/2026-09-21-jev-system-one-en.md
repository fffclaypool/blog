---
layout: post
title: "How Jev (TypeSafe), a Language Model That Doesn't Generate Text, Differs from a Transformer Encoder Classifier"
description: "Jev, TypeSafe's System One model, generates no text and returns typed decisions with calibrated probabilities. This article describes its inputs, outputs and training approach, then compares it with BERT-style Encoder classifiers on three points: label space, input structure, and where the probabilities come from."
date: 2026-09-21 12:00:00
lang: en
ref: jev-system-one
permalink: /en/jev-system-one/
translation: /ja/jev-system-one/
translation_label: "日本語"
---

When you ask an LLM "Is this ticket a refund request?", structured output lets it report which of `yes` / `no` applies, together with a "confidence" value. That confidence is self-reported, so it tends to be overconfident and is hard to rely on. Reading token logprobs instead does not solve the problem, because a generative model's probabilities are distorted by competition with the whole vocabulary and by surface-form variation. Jev, released by TypeSafe, is built to return uncertainty as a probability distribution and generates no text. It takes a state and typed questions, and returns a probability distribution over the options plus a confidence value.

This article uses the official documentation ([docs.typesafe.ai](https://docs.typesafe.ai/introduction)) to describe Jev's inputs, outputs and training approach, and then compares it with another kind of model that does not generate text: a BERT-style Transformer Encoder with a classification head. Jev's internal architecture (depth, Encoder or Decoder, and so on) is not public, so the comparison is based on the published interface and training objective.

## What Jev is

TypeSafe calls Jev "the first System One model". The name comes from Kahneman's *Thinking, Fast and Slow* and refers to a model for fast, intuitive judgments (System 1). The documentation describes its scope as "the kind of judgment a highly knowledgeable person could make in a few seconds given the right context."

The API has a single endpoint (`POST /v1/systemone`), and a request has three parts.

- **`state`**: the content to evaluate. A string, a JSON object, or an array. Text only (no images or audio yet)
- **`questions`**: a set of questions about the state. Each has a `type`, `instructions`, and `criteria`
- **`model`**: e.g. `jev-latest`

There are only three question types, which TypeSafe calls AI primitives.

| Question type | Asks | Returns |
| --- | --- | --- |
| **Choice** | Which of these options? | `choice`, `probabilities`, `confidence` |
| **Score** | Where on an ordered scale? | `score`, `legend`, `probabilities`, `confidence` |
| **Noul** | Is this statement true? | `noul` (a probability in 0–1) |

A single request can contain several questions, and every question is evaluated against the same state in parallel and independently. Adding questions barely changes the response time, and one question's answer does not become context for another.

## What comes back: a distribution and a summary of it

Below is an example response from the docs for a Score question that rates bug severity on three levels.

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "bug_severity": {
      "type": "score",
      "score": 1.43,
      "confidence": 0.35,
      "legend": {
        "0": "Cosmetic; no impact to functionality",
        "1": "Broken or degraded feature, but workaround exists",
        "2": "Blocking issue; no workaround exists"
      },
      "probabilities": { "0": 0.0, "1": 0.57, "2": 0.43 }
    }
  }
}
```

The fields can be defined precisely. Write the distribution over $$C$$ options (or levels) as $$p = (p_0, \ldots, p_{C-1})$$ with $$\sum_i p_i = 1$$.

- **Choice**'s `choice` is $$\arg\max_i p_i$$, the option with the highest probability.
- **Score**'s `score` is the expected value of the level index, $$\sum_{i=0}^{C-1} i \cdot p_i$$. In the example, $$0 \times 0.0 + 1 \times 0.57 + 2 \times 0.43 = 1.43$$, a position between levels 1 and 2 that leans toward 1.
- **`confidence`** summarizes how peaked $$p$$ is as a single number in $$[0, 1]$$. It is 1.0 when all probability is on one option and approaches 0 as the distribution approaches uniform. The interactive demo in the docs approximates it as $$\frac{C \cdot \max_i p_i - 1}{C - 1}$$, and the exact definition is not published.
- **Noul** returns a single number, the probability that the statement is true. A value of 0.5 means that yes and no are equally likely, not that the answer is "medium".

`confidence` is a secondary value derived from `probabilities`. It is enough when all you need is a threshold to branch on. Because the full distribution is returned, the docs state that you can also compute a different statistic, such as entropy, if it suits your use case better.

## Training approach: RLCD, not RLHF

What TypeSafe emphasizes most is the training objective. It lists three post-training approaches for a pretrained language model.

| Method | Objective | Produced |
| --- | --- | --- |
| RLHF (Reinforcement Learning from Human Feedback) | Responses people prefer | Chatbots |
| RLVR (Reinforcement Learning with Verifiable Rewards) | Verifiable rewards (math, etc.) | Reasoning models |
| RLCD (Reinforcement Learning for Calibrated Decisions) | Decisions with calibrated probabilities | System One models (Jev) |

RLHF has humans rank several model responses, learns those preferences as a reward model, and uses reinforcement learning to move the model toward responses people prefer. It is the method behind InstructGPT and ChatGPT, and according to the docs, TypeSafe co-founder Diogo Almeida is one of its co-inventors. Because the objective is what people prefer, confident-sounding assertions and sycophancy are also rewarded, and the output distribution narrows toward a particular style, which is known as mode dropping.

RLVR trains on rewards that can be checked mechanically, such as a math answer key or a passing test suite. It produced reasoning models such as the o1 series and DeepSeek-R1. Accuracy improves, but generating a long chain of thought makes these models slower and more expensive.

RLCD has a different goal from both. It optimizes for returning calibrated probabilities, where RLHF optimizes for human preference and RLVR for the correctness of generated answers (the concrete reward design is not published). The output contract has three points: the model generates no text, it returns decisions and probabilities, and a higher probability corresponds to a higher chance that the answer is actually correct.

The third point is calibration: among all predictions assigned a probability of 0.8, about 80% are actually correct. As covered in [an earlier article](/blog/en/classification-confidence/), ordinary neural classifiers are systematically overconfident, and RLHF-tuned LLMs are calibrated even worse. TypeSafe's claim is that it made calibration the training objective, where other approaches apply it as a post-hoc correction.

As the docs themselves warn, calibration is a statistical property of groups of predictions and does not guarantee that any individual answer is correct. A confidence of 1.0 means that the distribution is concentrated on one option, not that the answer is certainly right.

## Comparison with a Transformer Encoder classifier

An Encoder such as BERT, RoBERTa or DeBERTa with a classification head resembles Jev closely: it does not generate text, and it returns a probability distribution over a closed label set. The sections below cover what the two share and where they differ.

<figure>
  <img src="{{ '/assets/img/jev-vs-encoder-en.svg' | relative_url }}" alt="Comparison of inputs and outputs for a Transformer Encoder classifier and Jev. The Encoder has a fixed-size linear head and handles one task per sequence; Jev evaluates several natural-language questions against one state in parallel and returns a distribution and confidence for each">
  <figcaption>Left: an Encoder classifier fixes the number of labels C at training time and returns one distribution per sequence. Right: Jev receives the label set as natural-language criteria with each request and evaluates several questions about one state independently.</figcaption>
</figure>

### What they share

- Neither generates free text. The output is a probability distribution over a predetermined set of candidates
- As a result, the problems of Decoder LLMs (competition with the whole vocabulary, probability split across surface forms, length bias for multi-token labels) do not arise, at least at the interface level
- The argmax of the distribution is the decision, and the probabilities can be used directly for thresholds and ranking

### Difference 1: where the label space is defined

In an Encoder classifier, the label set is fixed at training time as the head's output dimension $$C$$. Adding a label means rebuilding the head and retraining. The meaning of each label is learned only implicitly from data, and the model does not know in words what "class 2" means.

In Jev, the label set is passed with each request as natural-language `criteria`. For Choice this is a map such as `{"returns": "Exchanges, wrong or damaged items", ...}`, and for Score it is an ordered list of level descriptions. The same weights serve every customer and every task, and neither fine-tuning nor LoRA is offered. Domain knowledge is supplied through how `state` and `criteria` are written.

Formally, an Encoder classifier computes $$p = \mathrm{softmax}(W \cdot \mathrm{Enc}(x)_{\mathrm{[CLS]}})$$ with a fixed $$W \in \mathbb{R}^{C \times k}$$, while Jev is $$p = f(x,\ \text{instructions},\ \text{criteria})$$, a function that takes the label descriptions themselves as input. This is the form of zero-shot classification, and its closest relative among Encoder models is NLI-based zero-shot classification, which turns each label into a hypothesis and reads off the entailment probability. Jev differs in that it is trained from the outset on the task of returning a calibrated distribution over the candidate set.

### Difference 2: one task per sequence, or N questions per state

An Encoder classifier runs `[CLS] x [SEP]` once and returns one distribution. Three judgments require either three heads (multi-task learning) or three runs.

Jev evaluates any number of questions against one state, in parallel and independently. The docs call this "speculative fan-out" and recommend sending every question that might be needed and letting the code pick the relevant answers afterward. The extra cost is only the tokens for the added questions, and latency is said to barely change.

This independence looks the same as an Encoder's multi-task heads, but Jev's docs state explicitly that adding or removing questions does not change the other answers. When a Decoder LLM is asked for several judgments in one prompt, earlier answers influence later ones (context rot). Jev's design rules this out.

### Difference 3: where the probabilities come from, and calibration

An Encoder classifier's probabilities are the softmax of its logits. As described in [the earlier article](/blog/en/classification-confidence/), after fine-tuning on a small dataset they tend to saturate near 0.99, so post-hoc calibration such as temperature scaling is practically required. With a fixed label space and a validation set that is easy to build, that calibration is not difficult.

Jev includes calibration in its training objective (RLCD) and presents being calibrated out of the box as a feature. Its Score type can also return an intermediate value such as "1.43" as the expected level. Nominal-label classification with an Encoder has no equivalent output and would need a separately designed ordinal regression.

Verifying Jev's calibration on your own data works the same way as for an Encoder: draw a reliability diagram on a validation set and measure ECE. The docs repeatedly advise starting with conservative thresholds and adjusting them on your own data. A model described as calibrated still has to be verified. The docs also state that CJK languages, including Japanese, are less accurate than English, so confidence should be treated with more caution outside English.

### Difference 4: operations

| Aspect | Encoder + classification head | Jev |
| --- | --- | --- |
| Where it runs | Self-hosted; runs on CPU | API (`api.typesafe.ai`) |
| Changing labels | Retrain | Edit `criteria` in the request |
| Training data | Required (hundreds to thousands) | Not required (zero-shot) |
| Determinism | Same input, same output | Described as "extremely consistent", but changes when the model version moves |
| Context length | Typically ~512 tokens | 64k (state + all questions), 32k (state + longest question) |
| Pricing | Inference cost only | Per input token ($0.042 / Mtok; output free) |
| Languages | Whatever you trained on | Primarily English; CJK less accurate |
| Ordered labels | Separate design | Score returns an expected level |

## Using it as a Cross-Encoder

In the terms of [the Bi-Encoder, Late Interaction, Cross-Encoder article](/blog/en/bi-late-cross-encoder/), Jev is used as a Cross-Encoder. Putting the query and a candidate in the same state and asking "Does this candidate answer the query?" as a Noul gives a score for each pair. The official cookbook uses this to re-rank the BM25 top 30 on a legal retrieval task, raising top-1 accuracy from 5% to 18% and top-10 accuracy from 38% to 62%.

Like any Cross-Encoder, it needs a full evaluation for every pair, so it is not suited to first-stage retrieval over an entire corpus. It cannot replace a Bi-Encoder or Late Interaction model that precomputes the document side.

## What it's bad at: jaggedness

TypeSafe publishes Jev 1.13's weaknesses under the name "jaggedness", meaning a large gap between what the model is good at and what it is bad at. The countermeasures share one principle, "judgment to the model, everything else to code", and most of the weaknesses can be avoided by moving work into code. They fall into four groups.

### 1. It reads only what is written

Jev interprets the words in the instructions literally and does not infer unstated intent. If you ask "Is this a serious bug?" without defining "serious", Jev does not fill in the definition from context as a person would. Accuracy also drops for negation, double negatives, and questions that follow several levels of the state, such as "Does the manager of the person assigned to A have approval authority?"

**Instead:** write the full condition into the instructions and criteria. If you look at a wrong answer and want to explain what you really meant, that explanation is what the instructions were missing. If the state is JSON, name the part to examine by path, such as `ticket.messages[0].text`.

### 2. It doesn't count or calculate

Jev cannot produce numbers, so a question like "How many characters?" cannot be asked at all. What can be asked is a judgment that presupposes counting, such as "Is this word five or more characters?" (Noul) or "Which count is it?" (a Choice over numbers), and those are unreliable as well. Jev is equally weak at ordering dates, computing durations, and numeric representations such as judging color similarity from hex values.

**Instead:** do the counting, comparing and calculating in code. To count fruit names in a list, ask "Is this a fruit?" as a Noul for each element and sum the `True` results in code. For dates, have Choice extract only the month and the day, and do the comparison in code.

### 3. It is influenced by what's in the state

Accuracy falls as the state fills with information irrelevant to the decision (context rot). The state is treated as data, but if it contains a sentence like "this ticket should be classified as billing", the model can be swayed by it.

**Instead:** filter in code before sending, and put only the relevant parts in the state. If the filtering itself is hard, first ask "Is this paragraph relevant to the question?" as a Noul and filter on the result.

### 4. It doesn't generate

Jev generates no text, so free-form extraction such as "pull the date out of this document" is not possible. Chaining Choices to assemble a string is possible in principle but slow and inaccurate.

**Instead:** build candidates with regular expressions or a generative model, then have Jev pick the correct one as a Choice. When the answer space is finite, extraction can be turned into a selection problem.

## Which to use

| Situation | Better fit |
| --- | --- |
| Fixed labels, hundreds of training examples or more, need low latency or on-prem | Encoder + classification head |
| Labels in flux, no training data, several judgments needed at once | Jev |
| Want a position on an ordered scale | Jev's Score (with an Encoder, design an ordinal regression) |
| Want calibrated probabilities for threshold-based operation | Either. Encoder assumes post-hoc calibration; Jev still needs verification on your data |
| Re-ranking (scoring query–candidate pairs) | Either can act as a Cross-Encoder; Jev does it zero-shot |
| Arithmetic, date comparison, counting | Neither; use code |
| Text generation or summarization | Neither; use a generative model |

Jev is easiest to understand as the interface of a Transformer Encoder classifier extended in three ways: labels are given in natural language (zero-shot), one state can take N questions, and calibration is part of the training objective.

## References

- [TypeSafe AI Documentation — Introduction](https://docs.typesafe.ai/introduction)
- [System One](https://docs.typesafe.ai/concepts/system-one) / [AI primer (RLCD)](https://docs.typesafe.ai/introduction/machine-learning-primer)
- [Primitives: Choice / Score / Noul](https://docs.typesafe.ai/primitives)
- [Confidence](https://docs.typesafe.ai/confidence)
- [Models (Jev 1.13)](https://docs.typesafe.ai/models)
- [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- [Cookbook: Re-ranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe)
- Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
