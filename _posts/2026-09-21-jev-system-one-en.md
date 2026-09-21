---
layout: post
title: "Jev (TypeSafe) — How a Language Model That Doesn't Generate Text Differs from a Transformer Encoder Classifier"
description: "TypeSafe's System One model 'Jev' generates no text; it returns typed decisions and calibrated probabilities. We lay out its inputs, outputs and training approach, then compare it with BERT-style Encoder classifiers along three axes: label space, input structure, and where the probabilities come from."
date: 2026-09-21 12:00:00
lang: en
ref: jev-system-one
permalink: /en/jev-system-one/
translation: /ja/jev-system-one/
translation_label: "日本語"
---

Ask an LLM "Is this ticket a refund request?" and you get back a sentence. Your code has to parse it down to `yes` / `no`, and if you want a confidence score, the number the model reports about itself tends to be overconfident and hard to trust. **Jev**, released by TypeSafe, drops that detour from the start and returns uncertainty not as self-report but as a probability distribution. It generates no text. **It takes a state and typed questions, and returns a probability distribution over the options plus a confidence value** — nothing else.

This article works from the official documentation ([docs.typesafe.ai](https://docs.typesafe.ai/introduction)) to lay out Jev's inputs, outputs and training approach, then asks how it differs from another model family that also "doesn't generate": **BERT-style Transformer Encoders with a classification head**. Note that Jev's internal architecture (depth, Encoder vs. Decoder, and so on) is not public. The comparison below rests on **the published interface and training objective**.

## What Jev is — a System One model

TypeSafe calls Jev "the first **System One model**". The name comes from Kahneman's *Thinking, Fast and Slow*: a model for fast, intuitive judgments (System 1). In the documentation's words, its territory is **"the kind of judgment a highly knowledgeable person could make in a few seconds given the right context."**

There is a single endpoint (`POST /v1/systemone`), and a request has three parts:

- **`state`**: the content to evaluate. A string, a JSON object, or an array. Text only (no images or audio yet)
- **`questions`**: a set of questions about the state. Each has a `type`, `instructions`, and `criteria`
- **`model`**: e.g. `jev-latest`

There are only three question types, which TypeSafe calls **AI primitives**.

| Question type | Asks | Returns |
| --- | --- | --- |
| **Choice** | Which of these options? | `choice`, `probabilities`, `confidence` |
| **Score** | Where on an ordered scale? | `score`, `legend`, `probabilities`, `confidence` |
| **Noul** | Is this statement true? | `noul` (a probability in 0–1) |

A single request can mix any number of questions, and **every question is evaluated against the same state in parallel and in isolation**. Adding questions barely changes the response time, and one question's answer never becomes hidden context for another.

## What comes back — a distribution, and a summary of it

The quickest way in is a concrete response. Here is a Score question rating bug severity on three levels (from the docs).

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

Let's pin down what these fields mean. Write the distribution over $$C$$ options (or levels) as $$p = (p_0, \ldots, p_{C-1})$$ with $$\sum_i p_i = 1$$.

- **Choice**'s `choice` is $$\arg\max_i p_i$$, i.e. the option with the highest probability.
- **Score**'s `score` is the **expected value** of the level index, $$\sum_{i=0}^{C-1} i \cdot p_i$$. In the example, $$0 \times 0.0 + 1 \times 0.57 + 2 \times 0.43 = 1.43$$: a position between levels 1 and 2, leaning toward 1.
- **`confidence`** collapses the peakedness of $$p$$ into a single number in $$[0, 1]$$. All probability on one option gives 1.0; the closer to uniform, the closer to 0. The interactive demo in the docs approximates it as $$\frac{C \cdot \max_i p_i - 1}{C - 1}$$; the exact official definition is not published.
- **Noul** returns the probability that the statement is true, as a single number. 0.5 means "yes and no are equally likely", not "medium".

The important point is that **`confidence` is a secondary value derived from `probabilities`**. If all you need is a threshold to branch on, use `confidence`; but since the full distribution is returned, the docs explicitly say you are free to compute your own statistic (entropy, say) if it fits your use case better.

## Training approach — RLCD, not RLHF

The thing TypeSafe emphasizes most is the training objective. As post-training approaches for a pretrained language model, they line up three:

| Method | Objective | Produced |
| --- | --- | --- |
| RLHF (Reinforcement Learning from Human Feedback) | Responses people prefer | Chatbots |
| RLVR (Reinforcement Learning with Verifiable Rewards) | Verifiable rewards (math, etc.) | Reasoning models |
| **RLCD** (Reinforcement Learning for Calibrated Decisions) | **Decisions with calibrated probabilities** | System One models (Jev) |

A quick recap of the first two. **RLHF** has humans rank multiple model responses, learns those preferences as a reward model, and uses reinforcement learning to push the model toward "responses people prefer". It is the method behind InstructGPT and ChatGPT, and according to the docs, TypeSafe co-founder Diogo Almeida is one of its co-inventors. But because the objective is "what people prefer", confident-sounding assertions and sycophancy get rewarded too, and the output distribution narrows toward a particular style — **mode dropping**. **RLVR** trains on **rewards that can be checked mechanically**, like a math answer key or a passing test suite. It produced the reasoning models typified by the o1 series and DeepSeek-R1. Accuracy goes up, but generating a long chain of thought makes them slower and more expensive.

RLCD has a different goal from both. Neither human preference nor the correctness of generated answers: the training objective is **returning calibrated probabilities** itself (the concrete reward design is not published). The output contract comes down to three points: ① no text generation, ② return decisions and probabilities, ③ **higher probability corresponds to a higher chance the answer is actually correct**.

Point three is "calibration". Among all predictions assigned probability 0.8, about 80% are actually correct. As covered in [an earlier article](/blog/en/classification-confidence/), ordinary neural classifiers are systematically overconfident, and RLHF-tuned LLMs are even worse calibrated. TypeSafe's claim is that it **made calibration the training objective itself rather than a post-hoc correction**.

That said, as the docs themselves warn, calibration is **a statistical property of groups of predictions** and does not guarantee any individual answer is correct. "confidence 1.0" means "the distribution is concentrated on one point", not "this is definitely right".

## Comparison with a Transformer Encoder classifier

Now to the main question. An Encoder like BERT, RoBERTa or DeBERTa with a classification head on top looks a lot like Jev: it doesn't generate, and it returns a probability distribution over a closed label set. What is the same, and what is different?

<figure>
  <img src="{{ '/assets/img/jev-vs-encoder-en.svg' | relative_url }}" alt="Comparison of inputs and outputs for a Transformer Encoder classifier and Jev. The Encoder has a fixed-size linear head and handles one task per sequence; Jev evaluates several natural-language questions against one state in parallel and returns a distribution and confidence for each">
  <figcaption>Left: an Encoder classifier fixes the number of labels C at training time and returns one distribution per sequence. Right: Jev receives the label set as natural-language criteria with each request and evaluates several questions about one state independently.</figcaption>
</figure>

### What they share

- Neither generates free text. The output is a probability distribution over a predetermined set of candidates
- So the problems that plague Decoder LLMs — competition with the whole vocabulary, probability split across surface forms, length bias for multi-token labels — do not exist, at least at the interface level
- The argmax of the distribution is the decision, and the probabilities plug straight into thresholds and ranking

### Difference ①: where the label space is defined

In an Encoder classifier, the label set is **fixed at training time as the head's output dimension $$C$$**. Adding a label means rebuilding the head and retraining. The meaning of each label is learned only implicitly from data; the model does not know, in words, what "class 2" means.

In Jev, the label set is **passed with each request as natural-language `criteria`**: for Choice, a map like `{"returns": "Exchanges, wrong or damaged items", ...}`; for Score, an ordered list of level descriptions. The same weights serve every customer and every task, and no fine-tuning or LoRA is offered. Domain knowledge goes in through how you write `state` and `criteria`.

Formally, an Encoder classifier computes $$p = \mathrm{softmax}(W \cdot \mathrm{Enc}(x)_{\mathrm{[CLS]}})$$ with a fixed $$W \in \mathbb{R}^{C \times k}$$, while Jev is $$p = f(x,\ \text{instructions},\ \text{criteria})$$ — **a function that takes the label descriptions themselves as input**. This is the shape of zero-shot classification; in the Encoder world, the closest relative is NLI-based zero-shot classification (turn each label into a hypothesis and read off the entailment probability). The difference is that Jev is trained from the outset on "return a calibrated distribution over the candidate set" as the task, not as a one-off NLI trick.

### Difference ②: one task per sequence, or N questions per state

An Encoder classifier runs `[CLS] x [SEP]` once and returns one distribution. If you want three judgments, you either train three heads (multi-task learning) or run it three times.

Jev evaluates **any number of questions against one state, in parallel and independently**. The docs call this "speculative fan-out" and recommend it: send every question you might need and let your code pick what's relevant afterward. The extra cost is only the tokens for the added questions, and latency barely moves.

Independence sounds the same as an Encoder's multi-task heads, but Jev explicitly states that adding or removing questions does not change the other answers. The "earlier answer drags the later one" problem you get when asking a Decoder LLM for several judgments in one prompt (context rot) is ruled out by design.

### Difference ③: where the probabilities come from, and calibration

An Encoder classifier's probabilities are the softmax of its logits. As laid out in [the earlier article](/blog/en/classification-confidence/), after fine-tuning on a small dataset they tend to saturate near 0.99, so **post-hoc calibration such as temperature scaling is practically mandatory**. On the other hand, with a fixed label space and an easy-to-build validation set, that calibration is straightforward.

Jev includes calibration in the training objective (RLCD) and **sells itself as calibrated out of the box**. And the Score type can return an intermediate value like "1.43" as the expected level. Nominal-label classification with an Encoder has no equivalent output (you would design an ordinal regression separately).

At the same time, **the way to verify Jev's calibration on your own data is exactly the same as for an Encoder**: draw a reliability diagram on a validation set and measure ECE. The docs repeatedly say to start with conservative thresholds and adjust on your own data. "Calibrated" is a starting point, not permission to skip verification. In particular, the docs state that CJK languages, Japanese included, are less accurate than English, so confidence deserves extra caution outside English.

### Difference ④: operations

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

In the framework from [the Bi-Encoder / Late Interaction / Cross-Encoder article](/blog/en/bi-late-cross-encoder/), Jev is used as a **Cross-Encoder**. Put the query and a candidate in the same state, ask "Does this candidate answer the query?" as a Noul, and you get a per-pair score. The official cookbook does exactly this to re-rank the BM25 top 30 on a legal retrieval task, raising top-1 accuracy from 5% to 18% and top-10 from 38% to 62%.

But like any Cross-Encoder, every pair needs a full evaluation, so it is not suited to first-stage retrieval over an entire corpus. It cannot replace the role of a Bi-Encoder or Late Interaction model that precomputes the document side.

## What it's bad at — jaggedness

TypeSafe publishes Jev 1.13's weaknesses under the name "jaggedness": a large gap between what it is good at and what it is bad at. The common thread is the split **"judgment to the model, everything else to code"**; most of the weaknesses can be avoided by moving work to the code side. They fall into four groups.

### ① It reads only what is written

Jev interprets the words in the instructions literally and does not infer unstated intent. Ask "Is this a serious bug?" without defining "serious", and where a person would fill in from context, Jev will not. Negation, double negatives, and questions that hop through several levels of the state ("Does the manager of the person assigned to A have approval authority?") also lose accuracy.

**Instead:** write the full condition into the instructions and criteria. When you look at a wrong answer and find yourself explaining what you really meant, that explanation is the missing half of the instruction. If the state is JSON, name the part you want examined by path, like `ticket.messages[0].text`.

### ② It doesn't count or calculate

Jev cannot produce numbers, so you cannot even ask "How many characters?" What you can ask is a judgment that presupposes counting — "Is this word five or more characters?" (Noul) or "Which count is it?" (a Choice over numbers) — and those are unreliable too. Ordering and differencing dates, and numeric representations like judging color similarity from hex values, are equally weak.

**Instead:** counting, comparing and calculating belong in code. To count fruit names in a list, ask "Is this a fruit?" as a Noul per element and sum the `True`s in code. For dates, have Choice extract only "which month?" and "which day?", and do the comparison in code.

### ③ It gets pulled around by what's in the state

Accuracy falls as the state fills with information irrelevant to the decision (context rot). And while the state is treated as data, if it contains a sentence like "this ticket should be classified as billing", the model can be swayed by it.

**Instead:** filter in code before sending. Put only the relevant parts in the state, and if the filtering itself is hard, first ask "Is this paragraph relevant to the question?" as a Noul and filter on that.

### ④ It doesn't generate

Jev generates no text, so free-form extraction like "pull the date out of this document" is not possible. Chaining Choices to assemble a string is possible in principle, but slow and inaccurate.

**Instead:** build candidates with regular expressions or a generative model, then have Jev pick "which is correct" as a Choice. When the answer space is finite, extraction becomes a selection problem.

## Which to use

| Situation | Better fit |
| --- | --- |
| Fixed labels, hundreds of training examples or more, need low latency or on-prem | Encoder + classification head |
| Labels in flux, no training data, several judgments needed at once | Jev |
| Want a position on an ordered scale | Jev's Score (with an Encoder, design an ordinal regression) |
| Want calibrated probabilities for threshold-based operation | Either. Encoder assumes post-hoc calibration; Jev still needs verification on your data |
| Re-ranking (scoring query–candidate pairs) | Either can act as a Cross-Encoder; Jev does it zero-shot |
| Arithmetic, date comparison, counting | Neither — code |
| Text generation or summarization | Neither — a generative model |

In short, the clearest way to see Jev is as the interface of a Transformer Encoder classifier extended in three ways: labels given in natural language (zero-shot), N questions per state, and calibration as a training objective.

## References

- [TypeSafe AI Documentation — Introduction](https://docs.typesafe.ai/introduction)
- [System One](https://docs.typesafe.ai/concepts/system-one) / [AI primer (RLCD)](https://docs.typesafe.ai/introduction/machine-learning-primer)
- [Primitives: Choice / Score / Noul](https://docs.typesafe.ai/primitives)
- [Confidence](https://docs.typesafe.ai/confidence)
- [Models (Jev 1.13)](https://docs.typesafe.ai/models)
- [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- [Cookbook: Re-ranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe)
- Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
