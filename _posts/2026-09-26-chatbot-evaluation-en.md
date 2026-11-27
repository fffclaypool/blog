---
layout: post
title: "A Systematic Guide to Evaluating LLM Chatbot Accuracy"
description: "A single number is a poor description of an LLM chatbot's accuracy. This article goes through how to measure each component (retrieval, scope check, generation) and each kind of failure with its own metric, evaluation data and judge, including how to validate LLM judges and how offline evaluation and online metrics divide the work."
date: 2026-09-26 18:00:00
lang: en
ref: chatbot-evaluation
permalink: /en/chatbot-evaluation/
translation: /ja/chatbot-evaluation/
translation_label: "日本語"
---

A single number is a poor description of a chatbot's accuracy. Whether retrieval found the right document, whether the answer is grounded in it, whether it fully answers the question, and whether the bot declined what it should decline are separate failures in separate components. Averaged into one figure, they no longer show what to fix.

This article goes through how to measure each component and each kind of failure with its own metric, evaluation data and judge. An offline evaluation set catches regressions when something changes, and online metrics show how real users behave.

## Separate the definitions of "correct" first

Answers can be wrong in different ways, so the kinds are named first.

Correctness is whether the answer matches the facts, and it is measured against a reference answer. Faithfulness is whether the answer follows from the context the model was given, and it can be judged without a reference as long as the context is available. In RAG the two can diverge. An answer that is faithful to an out-of-date context is an index problem. An answer that is factually right but relies on knowledge outside the context is a generation problem.

Helpfulness is whether the answer covers what was asked without gaps or padding. Replying "Yes, you can return it" to a customer who asked about the return deadline is correct and does not help. Policy compliance is whether the bot declines out-of-scope and harmful questions without declining in-scope ones. Format is whether the output follows the specified JSON, length and tone.

[The previous article](/blog/en/rag-lora-finetuning/) divided chatbot failures into "doesn't know" and "behaves wrong". Correctness and faithfulness fall under "doesn't know", and helpfulness, policy compliance and format fall under "behaves wrong". With the same division in the evaluation, a result points directly to the kind of fix it needs.

## Measure per component

<figure>
  <img src="{{ '/assets/img/eval-pipeline-en.svg' | relative_url }}" alt="Chatbot pipeline with per-component metrics. A query passes through scope check, retrieval and generation. The scope check is measured by over-refusal and leak rates, retrieval by Recall@k and MRR/nDCG, generation by correctness, faithfulness, helpfulness and format. The whole is measured by end-to-end task success and by production escalation rates and A/B tests">
  <figcaption>An upstream failure cannot be recovered downstream, so the scope check, retrieval and generation are evaluated one at a time in that order, and then the pipeline is measured end to end.</figcaption>
</figure>

The pipeline runs the scope check, then retrieval, then generation. If retrieval misses the right document, generation cannot produce the right answer however capable the model is. The evaluation therefore starts upstream and takes one component at a time.

### Retrieval: does the gold passage make it into the top k?

The evaluation data is a set of (query, passage containing the answer) pairs, with the gold passage labeled by a person. The central metric is Recall@k. If the top $$k$$ passages go into the prompt and the gold passage is not among them, generation has almost no chance of answering correctly. MRR or nDCG can be added when rank matters. Their definitions are in [the article on IR metrics](/blog/en/information-retrieval-metrics/).

This evaluation does not call an LLM and gives the same result every time. It finishes in seconds, so it is run after every change to chunking, the embedding model or the reranker.

### Scope check: count over-refusal and leakage separately

The scope check is a classification problem, so it is evaluated with a confusion matrix. The rate of refused in-scope questions (over-refusal) and the rate of out-of-scope questions that get through are reported separately. On an e-commerce site, over-refusal costs sales and leakage creates brand or legal risk, so the two errors do not weigh the same.

Over-refusal is the easier one to miss. If the evaluation set contains only typical in-scope questions and obviously out-of-scope ones, refusals of legitimate questions that merely look out of scope do not show up in the numbers. XSTest (Röttger et al., 2024) is a test set of harmless questions worded to look unsafe, built to measure this. The same approach applies to a product domain: collect legitimate questions that look borderline and add them to the evaluation set. If the scope check outputs a probability, the threshold and calibration are tuned with the procedure in [the article on classifier confidence](/blog/en/classification-confidence/).

### Generation: check against a reference or against the context

Generation takes the most effort to evaluate because its output is free text.

Questions with a reference answer, such as "the return window is 14 days", are measured by agreement with the reference. Exact match or token-level F1 works for short answers. Different wording with the same meaning should still count as correct, so in practice an LLM is often asked whether the answer says the same thing as the reference. BLEU and ROUGE correlate poorly with human judgment on free-text answers and are not used as the main metric.

Questions without a practical reference are checked against the context. The answer is split into individual claims, each claim is judged on whether it follows from the retrieved passages, and the fraction of supported claims is the faithfulness score.

$$
\mathrm{faithfulness} = \frac{\text{claims supported by the context}}{\text{claims in the answer}}
$$

<figure>
  <img src="{{ '/assets/img/claim-faithfulness-en.svg' | relative_url }}" alt="Claim-level faithfulness. The answer is split into four claims and each is checked against the retrieved passages. Three are supported and one is not, so faithfulness is 3/4 = 0.75">
  <figcaption>The answer is split into claims and each one is checked against the context. This is more stable than scoring the whole answer and identifies the unsupported claim.</figcaption>
</figure>

RAGAS (Es et al., 2023) and FActScore (Min et al., 2023) both work this way. Claim-level judgments vary less than a 1-to-5 score for the whole answer, and they identify the unsupported part.

Helpfulness is judged from the question and the answer together: whether the answer addresses the question and whether anything needed is missing. RAGAS generates questions back from the answer and uses their similarity to the original question as answer relevance. In practice, an LLM scoring against a rubric criterion such as "does this answer settle what the customer came for?" usually fits the purpose better.

Format is checked with code. Schema validation for JSON, a character count for length and regular expressions for forbidden phrases are more reliable than asking an LLM.

### Finally, measure end to end

Components can each score well while the combined result is poor. For example, retrieval returns five passages including the right one, and the model still answers wrongly because of the other four. Component-level evaluation does not detect this. The pipeline is therefore also evaluated from question to final answer, and a gap between that result and the component scores points to the hand-off between components.

Multi-turn conversations are evaluated per turn and as a whole. Per turn, the response is judged given the history so far. For the whole conversation, a person or a model judges whether the user got what they came for, and the results are reported as a task success rate.

## Choosing the judge

Anything that can be judged deterministically should be: exact match, schema validation, regular expressions and retrieval recall. These checks are reproducible, cost almost nothing and do not depend on a judge's tendencies.

Free text is judged by an LLM. Zheng et al. (2023) found that GPT-4's judgments of which of two answers was better agreed with human judges a little over 80% of the time, about as often as humans agree with each other. LLM judges do have known biases, and each needs a countermeasure.

A judge comparing two answers tends to prefer the one presented first (Wang et al., 2023). Judge twice with the order swapped and keep the verdict only when both runs agree. Judges also rate longer answers higher, so the rubric should penalize unnecessary information, and length should be measured separately. A model rates its own output higher (Panickssery et al., 2024), so the judge should be a different model from the generator. Absolute scores on a 1-to-10 scale are applied inconsistently, so pass/fail or A-versus-B comparisons are preferable. Having the model write out its evaluation steps, as in G-Eval (Liu et al., 2023), also helps to some extent.

An LLM judge is itself a classifier, so its accuracy is checked before use. Compute its agreement with 100 to 200 human-labeled examples (Cohen's κ, for example) and revise the rubric if agreement is low. ARES (Saad-Falcon et al., 2024) turns this idea into a framework that uses a small set of human labels to correct the bias in LLM judgments and produce confidence intervals.

Human evaluation is the most reliable option and the most expensive. Reviewing every response by hand is not sustainable, so human effort goes to labeling the examples that validate the LLM judge and to periodic audits.

## Building the evaluation set

Questions should come from production logs where possible, anonymized and kept in the users' own wording. Before launch, when there are no logs, an LLM can generate questions from the FAQ or the documents. Generated questions reuse the documents' wording, so retrieval recall measured on them is higher than it will be in production. They are replaced once logs accumulate.

Questions are grouped into strata: intent (shipping, returns, payment and so on), out-of-scope questions, ambiguous questions, legitimate questions that look borderline, prompt-injection attempts and multi-turn conversations. An overall average hides failures in low-volume categories such as returns and payments. Categories with a high business cost are tallied separately even when they are small.

The number of questions needed is estimated from the sampling error. The standard error of an accuracy $$p$$ estimated from $$n$$ examples is $$\sqrt{p(1-p)/n}$$, so at $$p = 0.8$$ the 95% confidence interval is as follows.

| n | 95% CI |
| --- | --- |
| 50 | ±11 pt |
| 100 | ±8 pt |
| 400 | ±4 pt |
| 1000 | ±2.5 pt |

With 100 examples, a 5-point improvement cannot be told apart from noise. The error for a stratum depends on that stratum's count, so 100 or more per intent is a reasonable target. When two configurations are compared on the same questions and the difference is taken per question, far fewer examples are enough.

The evaluation set is frozen and versioned. If its questions are used as few-shot examples or for prompt tuning, the score rises on those questions only and stops meaning anything. A development set for repeated use and a test set for the final check are kept separately.

## Offline and online: who does what

The purpose of offline evaluation is to catch regressions before they reach production. Whenever the prompt, the model, the index or the chunking changes, the evaluation set is run and compared with the baseline. It runs in CI, and changes that fall below a threshold are not merged. Running deterministic metrics every time and LLM judgments only on the main strata keeps the cost down.

Online metrics show what happens on questions that are not in the evaluation set. The available signals include thumbs-up/down buttons, the rate of escalation to a human, the rate at which users rephrase a question, and the share of conversations that end with the request resolved. All of them are indirect, and feedback buttons are pressed by a biased subset of users. The most trustworthy way to compare two configurations is an A/B test that splits users between them and compares escalation or resolution rates.

Production traffic is also sampled periodically and run through the same judgments as the offline set. Shifts in the distribution and new kinds of questions found this way are added to the evaluation set.

## Langfuse puts it on one screen

For running all of this in one tool, my choice is Langfuse. It is open source under the MIT license, can be self-hosted with Docker Compose, and is also available as a cloud service. Traces, scores, datasets and human annotation are handled on the same screen.

### Record each component in the trace

Traces are sent from the Python or JS SDK, or through OpenTelemetry. One request becomes one trace, and the scope decision, the retrieved chunks, and the prompt and output of generation are recorded as nested steps inside it. Integrations exist for LangChain, LlamaIndex, the OpenAI SDK and others, so a few added lines are often enough. Multi-turn conversations are grouped into sessions, which can be used for conversation-level evaluation. These records are what per-component evaluation runs on.

### Collect scores from every source in one place

In Langfuse, every evaluation result is attached to a trace as a score. Results of Recall@k or schema validation are sent from the SDK. An LLM judge registered as an evaluation template runs automatically on a sample of production traces. Human labels are added through the annotation queue described below. End-user thumbs can also be sent as scores from the SDK.

Scores from different sources sit on the same trace, so agreement between the LLM judge and human labels can be computed by filtering and exporting from the UI.

### Run human labeling through annotation queues

The first step is to define the scores. Fixing the name, the type (numeric, categorical or boolean) and the allowed range reduces variation between reviewers. Next, the traces to be labeled are put into a queue by filter or sampling. Reviewers label the items in order while looking at the question, the retrieved passages and the answer, and each label is saved as a score.

The main uses are producing the 100 to 200 labels that validate the LLM judge, periodic audits, and sorting borderline questions. Sending only traces with low judge scores to the queue keeps the review load small and shows where the judge tends to be wrong.

### Turn traces directly into the evaluation set

Choosing "add to dataset" in the trace view turns that question and its context into an evaluation item. The expected output can be edited afterwards. This is how the evaluation set is updated from logs.

Experiments (runs) are executed against a dataset from the SDK and compared side by side in the UI. Prompts are versioned too, so each run can be traced back to the prompt that produced it.

### The overall routine

The LLM judge runs on a sample of production traces, low-scoring traces go to the annotation queue for human labels, and the ones worth keeping are added to the dataset. After a configuration change, the dataset is run and compared with the previous run. The routine does not depend on Langfuse and can be set up the same way with another tool.

## Pitfalls

These problems come up often in evaluation design.

- Looking only at the average score. It does not show which component to fix, so when a number moves, check the per-component numbers first.
- Using the same model as generator and judge. The judge favors its own output, and the score comes out too high.
- Evaluating retrieval with questions generated from the indexed documents. Recall is overstated.
- Leaving borderline-looking legitimate questions out of the evaluation set. Over-refusal goes undetected.
- Using evaluation questions as few-shot examples or for prompt tuning. The score rises and production quality does not.
- Averaging across intents without regard to their cost. Expensive failures in returns or payments become hard to see.

## Procedure

1. Collect failures. Gather wrong answers from logs, or from anticipated questions if there are no logs yet, and classify each as a correctness, faithfulness, helpfulness, policy or format failure.
2. Build evaluation data for each component: (query, gold passage) for retrieval, (query, label) for the scope check, and (query, context, reference or rubric) for generation.
3. Start with the deterministic metrics: Recall@k, the confusion matrix and schema validation.
4. Build the LLM judge and check it against human labels before using it.
5. Put the evaluation in CI and compare against the baseline on every change.
6. In production, watch online metrics and add samples to the evaluation set.

The table below lists the metric, data and judge for each item.

| What | Metric | Data | Judge |
| --- | --- | --- | --- |
| Retrieval | Recall@k, MRR, nDCG | query + gold passage | deterministic |
| Scope check | over-refusal rate, leak rate, confusion matrix | query + label | deterministic |
| Correctness | agreement with reference | query + reference | deterministic or LLM |
| Faithfulness | fraction of supported claims | query + context | LLM (human-validated) |
| Helpfulness | rubric score, pairwise comparison | query + rubric | LLM (human-validated) |
| Format | schema, length, forbidden phrases | spec | deterministic |
| End to end | task success rate | conversations | human or LLM |
| Production | escalation rate, rephrase rate, A/B | logs | user behavior |

Describing a chatbot's "accuracy" means reporting a number for each row of this table. If a single number is required, the rows can be weighted by business cost and summed. That number does not show what to fix, so the single figure for reporting and the table for development are kept separately.

## References

- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.
- Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024 (System Demonstrations)*.
- Min, S., et al. (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. *EMNLP 2023*.
- Liu, Y., et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. *EMNLP 2023*.
- Wang, P., et al. (2023). Large Language Models are not Fair Evaluators. *ACL 2024*.
- Panickssery, A., Bowman, S. R., & Feng, S. (2024). LLM Evaluators Recognize and Favor Their Own Generations. *NeurIPS 2024*.
- Saad-Falcon, J., et al. (2024). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. *NAACL 2024*.
- Röttger, P., et al. (2024). XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. *NAACL 2024*.
- [Langfuse Documentation](https://langfuse.com/docs)
