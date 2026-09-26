---
layout: post
title: "A Systematic Guide to Evaluating LLM Chatbot Accuracy"
description: "An LLM chatbot's 'accuracy' is not one number. Measure each component (retrieval, scope check, generation) and each kind of failure (correctness, faithfulness, helpfulness, policy compliance) with its own metric, evaluation data and judge. Covers the biases of LLM-as-a-judge and how to validate it, how large an evaluation set needs to be and where to get it, the division of labor between offline evaluation and online metrics, and how to run it all in Langfuse."
date: 2026-09-26 18:00:00
lang: en
ref: chatbot-evaluation
permalink: /en/chatbot-evaluation/
translation: /ja/chatbot-evaluation/
translation_label: "日本語"
---

A chatbot's "accuracy" is not one number. Did retrieval find the right document? Is the answer grounded in it? Does it fully answer the question? Did the bot decline the questions it should decline? These are different failures of different components, and averaging them into one figure hides where to fix.

Evaluation is the work of measuring, separately, which component produces which kind of failure and how often. Give each component its own metric, evaluation data and judge; use an offline evaluation set to block regressions on every change; and watch real user behavior online. What follows is how to put that together.

## Separate the definitions of "correct" first

Wrong answers are wrong in different ways.

Correctness asks whether the answer matches the facts; you check it against a reference. Faithfulness asks whether the answer follows from the provided context; it can be judged without a reference as long as the context is available. In RAG the two can diverge. An answer can be faithful to a context that is itself out of date and factually wrong. An answer can be factually right yet drawn from knowledge not in the context, meaning the model filled it in on its own. The first is an index problem, the second a generation problem, and they are fixed in different places.

Helpfulness asks whether the answer covers the question without padding. "Yes, you can return it" is correct but useless to a customer who asked about the deadline. Policy compliance asks whether out-of-scope and harmful questions are declined, and whether in-scope questions are not declined too often. Format asks whether the specified JSON, length and tone are respected.

[The previous article](/blog/en/rag-lora-finetuning/) split chatbot failures into "doesn't know" and "behaves wrong". Correctness and faithfulness belong to the first; helpfulness, policy and format to the second. Dividing the evaluation along the same line lets you read the remedy off the result.

## Measure per component

<figure>
  <img src="{{ '/assets/img/eval-pipeline-en.svg' | relative_url }}" alt="Chatbot pipeline with per-component metrics. A query passes through scope check, retrieval and generation. The scope check is measured by over-refusal and leak rates, retrieval by Recall@k and MRR/nDCG, generation by correctness, faithfulness, helpfulness and format. The whole is measured by end-to-end task success and by production escalation rates and A/B tests">
  <figcaption>Upstream failures cannot be recovered downstream, so evaluate the scope check, retrieval and generation one at a time in that order, then end to end.</figcaption>
</figure>

The pipeline runs scope check, retrieval, generation. An upstream failure cannot be recovered downstream. If retrieval misses the right document, no amount of cleverness in generation produces the right answer. So evaluate from upstream, one component at a time.

### Retrieval: does the gold passage make it into the top k?

The evaluation data is (query, passage containing the answer) pairs, with the gold passage labeled by a person. The main metric is Recall@k. If the top $$k$$ go into the prompt and the gold passage is not among them, generation has no chance. Use MRR or nDCG when rank matters. The definitions are in [the article on IR metrics](/blog/en/information-retrieval-metrics/).

This evaluation is deterministic and runs without calling an LLM. It finishes in seconds, so run it every time you change chunking, the embedding model or the reranker.

### Scope check: count over-rejection and leakage separately

The scope check is a classification problem, so look at the confusion matrix. Report the rate at which in-scope questions are refused (over-refusal) and the rate at which out-of-scope questions get through, and do not blend them into one accuracy figure. For an e-commerce site, over-refusal is lost sales and leakage is brand or legal risk; the weights differ.

Over-refusal is the one that gets missed. If you evaluate only on in-scope questions, excessive refusal never shows up. Röttger et al. (2024) built XSTest, a set of harmless questions phrased to look unsafe, to measure exactly that. Apply the same idea to your domain: collect legitimate questions that look borderline. If the decision comes as a probability, tune the threshold and calibration by the procedure in [the article on classifier confidence](/blog/en/classification-confidence/).

### Generation: check against a reference or against the context

Generation is the hardest to evaluate because the answer is free text.

For questions with a reference answer ("the return window is 14 days"), measure agreement with the reference. Exact match or token F1 for short answers. Since you want to accept different wording with the same meaning, in practice an LLM is often asked "does this say the same thing as the reference?". N-gram overlap metrics such as BLEU and ROUGE correlate poorly with human judgment on free-text answers and should not be the primary metric.

For questions where a reference is hard to write, switch to checking against the context. Split the answer into individual claims, judge whether each follows from the retrieved passages, and use the fraction of supported claims as the faithfulness score.

$$
\mathrm{faithfulness} = \frac{\text{claims supported by the context}}{\text{claims in the answer}}
$$

<figure>
  <img src="{{ '/assets/img/claim-faithfulness-en.svg' | relative_url }}" alt="Claim-level faithfulness. The answer is split into four claims and each is checked against the retrieved passages. Three are supported and one is not, so faithfulness is 3/4 = 0.75">
  <figcaption>Split the answer into claims and check each one against the context. More stable than scoring the whole answer, and it shows which claim lacks support.</figcaption>
</figure>

RAGAS (Es et al., 2023) and FActScore (Min et al., 2023) take this form. Working at the claim level pinpoints which part is unsupported, is more stable than giving the whole answer a score from 1 to 5, and tells you what to fix.

Helpfulness is judged on the (question, answer) pair: does it answer the question, and is anything missing? RAGAS generates questions back from the answer and uses their similarity to the original as answer relevance, but in practice a rubric that states "does this answer settle the customer's business?" and an LLM scoring against it fits the purpose better.

Format is checked mechanically. Schema validation for JSON, character counts for length, regular expressions for forbidden phrases. There is no need to ask an LLM, and not asking is more reliable.

### Finally, measure end to end

Every component can score well and the combination can still be bad. Retrieval returns five passages including the right one, but the model is pulled off course by the other four: that failure is visible only end to end. Evaluate from question to final answer in one pass and compare against the component scores. A gap means the problem is at a seam between components.

For multi-turn conversations, separate per-turn evaluation, where each turn is judged given the history so far, from whole-conversation evaluation, where the question is whether the user's business got done. The latter is the task success rate, judged per conversation by a person or a model.

## Choosing the judge

Wherever a deterministic judgment is available, use it. Exact match, schema validation, regular expressions, retrieval recall. They are reproducible, free and unbiased.

For free text, use an LLM. Zheng et al. (2023) had GPT-4 judge which of two answers was better and found agreement with human judges above 80%, on par with agreement between humans. It works, but it has habits.

Shown two answers side by side, it tends to pick the one presented first (Wang et al., 2023); judge twice with the order swapped and keep only agreeing verdicts. It tends to rate longer answers higher, so write "penalize unnecessary information" into the rubric and measure length separately. When the judge and the generator are the same model, it rates its own output higher (Panickssery et al., 2024), so judge with a different model. Absolute scores on a 1 to 10 scale drift; pass/fail or A-versus-B comparisons are more stable. Having the model spell out its evaluation steps, as in G-Eval (Liu et al., 2023), helps somewhat.

What must not be forgotten is that an LLM judge is itself a classifier. Measure its accuracy before trusting it. Take 100 to 200 human-labeled examples, compute agreement (Cohen's κ or similar), and revise the rubric if it is low. ARES (Saad-Falcon et al., 2024) formalizes the same idea: a small set of human labels corrects the bias of LLM judgments and yields confidence intervals.

Human evaluation is the most reliable and the most expensive. Limit its role to producing the labels that validate the LLM judge and to periodic audits. Reviewing everything by hand does not last.

## Building the evaluation set

Where do the questions come from? Production logs first, anonymized and kept in the users' own wording. Before launch, when there are no logs, have an LLM generate questions from the FAQ or documents. Generated questions echo the documents' wording, though, so retrieval recall comes out higher than reality. Replace them as logs accumulate.

Stratify by kind: intent (shipping, returns, payment), out of scope, ambiguous, legitimate-but-borderline, injection attempts, multi-turn. Looking only at the overall average dilutes failures in low-volume categories like returns and payments. Categories with high business cost get their own tally even when the counts are small.

How many? The standard error of an accuracy $$p$$ estimated from $$n$$ examples is $$\sqrt{p(1-p)/n}$$, so at $$p = 0.8$$ the 95% confidence interval is:

| n | 95% CI |
| --- | --- |
| 50 | ±11 pt |
| 100 | ±8 pt |
| 400 | ±4 pt |
| 1000 | ±2.5 pt |

With 100 examples a 5-point improvement disappears into noise. Per-stratum figures depend on the stratum's own count, so aim for at least 100 per intent. When comparing two configurations, though, pairing them on the same question set and taking the difference needs far fewer examples.

Freeze the evaluation set and version it. Using its questions as few-shot examples or for prompt tuning inflates the score on exactly those questions and makes it meaningless. Keep a development set you look at repeatedly and a test set you look at only at the end.

## Offline and online: who does what

Offline evaluation exists to block regressions. Whenever the prompt, the model, the index or the chunking changes, run the evaluation set and compare against the baseline. Put it in CI and refuse to merge below the threshold. Run deterministic metrics every time and LLM judgments only on the main strata, and the cost stays manageable.

Online metrics tell you what is happening on questions that are not in the evaluation set. Available signals are the thumbs-up/down buttons, the escalation-to-human rate, the rate of rephrased questions, and the rate of conversations that end with the user's business done. All are indirect, and the people who press feedback buttons are a biased sample. The most trustworthy is an A/B test: split users between configurations A and B and compare escalation or resolution rates.

Periodically sample production traffic and run the same judgments as offline. Check whether the distribution has shifted or new kinds of questions are appearing, and add them to the evaluation set. The evaluation set is not built once; it is grown from logs.

## Langfuse puts it on one screen

If one tool is to carry all of the above, I use Langfuse. It is open source (MIT), self-hostable with Docker Compose, and also offered as a cloud service. Traces, scores, datasets and human annotation live on the same screen, so the offline-online loop from the previous section can be built as is.

### Record each component in the trace

Send traces from the Python or JS SDK, or via OpenTelemetry. One request becomes one trace, and inside it the scope decision, the chunks retrieval returned, and the prompt and output of generation are nested. Integrations for LangChain, LlamaIndex, the OpenAI SDK and others mean a few added lines are usually enough. Multi-turn conversations are grouped into sessions, which serves conversation-level evaluation.

Measuring per component requires per-component records. The trace is that foundation.

### Collect scores from every source in one place

In Langfuse every evaluation result is attached to a trace as a score. Deterministic judgments such as Recall@k or schema validation are sent from the SDK. For LLM judges, register an evaluation template in Langfuse and it runs automatically on a sample of production traces. Human labels come from the annotation queue described next. End-user thumbs can also be sent as scores from the SDK.

Because scores from different sources sit side by side on the same trace, computing agreement between the LLM judge and human labels is a matter of filtering and exporting. "Measure the judge before trusting it" needs no special machinery.

### Run human labeling through annotation queues

First define the scores: name, type (numeric, categorical, boolean) and allowed range, so that reviewers label consistently. Then put the traces to be labeled into a queue by filter or sampling. Reviewers open the queue and label each item in turn while looking at the question, the retrieved passages and the answer. Labels are stored directly as scores.

Uses: the 100 to 200 labels for validating the LLM judge, periodic audits, and sorting borderline questions. Routing only traces with low judge scores into the queue keeps the human workload small while revealing where the judge tends to be wrong.

### Turn traces directly into the evaluation set

From the trace view, "add to dataset" turns the question and its context into an evaluation item. The expected output can be edited afterwards. "Grow the evaluation set from logs" in the previous section is this operation, repeated.

Against a dataset you run experiments (runs) from the SDK and compare runs side by side in the UI. Prompts are versioned, so you can trace which prompt produced which run. The regression routine, run on every change and compare with the baseline, lives here too.

### The shape of the routine

Run the LLM judge on a sample of production traces, route the low-scoring ones into the annotation queue, have people label them, and add what is needed to the dataset. On every change, run the dataset and compare with the previous run. This loop closes inside the Langfuse UI. Swap the tool and the shape stays the same.

## Pitfalls

Arguing from the average alone leaves you unable to tell which component to fix. When a number moves, go back to the per-component numbers first.

If the generator and the judge are the same model, self-preference inflates the score. If the evaluation set was generated from the index, retrieval recall is overstated. If only in-scope questions are evaluated, over-refusal is invisible. If evaluation questions leak into the prompt, scores rise while production does not change. If costs are averaged across intents, expensive failures in returns or payments are diluted. Every one of these is avoided by deciding the evaluation design up front.

## Procedure

1. Collect failures. Gather wrong answers from logs or from expectation and sort them into correctness, faithfulness, helpfulness, policy and format.
2. Build evaluation data per component: (query, gold passage) for retrieval, (query, label) for the scope check, (query, context, reference or rubric) for generation.
3. Start with deterministic metrics: Recall@k, the confusion matrix, schema validation.
4. Build the LLM judge and validate it against human labels before using it.
5. Wire it into CI and compare against the baseline on every change.
6. In production, watch online metrics and feed samples back into the evaluation set.

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

Any one-line statement of "accuracy" ends up being this table with its rows filled in. If a single number is unavoidable, weight the rows by business cost and add them up. That number cannot tell you where to fix, so keep the one-line figure for reporting and the table for development.

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
