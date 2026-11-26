---
layout: post
title: "Choosing Between RAG, LoRA and Fine-Tuning for an LLM Chatbot"
description: "RAG, LoRA (low-rank adaptation) and full fine-tuning are often compared as interchangeable ways to build a chatbot that answers from a company's own information, but they change different things. This article explains each mechanism with a few equations, then compares them by which of two failures they fix, 'doesn't know' or 'behaves wrong', and by cost, update cadence, citations and access control. It also covers how to combine them, how to keep the bot inside one domain, and how to choose."
date: 2026-09-26 12:00:00
lang: en
ref: rag-lora-finetuning
permalink: /en/rag-lora-finetuning/
translation: /ja/rag-lora-finetuning/
translation_label: "日本語"
---

When people set out to build an LLM chatbot that knows their company's products and policies, three options come up first: RAG (Retrieval-Augmented Generation), LoRA (Low-Rank Adaptation) and fine-tuning. They are usually compared by which one gives the best accuracy, but they are not alternative solutions to the same problem. RAG changes the input at inference time, while LoRA and full fine-tuning change the model's weights. Because they change different things, they fix different failures.

So RAG is the choice for making the bot answer with your company's facts, and LoRA for making it match your tone and format. Full fine-tuning is worth considering only after both have been tried and found insufficient.

## Failures split into "doesn't know" and "behaves wrong"

When a stock LLM is given a company's support questions, its wrong answers fall into two groups.

The first is a knowledge failure. The model cannot answer what was not in its training data, such as your pricing terms, a feature shipped last week, internal jargon, or an incident report updated yesterday. It tries to answer anyway and produces a plausible-sounding hallucination.

The second is a behavior failure. The model knows the content, but the form of the output is off. The tone is too stiff, a preamble is added when bare JSON was requested, internal terms are paraphrased into generic ones, or the model answers a question it should have declined and declines one it should have answered.

The two are separated because their remedies point in opposite directions. Putting knowledge into weights is hard, and fixing behavior through the input is expensive. RAG addresses knowledge failures, and LoRA and fine-tuning address behavior failures.

## How the three methods work

An LLM models the conditional probability $$p_\theta(y \mid x)$$ of an output $$y$$ given an input token sequence $$x$$, with parameters $$\theta$$. The three methods differ in which part of this expression they change.

<figure>
  <img src="{{ '/assets/img/rag-lora-ft-where-en.svg' | relative_url }}" alt="Where RAG, LoRA and full fine-tuning intervene. RAG concatenates retrieved passages to the query and feeds a frozen LLM (changes the input). LoRA adds a small low-rank product BA to frozen weights W0 (changes a few added weights). Full fine-tuning updates all parameters θ (changes all weights)">
  <figcaption>Left: RAG leaves the LLM's weights unchanged and changes the input by appending retrieved passages to the query. Middle: LoRA freezes the original weights and learns only a low-rank delta. Right: full fine-tuning updates every parameter.</figcaption>
</figure>

### RAG changes the input

RAG does not change $$\theta$$. It retrieves documents relevant to the query $$x$$ from an external index and concatenates them into the prompt.

$$
p(y \mid x) \approx p_\theta\bigl(y \mid x,\ d_1, \ldots, d_k\bigr), \qquad d_1, \ldots, d_k = \mathrm{Retrieve}(x)
$$

In advance, documents are split into chunks, embedded and stored in an index. When a query arrives, it is embedded in the same space, nearest neighbors are retrieved and optionally reranked with a Cross-Encoder, and the top $$k$$ passages go into the prompt for generation. Retriever architectures (Bi-Encoder / Late Interaction / Cross-Encoder) are covered in [an earlier article](/blog/en/bi-late-cross-encoder/), and measuring retrieval quality in [the article on IR metrics](/blog/en/information-retrieval-metrics/).

Nothing is trained on the LLM side, so a document can be used in answers as soon as it is added to the index. In exchange, every request pays for retrieval latency and for the input tokens of the top $$k$$ passages.

### Full fine-tuning changes all the weights

Full fine-tuning takes (input, desired output) pairs and updates every parameter by gradient descent. For chatbots this usually means SFT (Supervised Fine-Tuning) of a pretrained model.

The first problem is cost. An Adam-style optimizer in mixed precision needs about 16 bytes per parameter for weights, gradients and optimizer state combined (Rajbhandari et al., 2020). For a 7B model that is roughly 112 GB before activations. This does not fit on one GPU, so distributed training across several GPUs is assumed.

The second problem is catastrophic forgetting. Updating every weight on company data can degrade general abilities acquired in pretraining, such as reasoning, other languages and common sense. After fine-tuning, regressions on general-capability benchmarks have to be checked along with accuracy on the target task.

### LoRA changes a few added weights

LoRA (Hu et al., 2021) starts from the hypothesis that the weight delta $$\Delta W$$ produced by fine-tuning is effectively low-rank. It freezes the original weight $$W_0$$ and expresses the delta as the product of two small matrices.

$$
h = W_0 x + \frac{\alpha}{r} B A x, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)
$$

Only $$A$$ and $$B$$ are trained ($$\alpha$$ is a scaling constant).

<figure>
  <img src="{{ '/assets/img/lora-decomposition-en.svg' | relative_url }}" alt="LoRA low-rank decomposition. A frozen d×k weight W0 is summed with the product of a d×r matrix B and an r×k matrix A. Trainable parameters drop from dk to r(d+k); with d=k=4096 and r=8, from about 16.8M to about 65K (about 0.4%)">
  <figcaption>The original weight W₀ is frozen and only the delta ΔW = BA is learned. With d = k = 4096 and r = 8, trainable parameters per layer drop from about 16.8M to about 65K.</figcaption>
</figure>

Trainable parameters per layer drop from $$dk$$ to $$r(d + k)$$. With $$d = k = 4096$$ and $$r = 8$$ that is 0.4% of the original, and around 0.3% across a 7B-class model. Most of the memory goes to the frozen weights and activations.

$$B$$ is initialized to zero, so training starts from exactly the original model's outputs. After training, $$W_0 + \frac{\alpha}{r} BA$$ can be merged into a single matrix, which adds no inference cost. If the adapter is left unmerged, one base model can serve several adapters (tens of MB each) that are swapped in and out. Adapters can be separated per customer or per task while sharing a single copy of the base weights.

QLoRA (Dettmers et al., 2023) is a variant that quantizes the frozen $$W_0$$ to 4 bits in memory and trains only the LoRA adapters in 16 bits. The paper showed that a 65B model can be fine-tuned on a single 48 GB GPU. It also reports that attaching LoRA to all linear layers matters more than increasing $$r$$.

The term "low-rank training" generally refers to LoRA and its variants (QLoRA, DoRA and so on). The rest of this article uses LoRA to represent them.

## Comparison

### For knowledge, RAG

With RAG the knowledge is in the index, and a document can be used in answers as soon as it is added. The model only has to read and answer, which it learned thoroughly in pretraining.

Injecting knowledge by fine-tuning is hard. Ovadia et al. (2023) report that RAG consistently outperforms unsupervised fine-tuning on both existing and new knowledge. The same experiments found that a model learns a new fact through fine-tuning only when it sees the fact repeatedly in multiple paraphrases. In pretraining, a single fact appears across the web in hundreds of different wordings. A fact that appears once in a few hundred Q&A pairs gives too weak a gradient signal.

LoRA is at a further disadvantage. Biderman et al. (2024) ran continued pretraining on code and math and found that LoRA learns substantially less than full fine-tuning. The rank of the $$\Delta W$$ produced by full fine-tuning was 10–100× the $$r$$ typically used in LoRA. The low-rank constraint does not have the capacity for large amounts of new knowledge. The same paper also reports that LoRA forgets less of the original abilities. It learns less and, correspondingly, forgets less.

Fine-tuning for knowledge therefore does not work with a few hundred Q&A pairs. It works only at the scale of continued pretraining on large volumes of domain text. To make a chatbot answer with your company's facts, start with RAG.

### For behavior, LoRA and fine-tuning

Tone, output format, use of terminology, refusal policy and answer length are all aspects of how to answer, and they are consistent patterns that are easy to learn from examples. In the original LoRA paper (Hu et al., 2021), LoRA with $$r$$ between 1 and 8 matched full fine-tuning on GLUE and E2E NLG. Biderman et al. (2024) likewise report that the gap between LoRA and full fine-tuning is large in continued pretraining and small in instruction tuning. A few hundred to a few thousand (input, ideal output) pairs are often enough to see an effect.

Fixing behavior with RAG alone means adding more and more instructions to the system prompt. Short instructions work well, but as they grow longer, parts of them are more likely to be ignored, and their tokens are paid for on every request. Few-shot examples make the prompt longer still. This is the sense in which fixing behavior through the input is expensive.

Full fine-tuning has the highest ceiling. LoRA can fall short when the domain shift is deep, for example a language barely present in pretraining, proprietary notation or code, or a change to the model's response distribution itself. Full fine-tuning in turn requires tens of thousands of examples, multiple GPUs and monitoring for forgetting.

### Update cadence and freshness

RAG only needs its index updated, so it can follow changes within minutes to hours. A document is deleted by removing it from the index. LoRA retraining takes hours and full fine-tuning takes days, so updates happen weekly to monthly. Putting frequently changing information into weights means running a training pipeline on every change, which is not practical.

### Citations and access control

RAG can show the chunks it used as sources. Users can verify them, and developers can tell whether a wrong answer came from retrieval or from generation. Filtering by the user's permissions at retrieval time keeps documents the user may not read out of the answer.

Fine-tuning offers neither. Knowledge stored in weights has no source and may be extractable by anyone with the right prompt. If departments are allowed to see different information, each department needs its own model. Removing specific information afterwards is also hard (machine "unlearning" remains an open problem). For uses that involve personal data or permissions, I give this difference more weight than accuracy.

### Cost

| Aspect | RAG | LoRA | Full fine-tuning |
| --- | --- | --- | --- |
| Training cost | None (index build only) | Hours on a single GPU | Days on multiple GPUs |
| Training memory (7B) | n/a | Frozen weights + activations; ~10 GB with QLoRA | ~112 GB for weights, grads, optimizer + activations |
| Inference cost | Retrieval latency + input tokens for $$k$$ passages | None once merged | None |
| Artifact size | Index (scales with corpus) | Adapter, tens of MB | One model copy, tens of GB |
| Data needed | The documents themselves | Hundreds to thousands of (input, output) pairs | Tens of thousands or more |

Which side has the cheaper inference depends on the workload. RAG adds tokens to every request, so it is expensive at high traffic. Fine-tuning recovers its training cost at inference time. With low traffic and frequent document changes, RAG is cheaper.

### Debugging and evaluation

With RAG, a failure can be decomposed and measured. Retrieval is evaluated with IR metrics such as Recall@k or nDCG, and generation is evaluated separately for faithfulness to the provided context. Knowing whether a wrong answer came from not retrieving the right document or from misreading it shows where to fix.

A fine-tuning failure is hard to localize. Whether the cause was data quality, quantity or distribution, the learning rate or the number of epochs can only be found by changing one and retraining. Regression in general ability also has to be measured.

### Access to the model

RAG is always available, including with a closed model behind an API. Methods that change weights are available only when the provider offers a fine-tuning API (whether it uses something like LoRA internally is usually not disclosed). With open-weight models, all three can be used.

### Summary table

| Aspect | RAG | LoRA | Full fine-tuning |
| --- | --- | --- | --- |
| What changes | The input | A few added weights | All weights |
| Knowledge injection | ◎ Immediate, with sources | △ Limited capacity | ○ With continued pretraining on large data |
| Behavior change | △ Re-instruct in every prompt | ○ Works with hundreds to thousands of examples | ◎ Highest ceiling |
| Update speed | Minutes to hours | Hours to days | Days to weeks |
| Citations | Yes | No | No |
| Per-user access control | Filter at retrieval | Separate models | Separate models |
| Forgetting risk | None | Low | High |
| With a closed API model | Yes | Provider-dependent | Provider-dependent |

## Combining them

In practice the three are combined more often than one is chosen.

The most common combination is RAG + LoRA. RAG supplies the facts and LoRA handles behavior: answering only from the given context, saying "I don't know" when the context lacks the answer, attaching citations in the prescribed format, and writing in the company's tone and terminology. Each fixes the failure it is suited to, and their roles do not overlap.

"Say you don't know when it isn't in the context" is easy to teach with LoRA because the behavior branches on a condition that can be read from the input. A pretrained model can already tell whether a passage answers a question, so LoRA only adds the output policy of declining when it does not. The training data can be built mechanically by removing the context from a question and labeling the answer "not in the context".

Saying "I don't know" without any context, when the fact is absent from the model's own knowledge, is different. It depends on a knowledge boundary inside the model that the input does not show. If hard-looking questions are uniformly labeled "I don't know", the model learns to refuse based on how a question looks. If answers to facts the model does not know are mixed into the training data, the model learns to answer fluently without knowing, and hallucination increases (Gekhman et al., 2024). Methods such as R-Tuning (Zhang et al., 2023) therefore have the base model attempt each question first and assign labels by whether it answered correctly. This is hard with LoRA and with full fine-tuning alike, and as with calibration, it can only be checked on a validation set after training.

Fine-tuning can also be designed around RAG. Zhang et al. (2024) call this RAFT (Retrieval-Augmented Fine-Tuning). Each training example consists of a question, a set of documents containing the oracle document plus irrelevant distractors, and a chain-of-thought answer with citations. The model learns to identify the document that contains the answer among several retrieved candidates, ignore the distractors and cite. The approach assumes an imperfect retriever and adapts the generator to it.

Another option leaves the generator unchanged and trains the embedding model or a Cross-Encoder reranker on domain-specific (query, relevant document) pairs. A failure such as "asked in jargon but only got generic documents" is a retrieval problem, and training the generator does not fix it. These models are small, so training is cheap and the return is often high.

## Keeping the bot from answering outside its domain

Consider a chatbot that answers only an e-commerce site's FAQ. It should answer questions about shipping, returns, payment, account registration and product specs, and decline unrelated ones such as "what's the weather today" or "write a sort in Python". No single one of the three methods provides this boundary. It is built by layering checks outside the generator with behavior trained into the generator.

<figure>
  <img src="{{ '/assets/img/scope-gates-en.svg' | relative_url }}" alt="Four layers that stop out-of-scope questions. A query is first labeled in-scope, out-of-scope or harmful by an input classifier, then checked against the FAQ index with a retrieval score threshold, then answered by an LLM trained with LoRA to answer only from context, and finally the answer is verified against the retrieved passages. A query stopped at any layer gets a canned refusal">
  <figcaption>An input classifier and a retrieval score threshold outside the generator stop out-of-scope queries, and LoRA and an output check inside it catch the rest. Every layer has gaps on its own, but the gaps are in different places.</figcaption>
</figure>

The simplest measure is to write "do not answer anything outside the FAQ" in the system prompt, and it works to a degree. On its own, though, the instruction weakens as the conversation gets longer, and it is vulnerable to inputs like "ignore the previous instructions". The prompt is best treated as the first layer, with further checks added outside it.

### A threshold on the retrieval score

RAG's retrieval step can double as a scope check. The query is searched against the FAQ index, and if the top relevance score is below a threshold, the bot returns "no matching FAQ" without generating. A weather question resembles no FAQ entry, so it stops here. No training is needed, only a threshold chosen on validation data.

The weakness is that embedding similarity is only a proxy for relevance. "What is the return policy in general?" embeds close to the company's returns FAQ, but the answer should be the company's policy, not a general one. In the other direction, a legitimate question with unusual wording scores low and is wrongly rejected. A threshold on a Cross-Encoder reranker's score is more stable than one on a Bi-Encoder's score ([earlier article](/blog/en/bi-late-cross-encoder/)). In either case the threshold is set on a validation set containing both in-scope and out-of-scope queries, with over-rejection and leakage measured separately.

### A classifier in front of the generator

For more certainty, a small classifier in front of the generator decides whether each question is in scope. A BERT-style Encoder with a classification head works, and so does [a zero-shot decision model like Jev](/blog/en/jev-system-one/). Labels such as "in scope / out of scope / harmful" are enough. The training data is only (query, label) pairs, which are far cheaper to collect than written answers. Sorting a few hundred queries from real support logs is often sufficient.

A classifier is independent of the generator. It can be run in-house even with a closed API model, and prompt injection does not reach it. Its decision is a probability, so the threshold and calibration can be tuned on a validation set as described in [the article on classifier confidence](/blog/en/classification-confidence/). The rate at which borderline questions like "the button doesn't work" are rejected and the rate at which unrelated questions get through are measured separately, and which one to prioritize is a business decision. For an e-commerce site, turning away a customer who wants to buy usually costs more.

The boundary is also designed here. "How do I wash this sweater?" is not in the FAQ but is about a product. Whether such questions are refused as out of scope or routed to a human is decided by the classifier's label design, not by the generator.

### Train the generator to answer from context only

Only queries that pass both checks reach the generator. Even so, the model may ignore the retrieved passages and answer from its own knowledge. The RAG + LoRA combination from the previous section handles this. LoRA trains the behavior "answer only from the supplied FAQ excerpts, and if they do not contain the answer, reply 'no information available'". The training data includes both questions with their FAQ excerpts attached and questions with the excerpts deliberately removed. The correct output for a question without excerpts is a fixed refusal. As noted earlier, this condition can be read from the input, so LoRA learns it reliably.

Doing the same with the system prompt alone requires spelling out the refusal conditions at length, which adds tokens to every request and is followed less reliably. Moving the behavior into LoRA keeps the prompt short.

### Verify the output before returning it

A final layer can check whether the generated answer is grounded in the retrieved passages. The answer and the passages are passed to an NLI model or a separate LLM, which judges whether the context supports the answer. If it does not, the answer is discarded and replaced with a template. This also stops answers the generator produced from general knowledge. It costs one more inference, so it is best limited to topics such as returns and payments where a wrong answer is expensive.

### Why the layers are stacked

Each layer has gaps on its own. The prompt is weak against long conversations and injection, the retrieval score confuses similarity with relevance, the classifier is unsure about borderline questions, and LoRA breaks on phrasings absent from its training data. The gaps are in different places, though, so stacking the layers greatly reduces what gets through. For an e-commerce FAQ bot, I consider the following a realistic configuration: stop out-of-scope queries with the input classifier and the retrieval threshold, train "never say what isn't in the context" with LoRA, and apply output verification only to topics where a wrong answer is costly.

## How to choose

1. Build with prompts only, and measure. Check how far a system prompt and few-shot examples go. If that is enough, stop there.
2. Sort wrong answers into knowledge failures and behavior failures. They are usually mixed, so look at the ratio.
3. Use RAG for knowledge failures. Build the index and measure retrieval quality on its own. If the right passage is retrieved and the answer is still wrong, fix generation. If it is not retrieved, fix retrieval.
4. Use LoRA for behavior failures. Prepare a few hundred to a few thousand ideal responses and train. With a closed API model, use the provider's fine-tuning API. When combining with RAG, give the training data the form "question with context → answer grounded in the context".
5. Consider full fine-tuning only if that is still not enough, and only when the domain shift is deep, tens of thousands of examples and GPUs are available, and regression in general ability can be measured.

| Situation | Suitable choice |
| --- | --- |
| Answer from internal docs, specs, policies | RAG |
| Information changes daily | RAG |
| Citations required; different users may see different information | RAG |
| Align tone, output format, terminology, refusal policy | LoRA (or the provider's fine-tuning API) |
| Enforce "never say what isn't in the context" | RAG + LoRA (with RAFT-style data) |
| Answer only within one domain (e.g. an e-commerce FAQ) | Stop with an input classifier + retrieval threshold, train the refusal with LoRA |
| Poor retrieval for jargon queries | Fine-tune the embedding model / reranker |
| Support a language or notation barely present in pretraining | Continued pretraining + SFT (full, or LoRA with large $$r$$) |
| High traffic; want to cut inference tokens | Move behavior into LoRA and shorten the prompt |

There are five steps, but steps 1 and 2 decide most of it. Once the wrong answers are sorted, it is clear whether the bot needs documents to read through RAG or behavior trained through LoRA. Full fine-tuning comes up only after both have been tried.

## References

- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.
- Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*.
- Biderman, D., et al. (2024). LoRA Learns Less and Forgets Less. *TMLR 2024*.
- Ovadia, O., et al. (2023). Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs. *arXiv:2312.05934*.
- Zhang, T., et al. (2024). RAFT: Adapting Language Model to Domain Specific RAG. *arXiv:2403.10131*.
- Gekhman, Z., et al. (2024). Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? *EMNLP 2024*.
- Zhang, H., et al. (2023). R-Tuning: Instructing Large Language Models to Say "I Don't Know". *NAACL 2024*.
- Rajbhandari, S., et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. *SC 2020*.
