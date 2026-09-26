---
layout: post
title: "Choosing Between RAG, LoRA and Fine-Tuning for an LLM Chatbot"
description: "When you build a chatbot that answers from your own company's information, RAG, LoRA (low-rank adaptation) and full fine-tuning are usually compared as if they were interchangeable, but they change different things. We pin down each mechanism with a few equations, then compare them by which of two failure modes they fix, 'doesn't know' versus 'behaves wrong', and by cost, update cadence, citations and access control, ending with how to combine them, how to keep the bot inside one domain, and how to choose."
date: 2026-09-26 12:00:00
lang: en
ref: rag-lora-finetuning
permalink: /en/rag-lora-finetuning/
translation: /ja/rag-lora-finetuning/
translation_label: "日本語"
---

Say you want a chatbot that knows your company's products and policies, built on an LLM. The three options that come up first are RAG (Retrieval-Augmented Generation), LoRA (Low-Rank Adaptation) and fine-tuning, and they are usually lined up as "which one gives the best accuracy". But the three are not alternative solutions to the same problem. RAG changes the input at inference time; LoRA and full fine-tuning change the model's weights. When the thing being changed differs, so does the kind of failure that can be fixed.

The conclusion first: if you want the bot to answer with your company's facts, use RAG. If you want it to match your tone and format, use LoRA. Full fine-tuning is for when you have done both and found them insufficient.

## Failures split into "doesn't know" and "behaves wrong"

Point a stock LLM at your company's support queue and the misses fall into two groups.

The first is a knowledge failure. The model cannot answer what was not in its training data: your pricing terms, the feature you shipped last week, internal jargon, the incident report updated yesterday. It tries to answer anyway, and the result is a plausible-sounding hallucination.

The second is a behavior failure. The model knows the content but the shape of the output is off. The tone is too stiff. It adds a preamble when you asked for bare JSON. It paraphrases your internal terms into generic ones. It answers a question it should have declined, or declines one it should have answered.

The two are worth separating because the remedies point in opposite directions. Putting knowledge into weights is hard, and fixing behavior through the input is expensive. RAG addresses the first failure; LoRA and fine-tuning address the second.

## How the three methods work

An LLM models the conditional probability $$p_\theta(y \mid x)$$ of an output $$y$$ given an input token sequence $$x$$, with parameters $$\theta$$. The three methods differ in which part of this expression they touch.

<figure>
  <img src="{{ '/assets/img/rag-lora-ft-where-en.svg' | relative_url }}" alt="Where RAG, LoRA and full fine-tuning intervene. RAG concatenates retrieved passages to the query and feeds a frozen LLM (changes the input). LoRA adds a small low-rank product BA to frozen weights W0 (changes a few added weights). Full fine-tuning updates all parameters θ (changes all weights)">
  <figcaption>Left: RAG leaves the LLM's weights alone and changes the input by appending retrieved passages to the query. Middle: LoRA freezes the original weights and learns only a low-rank delta. Right: full fine-tuning updates every parameter.</figcaption>
</figure>

### RAG changes the input

$$\theta$$ is never touched. Instead, documents relevant to the query $$x$$ are pulled from an external index and concatenated into the prompt.

$$
p(y \mid x) \approx p_\theta\bigl(y \mid x,\ d_1, \ldots, d_k\bigr), \qquad d_1, \ldots, d_k = \mathrm{Retrieve}(x)
$$

As a pipeline: split documents into chunks, embed them, and store them in an index. When a query arrives, embed it in the same space, run nearest-neighbor search, optionally rerank with a Cross-Encoder, and pack the top $$k$$ into the prompt before generating. I covered retriever architectures (Bi-Encoder / Late Interaction / Cross-Encoder) in [an earlier article](/blog/en/bi-late-cross-encoder/) and how to measure retrieval quality in [the article on IR metrics](/blog/en/information-retrieval-metrics/).

Nothing on the LLM side is "learned". Add a document to the index and it is usable in answers from that moment. The cost is paid at inference: retrieval latency plus the input tokens for the top $$k$$ passages, on every request.

### Full fine-tuning changes all the weights

Take pairs of (input, desired output) and update every parameter by gradient descent. In the chatbot setting this usually means SFT (Supervised Fine-Tuning) on top of a pretrained model.

The problem is cost. Running an Adam-style optimizer in mixed precision needs about 16 bytes per parameter for weights, gradients and optimizer state (Rajbhandari et al., 2020). For a 7B model that is roughly 112 GB, before activations. It does not fit on one GPU; distributed training across several is the baseline.

The other problem is catastrophic forgetting. Moving every weight on your own data can degrade the general abilities acquired in pretraining: reasoning, other languages, common sense. After fine-tuning you need to check not only accuracy on the target task but also for regressions on general-capability benchmarks.

### LoRA changes a few added weights

LoRA (Hu et al., 2021) starts from the hypothesis that the weight delta $$\Delta W$$ produced by fine-tuning is effectively low-rank. It therefore freezes the original weight $$W_0$$ and expresses the delta as a product of two small matrices.

$$
h = W_0 x + \frac{\alpha}{r} B A x, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)
$$

Only $$A$$ and $$B$$ are trained ($$\alpha$$ is a scaling constant).

<figure>
  <img src="{{ '/assets/img/lora-decomposition-en.svg' | relative_url }}" alt="LoRA low-rank decomposition. A frozen d×k weight W0 is summed with the product of a d×r matrix B and an r×k matrix A. Trainable parameters drop from dk to r(d+k); with d=k=4096 and r=8, from about 16.8M to about 65K (about 0.4%)">
  <figcaption>The original weight W₀ is frozen; only the delta ΔW = BA is learned. With d = k = 4096 and r = 8, trainable parameters per layer drop from about 16.8M to about 65K.</figcaption>
</figure>

Trainable parameters per layer drop from $$dk$$ to $$r(d + k)$$. With $$d = k = 4096$$ and $$r = 8$$ that is 0.4%, and around 0.3% across a 7B-class model. Most of the memory goes to the frozen weights and activations.

Because $$B$$ is initialized to zero, training starts from exactly the original model's outputs. After training, $$W_0 + \frac{\alpha}{r} BA$$ can be merged into a single matrix, so inference costs nothing extra. Left unmerged, one base model can serve several adapters (tens of MB each) swapped in and out, so per-customer or per-task adapters share a single copy of the base weights.

A notable variant is QLoRA (Dettmers et al., 2023). It quantizes the frozen $$W_0$$ to 4 bits in memory and trains only the LoRA adapters in 16 bits, showing that a 65B model can be fine-tuned on a single 48 GB GPU. The same paper reports that attaching LoRA to all linear layers matters more than increasing $$r$$.

What people call "low-rank training" is LoRA and its descendants (QLoRA, DoRA and so on). LoRA stands in for the family below.

## Comparison

### For knowledge, RAG

The knowledge lives in the index; add one document and it is answerable immediately. The model only has to "read and answer", an ability it acquired thoroughly in pretraining.

Injecting knowledge by fine-tuning is, counterintuitively, hard. Ovadia et al. (2023) report that RAG consistently beats unsupervised fine-tuning on both existing and new knowledge. The same experiments found that getting a model to learn new facts through fine-tuning requires showing the same fact repeatedly in multiple paraphrases. In pretraining, one fact appears across the web in hundreds of surface forms. A fact that appears once in a few hundred Q&A pairs is too weak a gradient signal.

LoRA is at a further disadvantage. Biderman et al. (2024) ran continued pretraining on code and math and found that LoRA learns substantially less than full fine-tuning. The rank of the $$\Delta W$$ produced by full fine-tuning was 10–100× the $$r$$ typically used in LoRA. The low-rank constraint simply lacks capacity for large amounts of new knowledge. The same paper also reports that LoRA forgets less of the original abilities; learning less and forgetting less are two sides of one coin.

So fine-tuning for knowledge does not work at the scale of "a few hundred Q&A pairs"; it becomes a matter of continued pretraining on large volumes of domain text. If you want a chatbot to answer with your company's facts, start with RAG.

### For behavior, LoRA and fine-tuning

Tone, output format, use of terminology, refusal policy, answer length. These "how to answer" properties are consistent patterns that are easy to learn from examples. In the original LoRA paper (Hu et al., 2021), LoRA with $$r$$ between 1 and 8 matched full fine-tuning on GLUE and E2E NLG. Biderman et al. (2024) likewise report that while the gap is large in continued pretraining, the gap between LoRA and full fine-tuning is small for instruction tuning. A few hundred to a few thousand (input, ideal output) pairs are often enough to see the effect.

Trying to fix behavior with RAG alone means piling instructions into the system prompt. Short instructions work well, but the longer they get, the more likely parts are ignored, and you pay those tokens on every request. Adding few-shot examples makes it longer still. That is what "fixing behavior through the input is expensive" means.

Full fine-tuning has the highest ceiling. When the domain shift is deep, say a language barely present in pretraining, proprietary notation or code, or a fundamental change to the model's response distribution, LoRA can fall short. But that requires tens of thousands of examples, multiple GPUs, and monitoring for forgetting.

### Update cadence and freshness

RAG only needs the index updated, so it can track changes within minutes to hours. Deleting a document means removing it from the index. LoRA retraining takes hours and full fine-tuning takes days, so updates land on a weekly-to-monthly cadence. Designing frequently changing information into weights means running a training pipeline on every change, which is not realistic.

### Citations and access control

RAG can present the chunks it used as sources. Users can verify them, and developers can tell whether a wrong answer came from retrieval or generation. Filtering by the user's permissions at retrieval time keeps documents the user cannot read out of the answer entirely.

Fine-tuning has neither. Knowledge baked into weights has no source, and may be extractable by anyone with the right prompt. If different departments may see different information, the only option is a separate model per department. Removing specific information afterwards is also hard (machine "unlearning" remains an open problem). Where personal data or permissions are involved, I weigh this difference more heavily than accuracy.

### Cost

| Aspect | RAG | LoRA | Full fine-tuning |
| --- | --- | --- | --- |
| Training cost | None (index build only) | Hours on a single GPU | Days on multiple GPUs |
| Training memory (7B) | n/a | Frozen weights + activations; ~10 GB with QLoRA | ~112 GB for weights, grads, optimizer + activations |
| Inference cost | Retrieval latency + input tokens for $$k$$ passages | None once merged | None |
| Artifact size | Index (scales with corpus) | Adapter, tens of MB | One model copy, tens of GB |
| Data needed | The documents themselves | Hundreds to thousands of (input, output) pairs | Tens of thousands or more |

How inference cost reads depends on the workload. RAG adds tokens to every request, so it is heavy at high traffic. Fine-tuning recoups its training cost at inference. Conversely, with low traffic and frequent document changes, RAG is cheaper.

### Debugging and evaluation

Being able to decompose a failure is RAG's strength. Retrieval is evaluated with IR metrics such as Recall@k or nDCG; generation is evaluated for faithfulness to the provided context. Once you know whether a wrong answer means "didn't retrieve the right document" or "retrieved it but misread it", you know where to fix.

A fine-tuning failure is end-to-end and hard to localize. Whether data quality, quantity, distribution, learning rate or epoch count was the cause can only be determined by changing it and retraining. On top of that you must measure regression in general ability.

### Access to the model

With a closed model behind an API, RAG is always available. Weight-changing methods are possible only if the provider offers a fine-tuning API (whether it uses something LoRA-like internally is usually not disclosed). With open-weight models all three are options.

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

In practice this is not a three-way choice; combinations are the norm.

The most common one is RAG + LoRA. RAG supplies the facts, and LoRA handles behavior: "answer only from the given context", "say you don't know when it isn't there", "attach citations in the prescribed format", "write in our tone and terminology". Each fixes the failure it is good at, with no overlap of roles.

LoRA can be taught "say you don't know when it isn't in the context" because that behavior branches on a condition visible in the input. A pretrained model can already read whether a passage answers a question; all LoRA adds is the output policy "if it doesn't, decline". The training data is also mechanical to build: take a question, drop or swap its context, and label the answer "not in the context". Saying "I don't know" without any context, when the fact is simply absent from the model's own knowledge, is a different matter, because it depends on a knowledge boundary inside the model that the input does not reveal. Labeling hard-looking questions "I don't know" teaches the model to refuse based on how a question looks. Mixing answers to facts the model does not know into the training data teaches it to answer fluently anyway, and hallucination goes up (Gekhman et al., 2024). Methods such as R-Tuning (Zhang et al., 2023) first have the base model attempt each question and assign the label by whether it answered correctly. That is hard regardless of LoRA versus full fine-tuning, and, as with calibration, can only be verified on a held-out set after training.

There is also fine-tuning that assumes RAG from the start. Zhang et al. (2024) named it RAFT (Retrieval-Augmented Fine-Tuning). Training examples take the form (question, documents including the oracle plus irrelevant distractors, a chain-of-thought answer with citations), teaching the model to pick out the document that contains the answer from several retrieved candidates, ignore the distractors, and cite. It starts from the premise that the retriever is imperfect and adapts the generator to that.

You can also leave the generator alone. Train the embedding model or a Cross-Encoder reranker on domain-specific (query, relevant document) pairs instead. The failure "asked in jargon but only got generic documents" is a retrieval problem, and no amount of training on the generator will fix it. These models are small, so training is cheap and the return on investment is often high.

## Keeping the bot from answering outside its domain

Consider a chatbot that answers only an e-commerce site's FAQ: shipping, returns, payment, account registration, product specs. It should answer those and decline unrelated questions such as "what's the weather today" or "write a sort in Python". This boundary is not something any one of the three methods provides. It is built by stacking checks that sit outside the generator with behavior trained into the generator itself.

<figure>
  <img src="{{ '/assets/img/scope-gates-en.svg' | relative_url }}" alt="Four layers that stop out-of-scope questions. A query is first labeled in-scope, out-of-scope or harmful by an input classifier, then checked against the FAQ index with a retrieval score threshold, then answered by an LLM trained with LoRA to answer only from context, and finally the answer is verified against the retrieved passages. A query stopped at any layer gets a canned refusal">
  <figcaption>An input classifier and a retrieval score threshold outside the generator stop out-of-scope queries; LoRA and an output check inside catch the rest. Every layer has holes, but in different places.</figcaption>
</figure>

The easiest thing is to write "do not answer anything outside the FAQ" in the system prompt. It works to a degree. Relied on alone, though, the instruction fades as the conversation grows, and it is weak against inputs like "ignore the previous instructions". Treat the prompt as the first layer and add checks outside it.

### A gate built from the retrieval score

RAG already has a gatekeeper. Search the FAQ index with the query, and if the top relevance score is below a threshold, return "no matching FAQ" without generating. A weather question resembles no FAQ entry, so it stops here. No training is needed; you only set the threshold on validation data.

The weakness is that embedding similarity is only a proxy for relevance. "What is the return policy in general?" embeds close to your returns FAQ, but the right answer is your policy, not a generality. Conversely, a legitimate question phrased unusually scores low and is wrongly rejected. Thresholds on a Cross-Encoder reranker's score are more stable than on a Bi-Encoder's ([earlier article](/blog/en/bi-late-cross-encoder/)). Either way, set the threshold on a validation set containing both in-scope and out-of-scope queries, measuring over-rejection and leakage separately.

### A classifier in front of the generator

For one more layer of certainty, put a small classifier in front of the generator that decides "is this question in scope?". A BERT-style Encoder with a classification head works, as does [a zero-shot decision model like Jev](/blog/en/jev-system-one/). Labels such as "in scope / out of scope / harmful" suffice. Training data is just (query, label) pairs, so it is orders of magnitude cheaper to collect than written answers. Sorting a few hundred queries from real support logs is often enough.

The advantage of a classifier is that it is independent of the generator. You can run your own even with a closed API model, and prompt injection never reaches it. Its decision comes back as a probability, so, as in [the article on classifier confidence](/blog/en/classification-confidence/), you can tune the threshold and calibration on a validation set. Measure the rate at which borderline questions like "the button doesn't work" get rejected separately from the rate at which unrelated questions get through, and let the business decide which matters more. For an e-commerce site, turning away a customer who wants to buy usually costs more.

The boundary itself is designed here too. "How do I wash this sweater?" is not in the FAQ but is about a product. Whether such questions are refused as out of scope or routed to a human queue is a question of the classifier's label design, not of the generator.

### Train the generator to answer from context only

Only queries that pass both gates reach the generator. Even then, the model may ignore the retrieved passages and answer from its own knowledge. This is where the RAG + LoRA pairing from the previous section applies. Train with LoRA the behavior "answer only from the supplied FAQ excerpts, and if the excerpts do not contain the answer, reply 'no information available'". The training data consists of questions with their FAQ excerpts attached and questions with the excerpts deliberately removed. The correct output for the latter is a fixed refusal, and since, as noted earlier, this is a condition visible in the input, LoRA learns it reliably.

Doing this with the system prompt alone means spelling out the refusal conditions at length, which adds tokens to every request and is followed less reliably. Moving it into LoRA keeps the prompt short.

### Verify the output before returning it

Finally, you can add a layer that checks whether the generated answer is grounded in the retrieved passages. Pass the answer and the passages to an NLI model or a separate LLM and ask "does the context support this answer?". If not, discard the answer and substitute a template. This is the last net for cases where the generator slipped through and answered from general knowledge. It costs one more inference, so reserve it for topics such as returns and payments where a wrong answer is expensive.

### Why the layers are stacked

Each layer alone has holes. The prompt is weak against long conversations and injection, the retrieval score confuses similarity with relevance, the classifier hesitates on borderline questions, and LoRA breaks on phrasings absent from its training data. But the holes are in different places, so stacking them sharply reduces what gets through. For an e-commerce FAQ bot, a realistic configuration is: stop out-of-scope queries with the input classifier and the retrieval threshold, train "never say what isn't in the context" with LoRA, and apply output verification only to the topics where a wrong answer is costly.

## How to choose

1. Build with prompts only, and measure. Find out how far a system prompt and few-shot examples get you. If that is enough, stop.
2. Sort wrong answers into knowledge failures and behavior failures. They are usually mixed; look at the ratio.
3. Knowledge failures: RAG. Build the index and measure retrieval quality on its own. If the right passage is retrieved but the answer is wrong, fix the generation side; if not retrieved, fix retrieval.
4. Behavior failures: LoRA. Prepare a few hundred to a few thousand ideal responses and train. With a closed API model, use the provider's fine-tuning API. If combined with RAG, make the training data "question with context → answer grounded in the context" as well.
5. Only then, full fine-tuning. Only when the domain shift is deep, you have tens of thousands of examples and GPUs, and you have a setup to measure regression in general ability.

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

Five steps, but in practice steps 1 and 2 decide almost everything. Once the wrong answers are sorted, you can already see whether the bot needs something to read or needs its manners trained. The decision to rebuild it only comes up after you have tried both.

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
