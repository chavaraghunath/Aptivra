# On-Device, Private, Continual Personalization of Small Language Models and "Second-Brain" Memory Architectures

Scope note: researched 2026-10-18 under a 15-call budget. Most fetches returned arXiv abstracts or HTML tables; where only an abstract was read, this is stated. Claims I could not source are in the Gaps sections, not the findings. Target design assessed: a frozen base model, per-skill LoRA adapters, a learned router, replay, and verifier-gated promotion with rollback.

## 1. Parametric vs non-parametric personalization: weights (LoRA, model editing) or external memory (RAG, graph, episodic)?

### Takeaway
For *facts about the user's world*, the evidence favours external memory. RAG beats unsupervised fine-tuning for injecting both known and new knowledge, often by large margins. Model editing (ROME, MEMIT) degrades general ability under sequential edits; null-space methods such as AlphaEdit reduce the damage but do not remove it. Weights are the right store for *behaviour, style and skills*, not for a growing, mutable, deletable set of personal facts.

### Cited Findings
- **Fine-tuning vs RAG (Ovadia et al., Microsoft, 2023/24).** On MMLU subsets and a current-events task, with Llama2-7B, Mistral-7B and Orca2-7B, "RAG consistently outperforms [unsupervised fine-tuning], both for existing knowledge encountered during training and entirely new knowledge." LLMs "struggle to learn new factual information through unsupervised fine-tuning." — [arXiv 2312.05934](https://arxiv.org/abs/2312.05934)
- **New knowledge (current-events task, accuracy):** Mistral-7B scored 0.481 base, **0.875 with RAG**, 0.504 after fine-tuning on raw text, 0.588 after fine-tuning with paraphrases, and 0.830 for paraphrase fine-tuning plus RAG. Llama2-7B scored 0.353 base and 0.585 with RAG. Fine-tuning on raw text *lowered* it to 0.219, and adding RAG to the fine-tuned model gave only 0.326. Fine-tuning can therefore damage a model's later use of retrieval. — [arXiv 2312.05934v3 HTML, Table 2](https://arxiv.org/html/2312.05934v3)
- **Paraphrase augmentation:** "For all models tested, the accuracy was a monotonically increasing function of the number of paraphrases used"; 10 paraphrases per chunk were used. — [arXiv 2312.05934v3](https://arxiv.org/html/2312.05934v3)
- **Already-seen knowledge (MMLU, Mistral-7B, anatomy 0-shot):** 0.556 base, 0.681 with RAG, 0.570 fine-tuned, 0.659 fine-tuned plus RAG. Fine-tuning gains are typically 0 to 5 points, against 3 to 12 points for RAG. — [arXiv 2312.05934v3, Table 1](https://arxiv.org/html/2312.05934v3)
- **Agent memory systems on LOCOMO (Mem0 paper, Apr 2025), LLM-as-judge J score:**

  | System | J score | p95 total latency |
  |---|---|---|
  | Full-context | 72.90 | 17.1 s |
  | Mem0 with graph (Mem0g) | 68.44 | 2.59 s |
  | Mem0 | 66.88 | 1.44 s |
  | Zep | 65.99 | 2.93 s |
  | Best RAG (k=2, 256-token chunks) | 60.97 | 1.91 s |
  | LangMem | 58.10 | 60.4 s |
  | OpenAI memory | 52.90 | 0.89 s |
  | A-Mem | 48.38 | 4.37 s |

  Context tokens per query: about 1.8k for Mem0 against about 26k for full context. Stored memory per conversation: about 7k tokens for Mem0, about 14k for Mem0g, and more than 600k for Zep. — [arXiv 2504.19413v1](https://arxiv.org/html/2504.19413v1)
- **Caveat on the Mem0 paper:** the authors are Mem0 developers (Chhikara et al.), so this is a vendor benchmark. Full context still wins on accuracy; Mem0's advantages are a 91% lower p95 latency and more than 90% fewer tokens. — [arXiv 2504.19413](https://arxiv.org/abs/2504.19413)
- **Model editing damages general ability (Gu et al., 2024).** The study covered 4 editing methods, 3 LLMs and 8 tasks. "It is challenging for current editing methods to simultaneously improve factuality … and maintain their general abilities." In one case, "a single pass of editing involving less than 1% parameters" sent LLaMA-1 7B to "nearly 0 on all the tasks." The authors' RECT regularizer reduces side effects while keeping more than 94% of editing performance. — [arXiv 2401.04700](https://arxiv.org/html/2401.04700v3)
- **Sequential-edit collapse and AlphaEdit.** As edits accumulate, overfitting erodes knowledge preservation and coherence and eventually causes model collapse. AlphaEdit projects each edit onto the null space of preserved knowledge, adding one line to MEMIT, and reports a 36.7% average improvement over the best baseline on LLaMA3, GPT2-XL and GPT-J (ICLR 2025). — [arXiv 2410.02355](https://arxiv.org/html/2410.02355v4)
- **AlphaEdit's limits.** BetaEdit (IJCAI 2026) argues that AlphaEdit's null space is approximate and so leaks knowledge. A spectral analysis finds that dominant singular directions "are progressively disrupted by repeated edits," which tracks the collapse in editing efficacy and general performance. A NeurIPS 2024 deck reports severe capability loss "even with only hundreds of edits" (secondary source). — [IJCAI 2026 BetaEdit](https://www.ijcai.org/proceedings/2026/0508.pdf); [arXiv 2410.18785](https://arxiv.org/html/2410.18785v1); [NeurIPS 2024 slides](https://nips.cc/media/neurips-2024/Slides/93785.pdf)

### Inferences
- For a "second brain", the primary store for user facts (people, projects, dates, preferences that change) should be non-parametric: an episodic log, extracted fact memory and an optional graph. It is more accurate per Ovadia, inspectable, deletable (see Q5) and costs no training. Adapters should hold *procedural* knowledge: format, tone, tool-use habits and domain skill.
- If facts must go into weights (for offline recall without retrieval, or for latency), train on many paraphrases rather than raw text, and always keep retrieval as a backstop. The Llama2 result above shows that naive fine-tuning can *reduce* RAG effectiveness, so a verifier gate must test "adapter plus RAG", not only "adapter alone".
- Locate-and-edit methods (ROME, MEMIT, AlphaEdit) are a poor fit for an always-learning personal device. Personal facts arrive at hundreds to thousands per year, the range where collapse is reported, and edits are hard to roll back individually. Adapter-scoped updates are reversible by construction (drop the adapter), which matches the target design's rollback.

### Gaps
- I did not fetch the MemGPT/Letta, A-Mem, Zep or GraphRAG primary papers. Their numbers above come only from Mem0's reproduction, which is vendor-run. A-Mem's original paper reports its results on different settings.
- I found no head-to-head of RAG vs LoRA at small (≤1B) scale on *personal* data. Ovadia used 7B models.
- I found no quantified hallucination comparison (parametric vs retrieval) specifically for personal memory.

## 2. Continual learning for LLMs and adapters: what reduces forgetting best at small scale?

### Takeaway
LoRA itself is a strong anti-forgetting regularizer, at the cost of learning less. Among adapter-continual methods, interference-aware subspace methods (InfLoRA, then gated or gradient-projection variants) beat O-LoRA on both accuracy and forgetting on TRACE. A per-skill adapter pool with routing, which is the target design, is architecturally the most forgetting-proof option, because frozen past adapters cannot be overwritten. Its risks move to router drift and adapter-pool growth.

### Cited Findings
- **"LoRA Learns Less and Forgets Less" (Biderman et al., 2024).** In code and math, under both instruction fine-tuning (~100K pairs) and continued pretraining (20B tokens), LoRA "substantially underperforms full finetuning" on the target domain but better preserves source-domain performance. It reduces forgetting more than weight decay or dropout and keeps generations more diverse. Full fine-tuning learns perturbations of 10 to 100 times higher rank than typical LoRA configurations. (Abstract read.) — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- **TRACE benchmark, LLaMA-2-7B-Chat, as reproduced in the GainLoRA paper.** O-LoRA reached average performance (AP) 41.04 with forgetting (FT) 8.05. InfLoRA reached AP 47.67 with FT 2.25. — [GainLoRA, arXiv 2505.15424](https://arxiv.org/pdf/2505.15424)
- **On a standard continual-learning benchmark (LLaMA-2-7B, not TRACE):** published baselines are O-LoRA 76.1, N-LoRA 77.6, GORP 78.6 and InfLoRA 79.6. — [via search excerpt of GainLoRA / related paper](https://arxiv.org/pdf/2505.15424)
- **GORP (ACL 2025)** reports 7.8% and 6.6% performance gains and 3.4% and 3.6% lower forgetting than O-LoRA on T5. — [ACL 2025 GORP](https://aclanthology.org/2025.acl-long.721.pdf)
- **Vision setting (CIFAR100, 10 tasks):** InfLoRA has the lowest forgetting among LoRA-based methods at 4.61%. — [Parameter-Efficient Continual Fine-Tuning survey, arXiv 2504.13822](https://arxiv.org/pdf/2504.13822)
- **TRACE** itself is the standard LLM continual-learning benchmark. It shows that aligned LLMs lose general ability and instruction-following under sequential tasks. — [arXiv 2310.06762](https://ar5iv.labs.arxiv.org/html/2310.06762v1)
- **Replay-free dynamic-memory continual LoRA** is an alternative line of work. — [arXiv 2411.00623](https://arxiv.org/pdf/2411.00623)

### Inferences
- **Assessment of the target design.**
  - **Frozen base plus per-skill LoRA:** strongly supported. Isolation gives near-zero forgetting of the base and of old skills, much as progressive networks do, with cheap growth: each rank-16 adapter is "10s of MB" for a 3B model (see Q6).
  - **Learned router:** this is now the point where forgetting can happen. If the router is retrained on new skills it can forget old routing decisions. Apply replay and EWC-style regularization to the router (a small model, where these methods are cheap and well understood), not to the adapters.
  - **Replay:** a defensible complement. Tables from Q1 suggest replay data should include paraphrases.
  - **Verifier-gated promotion with rollback:** this matches the "stop the line" need, because editing literature shows degradation can be silent and sudden (the collapse to ~0 in Gu et al.). The gate must measure general ability, not only new-skill accuracy.
- **Within-skill updates** (refreshing an existing skill adapter) are the case where O-LoRA, InfLoRA or gradient projection matter. InfLoRA-style interference-free subspaces are the best-evidenced choice among the methods surveyed.
- **Merging** adapters (TIES, DARE) is the main way to bound pool growth. Its forgetting cost at small scale was not measured in the sources gathered (see Gaps).

### Gaps
- I did not obtain primary numbers for TIES or DARE merging of LoRA experts, mixture-of-LoRA routers (e.g., LoRAMoE, MoLE), EWC on LLMs, or progressive networks on LLMs.
- No source quantified forgetting specifically at ≤1B scale. All TRACE and LoRA numbers above are 7B or T5-scale.
- The TRACE O-LoRA vs InfLoRA comparison is a third-party reproduction, not the original InfLoRA paper (which is vision-focused, CVPR 2024).

## 3. Test-time training and learning from documents and books continually

### Takeaway
Test-time training on retrieved neighbours gives large language-modelling gains for small models (117M to 1.3B) with one gradient step per neighbour. That is evidence that "train a little on retrieved personal context at query time" is a viable hybrid between RAG and fine-tuning. For continued pretraining on personal corpora, the Ovadia results imply that raw-text training is weak for fact recall unless the text is heavily paraphrased.

### Cited Findings
- **Test-time training on nearest neighbours (Hardt and Sun, ICLR 2024).** The method retrieves Pile neighbours and fine-tunes "on as few as 20 neighbors, each for only one gradient iteration." With 50 neighbours, bits per byte fall by about 20% across 22 Pile tasks. On pile-github the reduction is more than 60%. Models tested were GPT-2 (117M), GPT-2 Large (774M) and GPT-Neo (1.3B). — [arXiv 2305.18466](https://arxiv.org/pdf/2305.18466)
- **Independent reproduction (2025)** confirms the key claims, including bits per byte down to 51% of the original on GitHub and 68% on EuroParl. — [arXiv 2511.16691](https://arxiv.org/pdf/2511.16691v1)
- **Continued pretraining on raw documents** gives small or negative gains for new facts unless they are paraphrased (Llama2-7B: 0.353 → 0.219 with raw fine-tuning, → 0.392 with paraphrases). — [arXiv 2312.05934v3](https://arxiv.org/html/2312.05934v3)
- **LoRA continued pretraining** on 20B tokens learns less of the new domain than full fine-tuning but forgets less. — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)

### Inferences
- **A "read my books" pipeline:** (a) chunk and index for RAG on day one; (b) during idle and charging periods, synthesize paraphrases and QA pairs and train a document or topic adapter; (c) promote only if the verifier shows gains *with RAG enabled* and no general regression.
- **Ephemeral test-time training** is a cheap fit for small on-device models: per-query adapter deltas trained on the top-k retrieved personal chunks, then discarded. It improves fluency in the user's domain without persisting anything. Its latency cost on phones was not measured in the sources.

### Gaps
- No source measured test-time training latency or energy on a phone.
- I did not fetch newer test-time training or "learning to learn at test time" (TTT-layer) results for 2024–2026 language models, nor synthetic continued-pretraining work such as EntiGraph.

## 4. On-device training feasibility: memory, time, battery, thermal; federated learning

### Takeaway
On-device LLM fine-tuning works but is slow and memory-bound. Phones have fine-tuned models up to about 1.3B parameters, with backprop-free methods needing about 4 to 6.5 GB of RAM, and steps that take minutes to hours. Battery-aware scheduling is required. Google's production practice for keyboard personalization is federated learning with differential privacy and secure aggregation, not per-user on-device fine-tuning of an LLM.

### Cited Findings
- **PocketLLM (2024):** uses derivative-free MeZO optimization. RoBERTa-large and OPT-1.3B were fine-tuned on an OPPO Reno 6 using about 4 GB and 6.5 GB of memory respectively. No energy measurements were reported in the material I saw. — [arXiv 2407.01031](https://arxiv.org/abs/2407.01031)
- **MobileFineTuner (Dec 2025):** a C++ framework that fine-tunes GPT-2, Gemma 3 and Qwen 2.5 on real phones. It uses ZeRO-style parameter offload to disk, gradient accumulation, activation checkpointing and memory-efficient attention. Several configurations that would otherwise run out of memory completed training. Its energy-aware scheduler cut computation frequency from 0.081 to 0.164 hours per step (about 4.9 to 9.8 minutes) when battery fell below 60%. — [arXiv 2512.08211 v1](https://arxiv.org/html/2512.08211v1); [v2](https://arxiv.org/html/2512.08211v2)
- **Memory target:** related work cites less than 1 GB as a reasonable memory target for on-device fine-tuning. — [arXiv 2510.03425](https://arxiv.org/pdf/2510.03425)
- **Gboard federated learning with differential privacy:** DP-FTRL is used without requiring uniform client sampling. More than 20 Gboard language models were deployed with ρ-zCDP guarantees of ρ ∈ (0.2, 2). "All the next word prediction neural network LMs in Gboard now have DP guarantees, and all future launches … will require DP." The earlier ACL 2023 version reported more than 15 models with ρ ∈ (0.3, 2). — [Google Research](https://research.google/pubs/federated-learning-of-gboard-language-models-with-differential-privacy/); [arXiv 2305.18465](https://arxiv.org/pdf/2305.18465)
- **Gboard privacy infrastructure:** distributed differential privacy plus secure aggregation inside an isolation boundary. — [arXiv 2306.14793](https://arxiv.org/html/2306.14793v1)

### Inferences
- For a ~360M to 1B model, LoRA (not full) fine-tuning on a laptop GPU or Apple-silicon unified memory is clearly feasible. On a phone it is feasible only as a background job while charging, and at minutes per step. Plan for nightly batches, not interactive learning, and make non-parametric memory the instant-learning path.
- Federated learning is relevant only if there are many users and a shared model is wanted. For a single-user private "second brain" it adds nothing. Differential privacy matters only if anything leaves the device.

### Gaps
- I found no paper with per-step seconds, joules and temperature for LoRA on a ~1B model on iPhone or Android. MobileFineTuner's full tables were not fetched.
- I did not verify MobiLlama's training numbers (it is mainly an inference model) or any FedLoRA benchmark results.
- I found no Apple-published figures for on-device *training*. Apple trains adapters off-device.

## 5. Learning from user feedback: implicit signals, preference learning, risks and unlearning

### Takeaway
Removing knowledge from weights is unreliable. Benchmarks (TOFU, MUSE) show weak unlearning, and "relearning attacks" restore supposedly forgotten facts with small amounts of data. This is the strongest argument for keeping deletable personal facts out of the weights, and for treating each adapter as a deletable unit that can be retrained from a filtered log. Rollback is the only clean "right to be forgotten" mechanism.

### Cited Findings
- **TOFU** (200 fictitious author profiles): the original paper reported that "none of the baselines it considers show effective unlearning." — [OpenUnlearning summary, arXiv 2506.12618](https://arxiv.org/pdf/2506.12618); [TOFU arXiv 2401.06121](https://arxiv.org/abs/2401.06121)
- **MUSE** evaluates verbatim memorization, knowledge and privacy (membership inference) after removing books and news. — [OpenUnlearning, arXiv 2506.12618](https://arxiv.org/pdf/2506.12618)
- **Targeted relearning attacks:** "a small amount of carefully chosen data can 'jog' the model's memory" after unlearning. — [Jogging the Memory of Unlearned LLMs (review)](https://themoonlight.io/review/jogging-the-memory-of-unlearned-llms-through-targeted-relearning-attacks)
- **Benchmark critique:** many evaluations observe one output channel and declare success when that channel stops leaking. — [arXiv 2510.19422](https://arxiv.org/pdf/2510.19422)
- **Memorizing mistakes:** editing or fine-tuning on wrong data can damage general ability (see Q1, Gu et al.). — [arXiv 2401.04700](https://arxiv.org/html/2401.04700v3)

### Inferences
- **Implicit signals** (accepts, edits, corrections) suit preference pairs (DPO) or binary labels (KTO). They are noisy, though, and a user's correction may itself be wrong. Gate preference updates through the verifier and keep the raw feedback log so an adapter can be rebuilt without a disputed example.
- **Right to be forgotten in the target design:** delete the item from the episodic store, then retrain any adapter whose training set contained it, or roll back to an adapter version predating it. Requires provenance tracking from each training example to each adapter version.
- **Reward hacking and drift:** a learned verifier can itself be gamed by the adapter it judges. Use held-out fixed probes plus a general-ability canary set that the training loop never sees.

### Gaps
- I gathered no sources on on-device DPO or KTO measurements, nor on implicit-feedback reward hacking in personal assistants.
- Exact relearning-attack recovery percentages were not obtained.

## 6. What Apple Intelligence, Gemini Nano and similar products actually do, and deliberately do not do

### Takeaway
Shipping products use exactly the "frozen base + swappable LoRA adapters" pattern, but the adapters are **task adapters trained off-device by the vendor or app developer**. Personalization comes from on-device context and retrieval, not from training on the user's data. Apple states it does not train its foundation models on users' private data or interactions (training data is opt-in only).

### Cited Findings
- **Apple on-device model (2024):** about 3B parameters, mixed 2-bit/4-bit quantization averaging 3.7 bits per weight (3.5 bits per weight possible "without significant quality loss"). On iPhone 15 Pro: about 0.6 ms per prompt token time-to-first-token and 30 tokens/s. — [Apple ML Research](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
- **Apple adapters:** LoRA (rank 16 in the example) on attention, attention-projection and feed-forward layers. A rank-16 adapter in 16-bit "typically require[s] 10s of megabytes," and adapters "can be dynamically loaded, temporarily cached in memory, and swapped." — [Apple ML Research](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
- **Apple on data:** "We never use our users' private personal data or user interactions when training our foundation models." A later update adds an opt-in to help improve Siri and Apple Intelligence. — [Apple ML Research](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
- **Apple 2025 report:** a ~3B on-device model with KV-cache sharing and 2-bit quantization-aware training. A server PT-MoE model runs on Private Cloud Compute. The Swift Foundation Models framework exposes guided generation, constrained tool calling and LoRA adapter fine-tuning to developers. — [arXiv 2507.13575](https://arxiv.org/abs/2507.13575v2)
- **Gemini Nano / AICore (Dec 2023):** AICore "enables Low Rank Adaptation (LoRA) fine tuning with Gemini Nano." Apps supply "small LoRA adapters based on their own training data," loaded at runtime. AICore is "private by design," following Private Compute Core with network isolation. — [Android Developers Blog](https://android-developers.googleblog.com/2023/12/a-new-foundation-for-ai-on-android.html)
- **Google's per-user learning in production** goes through federated learning with differential privacy (Gboard), not local LLM fine-tuning. — [Google Research](https://research.google/pubs/federated-learning-of-gboard-language-models-with-differential-privacy/)

### Inferences
- **What vendors deliberately do not do:** per-user gradient updates to the on-device LLM. Likely reasons, inferred from Q1, Q4 and Q5: unverifiable quality drift, battery and thermal cost, inability to unlearn, and per-OS-release base updates that invalidate adapters.
- **The target design's on-device training of per-skill adapters is therefore *beyond* shipped practice.** That makes the verifier gate and rollback essential rather than optional. Base-model upgrades will require re-training every personal adapter from the retained data log, so the raw log, not the adapter, is the durable asset.
- **Recommended split**, consistent with the evidence:
  - Facts go to episodic, semantic and graph memory with retrieval (Mem0- or Zep-like, fully local).
  - Skills and style go to per-skill LoRA adapters, trained in nightly batches with paraphrase and replay data, InfLoRA-style subspace constraints when refreshing an existing skill, and verifier-gated promotion.
  - Query-time adaptation is optional ephemeral test-time training.
  - Forgetting means deleting from memory and retraining or rolling back the affected adapter.

### Gaps
- I did not confirm Apple's 2025 adapter-toolkit specifics (rank, the requirement to retrain adapters per base-model version) from a primary source in this session.
- I found no current (2025–2026) Google statement on whether Gemini Nano personalization (e.g., Pixel features) trains on user data, or whether AICore's third-party LoRA support is still offered.
- Samsung Galaxy AI, Microsoft Phi Silica and Qualcomm on-device personalization were not researched.
