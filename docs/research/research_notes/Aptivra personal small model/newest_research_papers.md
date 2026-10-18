# Newest Research Papers (late 2024 - Oct 2026) for a Tiny, Private, Continually Learning Personal Model

Scope note: this file covers recent papers that general surveys may have missed. It is aimed at a 100M-1B on-device model that learns a user's world, reasons, uses tools and improves itself. Research date: 2026-10-18.

Verification legend (applies to every entry):
- **[V-abs]**: numbers read directly from the arXiv abstract page in this session.
- **[V-snip]**: numbers from search-engine snippets of the primary source (arXiv, proceedings or official slides). Not read from the full paper tables.
- **[SECONDARY]**: numbers from a secondary write-up only. Treat as unverified.
- **[PRIOR]**: the citation is from the researcher's prior knowledge and was not re-fetched this session. Check before relying on it.

Peer-review status is marked **PEER-REVIEWED** (venue named) or **PREPRINT**.

---

## Q1. Continual and lifelong learning for LLMs (2025-2026): anti-forgetting, self-evolving agents, experience learning, memory-augmented CL, sparse memory finetuning, benchmarks

### Takeaway
The most actionable new idea for a sub-1B personal model is **sparse memory finetuning (SMF)**. It writes new facts only into memory-layer slots that the new data activates far more than pretraining data does. On NaturalQuestions it cut forgetting from 89% (full finetuning) and 71% (LoRA) to 11% at equal knowledge gain, and an April 2026 follow-up retrofits it onto Qwen-2.5-0.5B on consumer hardware. Work on self-evolving agents in 2026 is splitting in two. One branch limits weight drift. The other keeps weights frozen and curates a bounded, poisoning-resistant experience memory, which is the right shape for a device with tight RAM and energy limits.

### Cited Findings

**Sparse Memory Finetuning, Lin et al. (FAIR at Meta + UC Berkeley), arXiv 2510.15103, 16 Oct 2025, PREPRINT [V-abs]**
- Built on memory-layer models (Berges et al., 2024). It updates only the memory slots that are "strongly activated by new knowledge relative to their use on pretraining data", a TF-IDF-like ranking. — [arXiv 2510.15103](https://arxiv.org/abs/2510.15103)
- Key result: at the same level of new-knowledge acquisition, NaturalQuestions F1 drops **89% after full finetuning, 71% with LoRA, and 11% with sparse memory finetuning**. — [arXiv 2510.15103](https://arxiv.org/abs/2510.15103)
- The abstract page does not state model sizes or code availability. — [arXiv 2510.15103](https://arxiv.org/abs/2510.15103)

**Improving Sparse Memory Finetuning, Goyal, Kanchi, Shah, Gupta, arXiv 2604.05248, 6 Apr 2026, PREPRINT [V-abs]**
- An open-source pipeline that **retrofits sparse memory modules onto pretrained Qwen-2.5-0.5B** so it can learn continually "on consumer hardware". It adds KL-divergence-based slot selection that prioritises tokens that are informationally surprising relative to a background distribution. — [arXiv 2604.05248](https://arxiv.org/abs/2604.05248)
- Reports new factual knowledge with "minimal forgetting of held-out capabilities". The abstract gives no other numbers. A search snippet adds that full finetuning degrades GSM8K reasoning, while the method shows minimal forgetting on TriviaQA and SimpleQA recovery [V-snip]. The arXiv page shows no repository link. — [arXiv 2604.05248](https://arxiv.org/abs/2604.05248); [alphaXiv](https://www.alphaxiv.org/abs/2604.05248)

**Self-Adapting Language Models (SEAL), Zweiger et al. (MIT), arXiv 2506.10943, Jun 2025, PEER-REVIEWED (NeurIPS 2025)**
- The model writes its own "self-edits": synthetic finetuning data plus update directives. An inner SFT loop applies each self-edit to the weights, and an outer RL loop rewards self-edits that improve downstream performance. — [arXiv 2506.10943](https://arxiv.org/pdf/2506.10943); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118690)
- Knowledge incorporation (SQuAD no-context QA) on Qwen2.5-7B: **33.5% to 47.0%**, beating GPT-4.1-generated synthetic data at 46.3%. Few-shot ARC on **Llama-3.2-1B-Instruct**: a 72.5% success rate, against 20% without RL and 0% for ICL alone **[SECONDARY]**. — [sltcreative summary](https://www.sltcreative.com/self-adapting-language-models); [CO/AI write-up](https://getcoai.com/news/mit-researchers-create-ai-that-teaches-itself-by-generating-training-data/)
- The authors name catastrophic forgetting across sequential self-edits as an open limitation. — [arXiv 2506.10943](https://arxiv.org/pdf/2506.10943)

**Nested Learning / HOPE, Behrouz, Razaviyayn, Zhong, Mirrokni (Google), arXiv 2512.24695 (posted 31 Dec 2025). The Google blog says it was published at NeurIPS 2025 — PEER-REVIEWED per the blog [V-snip]**
- Treats architecture and optimiser as nested optimisation problems, each with its own "context flow" and update frequency. **HOPE** is a self-modifying recurrent architecture augmented with **Continuum Memory System (CMS)** blocks, which are memories updated at multiple frequencies. It targets catastrophic forgetting. — [Google Research blog, 7 Nov 2025](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/?issue=64); [arXiv 2512.24695](https://arxiv.org/abs/2512.24695v1)
- At the 1.3B scale, HOPE is reported to reach **57.23% average accuracy on reasoning tasks and 15.11 Wiki perplexity**, the best among the compared recurrent models and Transformers **[SECONDARY]**. — [arxiviq summary](https://arxiviq.substack.com/p/nested-learning-the-illusion-of-deep)
- No official code release was found in this session.

**Continually self-improving AI, Zitong Yang, PhD thesis, arXiv 2603.18073, 18 Mar 2026, PREPRINT (thesis) [V-abs]**
- Three chapters: (1) synthetic-data expansion of small corpora so a model can learn from limited source material (the EntiGraph line); (2) self-generated synthetic data to bootstrap pretraining without distilling from an instruction-tuned LM; (3) test-time search over learning algorithms. The abstract gives no numbers and no code link. — [arXiv 2603.18073](https://arxiv.org/abs/2603.18073)

**Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory, Wu, Ding, Huang, Zhao, arXiv 2606.25115, 23 Jun 2026, PREPRINT [V-abs]**
- Keeps weights frozen and curates the experience memory with one score, *net value per byte* (value minus harm). The score drives three decisions: KEEP (evict under RAM and energy budgets), SHARE (send an insight only when its value exceeds the uplink cost) and TRUST (gate peer entries by provenance). — [arXiv 2606.25115](https://arxiv.org/abs/2606.25115)
- On a Jetson testbed (two robot-arm nodes plus a hub) it gives **2.7x less memory, 2.4x less uplink, and memory-injection success down from 0.75 to 0**, with higher accuracy on cases with poisoned or stale memory. No model size or code is given on the abstract page. — [arXiv 2606.25115](https://arxiv.org/abs/2606.25115)

**Other 2026 continual-learning preprints (abstract or snippet level only; not verified in depth)**
- *RL Forgets! Towards Continual Policy Optimization* (arXiv 2607.04364): a replay-free parameter-movement regulariser. Reports **13.7% less forgetting on Qwen3-VL-8B** [V-snip]. — [arXiv 2607.04364](https://arxiv.org/html/2607.04364v1)
- *Towards Continual Learning: Mitigating Forgetting With Subspace Tuning* (arXiv 2610.22293, Oct 2026): ties forgetting to the magnitude and direction of weight movement [V-snip]. — [arXiv 2610.22293](https://arxiv.org/abs/2610.22293)
- *Do Self-Evolving Agents Forget?* (arXiv 2605.09315), *SEAGym* (arXiv 2606.17546, an evaluation environment for self-evolving agents), *ContinualSkillBench* (arXiv 2608.03874), *Continual Learning in Transition* (arXiv 2608.06216, a survey) and *Building Self-Evolving Agents via Experience-Driven Lifelong Learning: A Framework and Benchmark* (arXiv 2508.19005). Only titles or reference lists were seen. — [2605.09315](https://arxiv.org/pdf/2605.09315); [2606.17546](https://arxiv.org/pdf/2606.17546); [2608.03874](https://arxiv.org/pdf/2608.03874); [2608.06216](https://arxiv.org/pdf/2608.06216); [2508.19005](https://arxiv.org/pdf/2508.19005)

### Inferences
- For a 100M-1B model, a reasonable weight-level learning path is: add product-key memory layers at pretraining time (cheaper than retrofitting), then write the user's facts with SMF-style slot selection. Updates touch only a small number of slots, so they are cheap on device, auditable, and reversible per slot. That suits privacy deletion ("forget this").
- SEAL-style self-edits are the natural *data generator* for SMF: the model rewrites a user's note into training items, and SMF absorbs them. No paper combining the two was found.
- "Forget to Improve" is the closest published match to Aptivra's deployment constraints (bounded RAM, poisoning, no weight updates). It is an argument for a two-tier design: a curated episodic memory now, and consolidation into weights later.

### Gaps
- Model sizes, code links and the full forgetting tables for SMF (2510.15103) were not visible on the abstract page. Read the full PDF before relying on them.
- No official HOPE code or weights were found, and the 1.3B numbers are from a secondary summary only.
- The essay "Welcome to the Era of Experience" (Silver and Sutton, 2025) could not be retrieved; the search returned irrelevant results. It is cited from prior knowledge only and should not be used without checking.
- No 2026 paper was found that demonstrates continual weight-level learning running entirely on a phone-class device.

---

## Q2. Test-time learning and memory architectures: Titans successors (ATLAS, MIRAS), TTT-E2E, Nested Learning/HOPE, memory layers, HRM/TRM

### Takeaway
2025-2026 turned "memory" into *learning at inference time*. Titans led to MIRAS (a unifying design space), then ATLAS (ICML 2026), then Nested Learning/HOPE. TTT-E2E (Dec 2025) shows that a plain sliding-window Transformer updating its own weights at test time scales with context like full attention, at constant latency. For reasoning, tiny recursive networks (TRM, 7M parameters) beat HRM and many LLMs on ARC-AGI, but ARC Prize's audit of HRM shows that the gains come mostly from iterative refinement and task-specific training, not from general reasoning.

### Cited Findings

**Titans, Behrouz et al. (Google), arXiv 2501.00663, Jan 2025 [PRIOR]**: a neural long-term memory module updated at test time by a "surprise" (gradient) signal. It is the base that MIRAS, ATLAS and HOPE build on. — [arXiv 2501.00663](https://arxiv.org/abs/2501.00663)

**MIRAS: "It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization", Behrouz, Razaviyayn, Zhong, Mirrokni, arXiv 2504.13173, 17 Apr 2025, PREPRINT [V-snip]**
- Frames Transformers, Titans and linear RNNs as associative memories defined by four choices: memory architecture, **attentional bias** objective, **retention gate** (forgetting recast as retention regularisation) and memory learning algorithm. It introduces three models: **Moneta, Yaad and Memora**. — [arXiv 2504.13173](https://arxiv.org/abs/2504.13173v1); [HF papers](https://huggingface.co/papers/2504.13173)
- The abstract claims that some instances beat Transformers and linear RNNs on language modelling, commonsense and recall-intensive tasks. No independent numbers were found. — [arXiv 2504.13173](https://arxiv.org/abs/2504.13173v1)

**ATLAS: Learning to Optimally Memorize the Context at Test Time, Behrouz et al., arXiv 2505.23735, 29 May 2025, PEER-REVIEWED (ICML 2026, PMLR v306) [V-snip]**
- Optimises memory over a window of current *and past* tokens (the "Omega rule") rather than only the latest token. This addresses limited capacity, online-only updates and weak management of fixed-size memory. — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/65037); [PMLR](https://proceedings.mlr.press/v306/behrouz26b.html)
- Reaches **about 80% accuracy at 10M-token context on BABILong** and improves on Titans. — [arXiv 2505.23735](https://arxiv.org/pdf/2505.23735); [HF papers](https://huggingface.co/papers/2505.23735)

**TTT-E2E: End-to-End Test-Time Training for Long Context, Tandon, Dalal, ..., Yu Sun (Stanford/NVIDIA/UC Berkeley), arXiv 2512.23675, 29 Dec 2025, PREPRINT [V-snip]**
- A standard Transformer with sliding-window attention that keeps training its weights on the input at inference time. Meta-learning at training time sets the initialisation for that adaptation, so long context becomes a continual-learning problem rather than an architecture problem. — [arXiv 2512.23675](https://arxiv.org/abs/2512.23675v1)
- With **3B models trained on 164B tokens**, it scales with context length like full attention, where Mamba 2 and Gated DeltaNet do not. It has **constant inference latency** and is **2.7x faster than full attention at 128K on an H100**. The authors state that the code is public. — [arXiv 2512.23675](https://arxiv.org/pdf/2512.23675)

**Memory layers**
- *Memory Layers at Scale*, Berges et al. (Meta), arXiv 2412.09764, Dec 2024 [PRIOR]: trainable product-key key-value memory layers that replace FFNs. This is the substrate SMF depends on. — [arXiv 2412.09764](https://arxiv.org/abs/2412.09764)
- SMF and Improving SMF: see Q1.

**Hierarchical Reasoning Model (HRM), Wang et al. (Sapient), arXiv 2506.21734, Jun 2025, PREPRINT [V-snip]**
- **27M parameters**, about 1,000 training examples. Claims **40.3% on ARC-AGI-1**. — [arXiv 2506.21734](https://arxiv.org/pdf/2506.21734)
- **ARC Prize independent audit (15 Aug 2025):** **32% on ARC-AGI-1 and 2% on ARC-AGI-2** on the semi-private sets. The hierarchical design had "minimal performance impact" compared with a similarly sized transformer. The under-documented **outer-loop refinement** drove the gains. Performance came mostly from training on the evaluation tasks, with limited cross-task transfer. — [ARC Prize blog](https://arcprize.org/blog/hrm-analysis)

**Tiny Recursive Model (TRM): "Less is More: Recursive Reasoning with Tiny Networks", Alexia Jolicoeur-Martineau (Samsung SAIL Montreal), arXiv 2510.04871, 6 Oct 2025, PREPRINT [V-snip]**
- A single **2-layer, 7M-parameter** network that recursively refines a latent state and an answer. It is simpler than HRM and generalises better. — [arXiv 2510.04871](https://arxiv.org/abs/2510.04871)
- **45% on ARC-AGI-1 and 8% on ARC-AGI-2** (44.6% and 7.8%, against HRM's 40.3% and 5.0%). The paper says this beats DeepSeek-R1, o3-mini and Gemini 2.5 Pro with under 0.01% of their parameters. — [arXiv 2510.04871](https://arxiv.org/html/2510.04871v1)
- Code was released by Samsung SAIL Montreal. The official repo was later archived as read-only, and mirrors exist. — [README mirror](https://cdn.jsdelivr.net/gh/samsungsailmontreal/tinyrecursivemodels@main/README.md); [GitHub mirror](https://github.com/mvanhorn/TinyRecursiveModels)

### Inferences
- TTT-E2E is the most deployable test-time-learning result for Aptivra. It needs no exotic layer, only sliding-window attention plus test-time gradient steps, and its latency is constant. The open question is whether the meta-learned initialisation transfers to models of 1B parameters and below. The paper reports 3B.
- TRM and HRM are puzzle solvers trained per task family. They are not language models. The reusable lesson is the *recursion / outer-loop refinement* mechanism, which also appears in Ouro (Q3). It is not evidence that a 7M model can be a general assistant.
- The Titans, MIRAS, ATLAS and HOPE line comes from one Google group. Independent replications at small scale were not found, so treat claims of beating Transformers as unreplicated.

### Gaps
- ATLAS and MIRAS: no confirmed official code or weights. Exact parameter scales were not extracted (the earlier Titans and ATLAS work reportedly trained at roughly 170M-1.3B, but this was not verified this session).
- TRM numbers on Sudoku-Extreme and Maze-Hard were not re-verified this session.
- No head-to-head comparison of ATLAS, HOPE and TTT-E2E at the same scale was found.

---

## Q3. Small-model reasoning breakthroughs (2025-2026): looped/recursive transformers, latent reasoning, diffusion LMs, RL on tiny models

### Takeaway
The strongest evidence for sub-3B reasoning is **looped (weight-shared, recurrent-depth) pretraining**. Ouro-1.4B and Ouro-2.6B (Oct 2025) reportedly match dense models up to 12B, and loop count acts as a third scaling axis. On the RL side, **one-shot RLVR** (NeurIPS 2025) lifts Qwen2.5-Math-1.5B from 36.0% to 73.6% on MATH500 with a single training example. Diffusion-LM reasoning with RL works (d1, DCoLT), but only at 8B so far.

### Cited Findings

**Ouro: Scaling Latent Reasoning via Looped Language Models, Rui-Jie Zhu et al. (including Yoshua Bengio), arXiv 2510.25741, Oct 2025, PREPRINT [V-snip]**
- Looped LMs pretrained on **7.7T tokens**, with an entropy-regularised objective for learned depth allocation (adaptive exit). — [arXiv 2510.25741](https://arxiv.org/abs/2510.25741); [HF papers](https://huggingface.co/papers/2510.25741)
- **The 1.4B and 2.6B models match up to 12B state-of-the-art LLMs** on a wide set of benchmarks. Controlled experiments attribute the gain to better *knowledge manipulation*, not more knowledge capacity. — [arXiv 2510.25741](https://arxiv.org/html/2510.25741v1)
- Specific numbers **[SECONDARY]**: Ouro-2.6B scores 55.73 on MMLU-Pro (Qwen3-8B: 53.72), and Ouro-1.4B scores 82.40 on MATH500 (Qwen3-4B: 59.60). Thinking variants with 4 recurrent steps are competitive with 4B and 8B models. — [arxiviq summary](https://arxiviq.substack.com/p/scaling-latent-reasoning-via-looped)
- Project page: ouro-llm.github.io. Weights are reported as released on the Hugging Face paper page; not verified this session. — [HF papers](https://huggingface.co/papers/2510.25741)

**Coconut: Training LLMs to Reason in a Continuous Latent Space, Hao et al. (Meta), arXiv 2412.06769, Dec 2024 [PRIOR]**: feeds the last hidden state back as the next input embedding ("continuous thought"), enabling breadth-first-like search in latent space. — [arXiv 2412.06769](https://arxiv.org/abs/2412.06769)

**One-shot RLVR: Reinforcement Learning for Reasoning in LLMs with One Training Example, Yiping Wang et al., arXiv 2504.20571, Apr 2025, PEER-REVIEWED (NeurIPS 2025) [V-snip]**
- On Qwen2.5-Math-1.5B, **MATH500 rises from 36.0% to 73.6% with one example** (74.8% with two). This matches a 1.2k-example DeepScaleR subset. Similar gains on Qwen2.5-Math-7B, Llama3.2-3B-Instruct and DeepSeek-R1-Distill-Qwen-1.5B with GRPO or PPO. The policy-gradient loss drives the effect, and an entropy bonus aids exploration. — [NeurIPS 2025 proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/b1ea3f93167c55f3fc2999b68a170ff3-Abstract-Conference.html); [arXiv 2504.20571](https://arxiv.org/abs/2504.20571)

**Diffusion LMs for reasoning**
- *d1*, arXiv 2504.12216 (Apr 2025): SFT followed by **diffu-GRPO** on **LLaDA-8B-Instruct**. LLaDA's fixed generation length forces a predefined sequence length during RL training. — [arXiv 2504.12216](https://arxiv.org/pdf/2504.12216)
- *DCoLT: Diffusion Chain of Lateral Thought*, arXiv 2505.10446, PEER-REVIEWED (NeurIPS 2025): treats each reverse-diffusion step as a latent thinking action trained with outcome RL. On LLaDA: **+9.8% GSM8K, +5.7% MATH, +11.4% MBPP, +19.5% HumanEval**. — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/117288); [arXiv 2505.10446](https://arxiv.org/html/2505.10446v2)

### Inferences
- For a 100M-1B on-device model, looping gives more reasoning per stored byte. Flash memory and RAM limit parameters, while compute per token can be traded for latency, so a looped 300-500M model with adaptive exit is a defensible design point. The Ouro recipe used 7.7T tokens, far beyond a small budget, so the gain at about 10-50B tokens has not been shown.
- One-shot RLVR suggests that on-device RL from a handful of verified user tasks is plausible *if* the base already has latent ability. These results come from math-specialised Qwen bases, and later critiques of Qwen-specific RL effects (spurious rewards and the like) were not checked here.
- Diffusion-LM reasoning has no sub-1B evidence, so it is low priority for Aptivra.

### Gaps
- No verified 2026 sub-1B diffusion-LM reasoning paper was found.
- Ouro numbers on the smallest scales below 1B and the weight licence were not verified.
- No 2025-2026 Coconut follow-up at sub-1B scale was verified this session.

---

## Q4. Personal agents and memory (2025-2026): LongMemEval, Mem-alpha, MemoryOS, A-MEM follow-ups, agentic memory, personalisation benchmarks, on-device personalisation

### Takeaway
Two results matter most for Aptivra. **PersonaMem-v2** (Dec 2025) shows that a **4B model trained with reinforcement fine-tuning beats GPT-5 on implicit personalisation** (53%), and an agentic 2k-token memory reaches 55% with 16x fewer input tokens. **Mem-alpha** (Sep 2025) shows that *memory-management policy itself* can be learned with RL and generalises from 30k to over 400k tokens. Benchmarks have moved on to LongMemEval-V2 (2026) and implicit-preference tests, where frontier models still score only 37-48%.

### Cited Findings

**Mem-alpha: Learning Memory Construction via Reinforcement Learning, Yu Wang, Takanobu, et al. (UCSD, Stanford, Anuttacon), arXiv 2509.25911, 30 Sep 2025, PREPRINT [V-snip]**
- An RL-trained agent manages a memory with **core, episodic and semantic** components through memory-operation tools. The reward is downstream QA accuracy over the full interaction history. — [arXiv 2509.25911](https://arxiv.org/abs/2509.25911v1); [HF papers](https://huggingface.co/papers/2509.25911)
- Trained on instances of **up to 30k tokens, it generalises to over 400k tokens (more than 13x)**, with significant gains over memory-augmented baselines. Code availability is claimed by an aggregator, but the repo was not confirmed. — [arXiv 2509.25911](https://arxiv.org/html/2509.25911v1)

**PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory, arXiv 2512.06688, Dec 2025, PREPRINT [V-snip]**
- **1,000 simulated user-chatbot interactions, 300+ scenarios, 20,000+ user preferences and 128k-token contexts**. Most preferences are implicit. — [arXiv 2512.06688](https://arxiv.org/abs/2512.06688v1)
- Frontier LLMs score **37-48%** on implicit personalisation. **Qwen3-4B with reinforcement fine-tuning reaches 53%, beating GPT-5**. Agentic memory reaches **55% using a 2k-token memory, with 16x fewer input tokens**. — [arXiv 2512.06688](https://arxiv.org/html/2512.06688v1)
- The predecessor, *PersonaMem ("Know Me, Respond to Me")*, has 180+ simulated users and up to 60 sessions each across 15 tasks, and is PEER-REVIEWED (NeurIPS 2025). — [NeurIPS 2025](https://neurips.cc/virtual/2025/122431)

**MemoryOS: Memory OS of AI Agent, Kang, Ji, Zhao, Bai (BUPT, Tencent AI Lab), arXiv 2506.06326, May 2025, PEER-REVIEWED (EMNLP 2025) [V-snip]**
- Short-, mid- and long-term personal memory tiers, with storage, update, retrieval and generation modules. On LoCoMo with GPT-4o-mini: **+49.11% F1 and +46.18% BLEU-1** over baselines. The ACL Anthology abstract says 48.36% F1, so the two versions conflict. F1 is 36.23 with about 3,874 recalled tokens, against MemGPT at 29.13 with about 16,977. Code: github.com/BAI-LAB/MemoryOS. — [arXiv 2506.06326](https://arxiv.org/abs/2506.06326v1)
- A 2026 re-evaluation (arXiv 2602.14038, *Choosing How to Remember: Adaptive Memory Structures for LLM Agents*) ranks MemoryOS below several other systems on LoCoMo, so the original gains are not independently confirmed. — [arXiv 2602.14038](https://arxiv.org/pdf/2602.14038)

**Benchmarks and the state of the art**
- *LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues* (arXiv 2605.12493, 2026, PREPRINT). It reframes memory evaluation around agents becoming experienced operators of specialised environments, across five memory abilities. — [arXiv 2605.12493](https://arxiv.org/pdf/2605.12493)
- *Memora* (Microsoft Research, ICML 2026, PEER-REVIEWED per the blog). It claims the state of the art on LoCoMo and LongMemEval and beats Mem0, RAG and full-context inference with **up to 98% fewer context tokens**. It is a different system from MIRAS's "Memora" sequence model despite the name. — [Microsoft Research blog](https://microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity)
- Vendor claims (**not peer-reviewed; treat as marketing**): Mem0 cites 94.4 on LongMemEval at about 6,900 tokens per query, and Memanto claims 89.8 on LongMemEval and 87.1 on LoCoMo. — [Mem0 blog](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- MemoryAgentBench (arXiv 2507.05257) criticises LongMemEval's limited topical diversity. — [arXiv 2507.05257](https://arxiv.org/html/2507.05257v3)
- On-device: *Forget to Improve* (Q1) is the one verified 2026 paper on budget-limited, poisoning-resistant agent memory on edge hardware. — [arXiv 2606.25115](https://arxiv.org/abs/2606.25115)

### Inferences
- PersonaMem-v2 is the strongest evidence that **small models plus RL fine-tuning plus compact agentic memory** can beat frontier models on personalisation. Aptivra should adopt it, together with LongMemEval-V2, as its acceptance benchmarks.
- Mem-alpha's design (RL over memory-write tools with a QA reward) can be reproduced at small scale. The memory policy is a learnable skill, separate from knowledge.
- The 2k-token memory result fits a sub-1B context budget well.

### Gaps
- Model sizes for Mem-alpha's trained agent were not extracted (reportedly Qwen3-4B; unverified).
- A-MEM follow-ups, PrefEval and LaMP 2025-2026 updates were not covered in this session's searches. These are gaps, not negative findings.
- A "PersonaMem-v3" (Jul 2026) was mentioned by one aggregator but not confirmed.
- No LongMemEval or PersonaMem result was found for a model under 1B.

---

## Q5. Self-play and self-generated data: Absolute Zero, R-Zero, Self-Questioning LMs, EntiGraph

### Takeaway
Self-play with zero external data works at 3-8B scale, provided there is a verifier: a code executor (Absolute Zero), majority vote (R-Zero, SQLM) or unit tests. For learning a *personal corpus*, **EntiGraph** is the reference result. 1.3M source tokens were expanded to 455M synthetic tokens, raising closed-book QA from 39.49 to 56.22 on Llama 3 8B, which recovers 80% of the open-book gain.

### Cited Findings

**Absolute Zero (AZR), Zhao et al., arXiv 2505.03335, May 2025 (revised Oct 2025), PEER-REVIEWED (NeurIPS 2025) [V-snip]**
- One model proposes code and reasoning tasks that maximise its own learning progress and solves them. A **code executor** validates tasks and answers, providing a verifiable reward with no human data. — [arXiv 2505.03335](https://arxiv.org/abs/2505.03335); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/116121)
- Reports overall state-of-the-art coding and math results among zero-setting models, beating models trained on tens of thousands of curated examples, across model scales and classes. Specific benchmark numbers were not extracted this session. — [arXiv 2505.03335](https://arxiv.org/html/2505.03335v3)

**R-Zero: Self-Evolving Reasoning LLM from Zero Data, Huang et al., arXiv 2508.05004, 7 Aug 2025 (v4 Feb 2026), PEER-REVIEWED (ICLR 2026) [V-snip]**
- A Challenger and a Solver are initialised from one base model and co-evolve. The Challenger is rewarded for tasks at the edge of the Solver's ability. — [arXiv 2508.05004](https://arxiv.org/abs/2508.05004); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d49b9aacebda61051166335af6fd3061-Abstract-Conference.html)
- **Qwen3-4B-Base: +6.49 on math reasoning and +7.54 on general-domain reasoning**. Public GitHub repo. — [arXiv 2508.05004](https://arxiv.org/html/2508.05004v2)

**Self-Questioning Language Models (SQLM), Chen, Prabhudesai, Fragkiadaki, Liu, Pathak (CMU), arXiv 2508.03682, 5 Aug 2025, listed at ICLR 2026 [V-snip]**
- Asymmetric self-play: given only a *topic*, a proposer writes questions and a solver answers them. The proposer is rewarded for questions that are neither too easy nor too hard, and the solver by majority vote; for code, the proposer writes unit tests. Evaluated on 3-digit multiplication, OMEGA algebra and Codeforces. — [arXiv 2508.03682](https://arxiv.org/abs/2508.03682); [ICLR 2026](https://iclr.cc/virtual/2026/10018466)

**EntiGraph: Synthetic Continued Pretraining, Zitong Yang et al. (Stanford), arXiv 2409.07431, Sep 2024, PEER-REVIEWED (ICLR 2025) [V-snip]**
- Extracts entities from a small corpus and prompts an LLM to describe relations between entity pairs and triples. Applied to QuALITY (**265 books, 1.3M tokens**), it produced **455M synthetic tokens** using gpt-4-turbo, which were then used for continued pretraining of **Llama 3 8B**. — [arXiv 2409.07431](https://arxiv.org/pdf/2409.07431); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf)
- Closed-book accuracy goes from **39.49 to 56.22**, scaling log-linearly with synthetic tokens and recovering **about 80% of the open-book gain**. With retrieval, open-book accuracy goes from 60.35 to 62.60. — [ICLR 2025 slides](https://iclr.cc/media/iclr-2025/Slides/31270_nBGAjHj.pdf)
- Extended in the 2026 thesis *Continually self-improving AI* (Q1). — [arXiv 2603.18073](https://arxiv.org/abs/2603.18073)

### Inferences
- In a private setting, EntiGraph's generator (GPT-4) cannot be a cloud LLM without breaking privacy. Aptivra would need a local generator, possibly the model itself in the style of SEAL self-edits, or an explicitly consented offline pass. The quality cost of a small generator has not been measured in these papers.
- EntiGraph is complementary to SMF. EntiGraph decides *what* tokens to train on, and SMF decides *where* in the weights they go.
- R-Zero and SQLM are verified only at 3-8B. A sub-1B Challenger may be too weak to produce a useful curriculum. That is an open risk.

### Gaps
- Exact AZR benchmark numbers and model sizes were not extracted.
- No self-play paper at the 100M-1B scale was found and verified.
- No study of EntiGraph with a small (under 1B) local generator was found.

---

## Q6. Efficient small-model pretraining (2025-2026): data selection, curriculum, Muon, speedruns

### Takeaway
The 2026 small-model recipe is converging on Muon-family optimisers (NorMuon), muP, QK-norm, value residuals, gating, staged schedules and checkpoint EMA. IMU-1 (430M parameters, 72B tokens, open weights) claims to approach models trained on 56x more data. In NanoChat speedruns, data choice is as large a lever as the optimiser: switching to ClimbMix cut time-to-target by about 27%.

### Cited Findings

**IMU-1: Sample-Efficient Pre-training of Small Language Models, George Grigorev, arXiv 2602.02522, 25 Jan 2026, PREPRINT [V-abs]**
- **430M parameters trained on 72B tokens**. It "approaches the benchmark performance of models trained on 56x more data". The abstract lists no specific scores. — [arXiv 2602.02522](https://arxiv.org/abs/2602.02522)
- Recipe: **QK-norm attention, per-head gating, value residuals, LayerNorm scaling, NorMuon with cautious weight decay, muP, a three-stage schedule and post-hoc checkpoint EMA**, with ablations for each. Code, weights and data are released, with weights at huggingface.co/thepowerfuldeez/imu1_base. — [arXiv 2602.02522](https://arxiv.org/abs/2602.02522)

**Optimiser line (Muon successors), all PREPRINT [V-snip]**
- *Variance-Adaptive Muon* (arXiv 2601.14603, Jan 2026): time-to-target speedups of **1.24x and 1.15x over Muon** on GPT-2 Small and Medium. — [arXiv 2601.14603](https://arxiv.org/pdf/2601.14603)
- *The Newton-Muon Optimizer* (arXiv 2604.01472, Apr 2026): benchmarked against Modded-NanoGPT records, with a short track at validation loss 3.28 and a medium track at 2.92. — [arXiv 2604.01472](https://arxiv.org/pdf/2604.01472)
- *Spectral Scaling Laws of Muon* (arXiv 2606.04058): notes that the NanoGPT speedrun uses optimised 5-step Newton-Schulz polynomials, while DeepSeek-V4 uses 10 steps. — [arXiv 2606.04058](https://arxiv.org/html/2606.04058)

**Data selection, all PREPRINT [V-snip]**
- *AutoData: Agentic Search for Pre-training Data Selection* (arXiv 2609.19754, Sep 2026): in the NanoChat speedrun harness, switching from FineWeb-Edu to **ClimbMix cuts time to the CORE threshold by about 27%**. — [arXiv 2609.19754](https://arxiv.org/pdf/2609.19754)
- *OPUS: Efficient and Principled Data Selection in LLM Pre-training in Every Iteration* (arXiv 2602.05400, Feb 2026): per-iteration data selection. Numbers not extracted. — [arXiv 2602.05400](https://arxiv.org/pdf/2602.05400)
- *Automated LLM Speedrunning Benchmark* (arXiv 2506.22419, Jun 2025): reasoning models such as R1 and o3-mini, even with a state-of-the-art agent scaffold, struggle to reproduce known NanoGPT speedrun records. — [arXiv 2506.22419](https://arxiv.org/pdf/2506.22419)

### Inferences
- IMU-1 is the closest open, reproducible match to a 360M-500M Aptivra pretraining run: a similar parameter count, a stated token budget, open weights and per-component ablations. It is the obvious baseline to reproduce first.
- A 27% time saving from data choice alone is evidence that data-mix selection deserves as much measurement budget as optimiser tuning.
- The recipe (QK-norm, value residuals, gating) can be combined with looping (Ouro) and memory layers (SMF), but no paper tests that combination.

### Gaps
- IMU-1's benchmark table, licence and exact comparison models were not extracted. The "56x more data" claim needs checking against the full paper.
- The current Modded-NanoGPT and NanoChat record times as of Oct 2026 were not retrieved. Those live on GitHub, not in papers.
- Curriculum-learning papers for small LMs in 2025-2026 were not specifically surfaced.
- OPUS numbers were not extracted.

---

## Cross-cutting note for the report writer
- **Peer-reviewed (2025-2026) in this file:** SEAL (NeurIPS'25), Nested Learning (NeurIPS'25 per the Google blog), ATLAS (ICML'26), one-shot RLVR (NeurIPS'25), DCoLT (NeurIPS'25), Absolute Zero (NeurIPS'25), R-Zero (ICLR'26), SQLM (ICLR'26 listing), EntiGraph (ICLR'25), MemoryOS (EMNLP'25), PersonaMem v1 (NeurIPS'25), Memora (ICML'26 per the blog).
- **Preprints:** SMF, Improving SMF, TTT-E2E, MIRAS, HRM, TRM, Ouro, d1, Mem-alpha, PersonaMem-v2, LongMemEval-V2, IMU-1, Forget to Improve, the Muon variants, AutoData, OPUS and the 2026 continual-learning preprints.
- **Strongest sub-1B-relevant evidence:** Improving SMF (Qwen-2.5-0.5B, consumer hardware), IMU-1 (430M, open weights), TRM (7M, task-specific), and one-shot RLVR (1.5B).
