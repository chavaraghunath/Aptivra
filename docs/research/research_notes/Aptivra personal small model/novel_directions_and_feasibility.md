# Novel Directions and Hard Feasibility Limits for a Tiny, Private, Continually Learning Personal Model (about 100M-1B parameters)

Provenance key. **[F]** means the claim was fetched and read in this research session (2026-10-18). **[K]** means the claim comes from a well-known paper abstract cited from the researcher's prior knowledge and was not re-fetched this session; the report writer should treat [K] items as needing a spot check before publication. Speculation is labelled **SPECULATION**.

## Part A1: Modular and expandable architectures (growing experts, MoLoRA, BTM/BTX, merging and soups, expert libraries)

### Takeaway
Modular growth is the best-evidenced route for a small model to add capability over time without full retraining. BTX-style "branch, train separately, then merge into an MoE with a learned router" is a published recipe. Its evidence is at 7B scale with datacentre compute, and it has not been shown for on-device continual personal learning. LoRA-expert libraries are the realistic on-device analogue.

### Cited Findings
- BTX copies a seed model, trains domain experts in parallel "with high throughput and reduced communication cost", then places the experts' feedforward parameters into MoE layers, averages the remaining parameters, and finetunes a router. The paper presents BTX as generalising Branch-Train-Merge (no router) and sparse upcycling (no asynchronous expert training). It targets coding, maths and world knowledge and claims "the best accuracy-efficiency tradeoff" against the alternatives [F] — [BTX, arXiv 2403.07816](https://arxiv.org/abs/2403.07816)
- Branch-Train-Merge trains independent expert LMs on domains of data and ensembles or averages them, so new experts can be added without retraining the others [K] — [BTM, arXiv 2208.03306](https://arxiv.org/abs/2208.03306)
- Model soups: averaging the weights of several finetuned models from the same initialisation can improve accuracy at no extra inference cost [K] — [Model soups, arXiv 2203.05482](https://arxiv.org/abs/2203.05482)
- LoraHub composes several task LoRA modules with learned weights for few-shot adaptation to an unseen task, which amounts to a "library of LoRA experts" [K] — [LoraHub, arXiv 2307.13269](https://arxiv.org/abs/2307.13269)
- Aptivra's own history matters here. The repo's ADR-0006 v0 namespace already implemented growable T2 experts on NVMe with usage-heat pinning, and ADR-0011 retired it from the standard path (repo `CLAUDE.md`, Repository-Specific Notes). Any return to expert growth should explain why the v0 attempt was retired before re-proposing it.

### Inferences
- Fit for Aptivra: **medium-high** as LoRA or adapter experts on a frozen base. Per-domain adapters (finance, health, work, hobbies) are trained during "sleep" on laptop or phone and routed per query. **Low** as full BTX with FFN experts at 360M, because each expert copies the FFN weights, which multiplies storage and needs a router-finetune stage.
- The main risk is routing quality. A tiny router on a tiny model misroutes, and BTX mitigates this with a dedicated router-finetuning stage. That stage needs mixed replay data from every domain, which links directly to Part A2.
- Merging only works within a shared initialisation (a "basin"). Once the owner pretrains a new base, every adapter or expert must be retrained. Base-model versioning therefore becomes a product constraint.

### Gaps
- BTX's exact numbers (seed model, expert count, GSM8K/HumanEval deltas) were not on the abstract page, and the full PDF was not read this session.
- I found no published evidence of BTX or BTM applied below 1B parameters or under on-device compute.

## Part A2: Brain-inspired consolidation (sleep replay, complementary learning systems, generative replay)

### Takeaway
The CLS pattern (fast episodic store plus slow weight consolidation during offline "sleep") has moved from theory into LLM papers in 2024-2026, and early results are positive. The strongest evidence says the offline phase must *transform* memories (synthesise, recombine, self-generate training data), because verbatim replay is not enough. All results come from research-scale setups, so this is an emerging direction rather than a mature one.

### Cited Findings
- A Google Research / Cornell "Sleep" paradigm (arXiv 2606.03979) alternates active learning with offline consolidation. It has a "Knowledge Seeding" step that distils in-context knowledge into parameters, a "Dreaming" phase that uses RL-generated synthetic data, and a "Continuum Memory System" of MLP blocks with different update frequencies per layer to fight forgetting. It reports gains on continual-learning and knowledge-incorporation tasks. Retrieved only via secondary summaries (hyper.ai and Modelwire), so the primary paper was not read [F, secondary] — [hyper.ai summary of 2606.03979](https://hyper.ai/en/papers/2606.03979); [Modelwire](https://themodelwire.com/article/language-models-need-sleep-learning-to-self-modify-and-consolidate-memories-01KT5QWCZ4CDGHJK37E2EWXSD4)
- "Discovery by Dreaming" (arXiv 2607.16256) reports an ablation in which removing the dreaming phase drops SQuAD knowledge-incorporation accuracy from 46.2 to 36.2, which suggests the offline phase must transform memories rather than replay them [F, search snippet] — [arXiv 2607.16256](https://arxiv.org/pdf/2607.16256)
- MIRROR (ICLR 2026) pairs fast encoding of experience with slow "reconstructive consolidation" for persistent state in LLM agent systems. It operates at the system or state level, not as weight-level continual learning [F, listing only] — [ICLR 2026 MIRROR](https://www.iclr.cc/virtual/2026/10021270)
- Wake-Sleep Consolidated Learning (Kim et al., NeurIPS 2024) uses generative replay in alternating learning and consolidation phases. It is cited in 2607.16256 and was not read directly [F, secondary].
- Deep Generative Replay is the classic method: a generator produces pseudo-samples of past tasks that are interleaved with new data [K] — [Shin et al., arXiv 1705.08690](https://arxiv.org/abs/1705.08690)
- Hard counter-evidence on writing facts into weights: Gekhman et al. (EMNLP 2024) found that LLMs learn new facts through finetuning slowly, and as those examples are eventually learned, the tendency to hallucinate rises linearly. The authors conclude that models acquire knowledge mostly in pretraining and that finetuning mainly teaches them to use it [F] — [arXiv 2405.05904](https://arxiv.org/abs/2405.05904); [ACL Anthology](https://aclanthology.org/2024.emnlp-main.444)

### Inferences
- The Gekhman result is the single most important constraint for Aptivra. **Do not consolidate raw user facts into weights by SFT.** Facts (names, dates, preferences) belong in the fast store (an episodic log plus a knowledge graph). Weights should consolidate *skills, style, and recurring patterns* (how the user writes and what they usually mean). This is the CLS split applied literally.
- A defensible "sleep" job for Aptivra is to (1) cluster the day's episodes, (2) have the model (or a cloud teacher, with consent) generate transformed QA and skill exercises, (3) train a LoRA with replay from a frozen general-data buffer, and (4) gate promotion on a held-out regression suite plus a noise-floor-calibrated comparison. Per the owner's memory notes, equivalence gates need a measured noise floor.
- SPECULATION: "dreaming" by a 360M model will produce lower-quality synthetic data than the 2026 papers, which use larger models, and self-generated data from a weak model risks model-collapse-style drift. A cloud teacher used only during sleep, over redacted summaries, is likely to beat pure self-dreaming.

### Gaps
- No primary-source numbers for 2606.03979 (model size, compute) were read. Whether any of these consolidation papers worked below 1B parameters is unknown.

## Part A3: World models and predictive architectures (JEPA, V-JEPA, LeCun)

### Takeaway
JEPA is real and producing results in vision, video and robotics. It has no demonstrated payoff for a text-centric personal assistant today, so it is a long-horizon research bet rather than a product lever.

### Cited Findings
- V-JEPA 2 was pretrained action-free on more than 1 million hours of internet video. It reaches 77.3% top-1 on Something-Something v2 and 39.7 recall@5 on Epic-Kitchens-100 anticipation. After LLM alignment at 8B scale it scores 84.0 on PerceptionTest. V-JEPA 2-AC, post-trained on fewer than 62 hours of unlabelled DROID robot video, performs zero-shot pick-and-place on Franka arms in new labs [F] — [V-JEPA 2, arXiv 2506.09985](https://arxiv.org/abs/2506.09985)
- I-JEPA predicts representations of masked image regions in latent space rather than pixels, and LeCun's position paper proposes hierarchical JEPA world models plus a cost module for planning, as an alternative to autoregressive LLMs [K] — [I-JEPA, arXiv 2301.08243](https://arxiv.org/abs/2301.08243); [LeCun, "A Path Towards Autonomous Machine Intelligence", OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf)

### Inferences
- Fit: **low for v1**. The nearest useful application is a *latent predictive model of the user's routine* (calendar, location, app usage) that anticipates needs, such as "you usually leave at 8:10; traffic is bad." That is a small time-series or embedding predictor, not a JEPA-scale model, and it could be built with a few-million-parameter model beside the LM. SPECULATION.
- Claims that JEPA will make a 360M model "understand the world" have no evidence in a language-assistant setting.

### Gaps
- I found no published JEPA-for-text or JEPA-for-personal-assistant result with benchmark numbers.

## Part A4: Neuro-symbolic and structured memory (knowledge graph plus LLM, program synthesis, symbolic planner)

### Takeaway
External structured memory is the most evidenced way for a small model to "know" far more than its weights. Even frontier assistants lose about 30% accuracy on long-term memory, so memory architecture (indexing, retrieval, time-awareness) matters more than model size for a personal assistant.

### Cited Findings
- LongMemEval (ICLR 2025) uses 500 questions embedded in scalable chat histories and tests information extraction, multi-session reasoning, temporal reasoning, knowledge updates and abstention. Commercial assistants and long-context LLMs show about a **30% accuracy drop** on sustained-interaction memory. The proposed fixes are session decomposition, fact-augmented key expansion and time-aware query expansion, which substantially improve recall and QA [F] — [LongMemEval, arXiv 2410.10813](https://arxiv.org/html/2410.10813v2)
- MemGPT treats the LLM context window as virtual memory with OS-like paging between the context and external storage [K] — [MemGPT, arXiv 2310.08560](https://arxiv.org/abs/2310.08560)
- Voyager stores LLM-written programs in an ever-growing skill library that is retrieved and composed for new tasks, an early "program synthesis as memory" result [K] — [Voyager, arXiv 2305.16291](https://arxiv.org/abs/2305.16291)
- Self-RAG trains a model to decide when to retrieve and to critique its own generations using reflection tokens [K] — [Self-RAG, arXiv 2310.11511](https://arxiv.org/abs/2310.11511)

### Inferences
- Fit: **very high**. A typed personal knowledge graph (entities, relations, timestamps, provenance, confidence), with a "knowledge update" supersedes-edge and an explicit abstention path, directly attacks the LongMemEval failure modes. This is where Aptivra can plausibly be *best-in-class*, because frontier assistants do not do this well either.
- A skill library of verified small programs (calendar arithmetic, budget roll-ups, unit conversions) offloads exactly the tasks small models fail (GSM8K-style arithmetic; see Part B). Program-aided execution replaces neural arithmetic.

### Gaps
- No sub-1B-specific results on LongMemEval were found. The per-model breakdown behind the 30% figure was not checked.

## Part A5: Hybrid small-large (cascades, routing, privacy-preserving cloud escalation)

### Takeaway
Cascades and routers deliver large, measured cost savings (up to 98% in FrugalGPT and more than 2x in RouteLLM) while keeping near-frontier quality. Apple PCC is the strongest published template for private cloud escalation. The hybrid design is the only evidenced path to "smartest advisor"-grade answers from a sub-1B product.

### Cited Findings
- FrugalGPT notes that LLM API prices differ by two orders of magnitude. Its cascade "can match the performance of the best individual LLM (e.g. GPT-4) with up to 98% cost reduction or improve the accuracy over GPT-4 by 4% with the same cost" [F] — [FrugalGPT, arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- RouteLLM learns routers from preference data between a strong and a weak model and "significantly reduces costs, by over 2 times in certain cases, without compromising the quality of responses." Per-benchmark numbers were not on the abstract page [F] — [RouteLLM, arXiv 2406.18665](https://arxiv.org/abs/2406.18665)
- Apple Private Cloud Compute rests on five requirements, each enforced technically [F] — [Apple Security Research, PCC](https://security.apple.com/blog/private-cloud-compute/):
  - (1) Stateless computation: data volume keys are randomised per reboot and never persisted, and request data is deleted after inference.
  - (2) Enforceable guarantees: requests are encrypted only to the public keys of validated nodes, load balancers cannot decrypt, and Secure Boot and code signing apply with no JIT.
  - (3) No privileged runtime access: no remote shell, debugger or general logging, only pre-specified, audited logs.
  - (4) Non-targetability: single-use RSA blind-signature credentials, a third-party OHTTP relay hiding the IP address, and statistically auditable node selection.
  - (5) Verifiable transparency: production images are published, measurements sit in an append-only transparency log, and devices send data only to nodes whose attested measurements match the log. A research VM and bounty exist.

### Inferences
- An Aptivra routing policy should combine (a) a calibrated confidence or uncertainty estimate from the small model, (b) a task-type classifier (arithmetic, multi-step reasoning, broad world knowledge, so escalate; personal recall, style, short rewrite, so stay local), and (c) a privacy classifier that redacts or pseudonymises entities from the local KG before any outbound call. The local KG is what makes redaction good: it knows which tokens are personal.
- An individual developer cannot replicate PCC's hardware attestation. Realistic tiers are (i) a commercial API with zero-retention terms plus local redaction, (ii) confidential-computing VMs (cloud TEEs with attestation) for a self-hosted teacher, and (iii) local differential privacy only for *aggregate* signals, since LDP noise destroys the utility of a single user's query. SPECULATION on the exact trade-offs, because no cost measurement was made here.
- Routing thresholds, redaction rules and escalation budgets are configuration under the repo's Configurability Mandate and must be registered.

### Gaps
- RouteLLM's exact "% of strong-model calls at 95% quality" numbers were not retrieved. Split-inference privacy leakage (activation inversion) was not researched this session.

## Part A6: Learning from the internet and books safely (retrieval-augmented continual learning, curricula, active learning, curiosity)

### Takeaway
Evidence favours *retrieve at query time, consolidate selectively*. Pushing web or book facts into a tiny model's weights collides with the Gekhman hallucination finding and with the limited knowledge capacity of small models (A7, B1).

### Cited Findings
- Finetuning on new facts increases hallucination roughly linearly as those facts are learned [F] — [Gekhman et al., arXiv 2405.05904](https://arxiv.org/abs/2405.05904)
- Self-RAG-style learned retrieval decisions and self-critique improve factuality over always-retrieve or never-retrieve baselines [K] — [arXiv 2310.11511](https://arxiv.org/abs/2310.11511)
- Voyager's automatic curriculum (the LLM proposes the next task based on its current skill state) plus a verified skill library is the reference design for curiosity-driven self-curricula [K] — [arXiv 2305.16291](https://arxiv.org/abs/2305.16291)

### Inferences
- Safe pattern: reading the web or a book produces (1) an indexed corpus in local retrieval, (2) extracted claims with source and confidence in the KG, and (3) only *procedural* skills (for example, how to read a balance sheet) are distilled into adapters during sleep, gated by evaluation. A curiosity signal can be framed as "questions the user asked that the system abstained on or escalated." Those are the active-learning targets. SPECULATION, as no paper tests this exact loop on device.
- Prompt-injection and poisoning risk rises when the assistant ingests the open web autonomously. Ingested text must be treated as data, never as instructions, which is the same rule this research session follows.

### Gaps
- No quantitative study was found of retrieval-augmented *continual* learning for sub-1B models.

## Part A7: Alternative small-model architectures (SSM and hybrids, diffusion LMs, memory layers, TTT and Titans)

### Takeaway
Hybrid conv- or SSM-plus-attention designs (Hymba, LFM2) are the best-evidenced efficiency win at this scale: smaller KV caches, higher throughput and better accuracy per parameter. Memory layers are the most interesting "cheap knowledge capacity" idea, but the evidence is at 1T-token, billion-parameter scale. Titans and TTT are the closest published match to "learns continuously at inference time" and remain research-grade.

### Cited Findings
- Hymba-1.5B-Base (hybrid heads with attention and SSM in parallel) "outperforms Llama-3.2-3B with 1.32% higher average accuracy, an 11.67x cache size reduction, and 3.49x throughput," and beats all public sub-2B models per the abstract [F] — [Hymba, arXiv 2411.13676](https://arxiv.org/abs/2411.13676)
- LFM2 (Liquid AI) is a family of 350M-8.3B models (dense 350M, 700M, 1.2B and 2.6B, plus an 8.3B MoE with 1.5B active), all with 32K context, built from gated short convolutions plus a few GQA blocks. LFM2-2.6B reports 79.56% IFEval and 82.41% GSM8K [F, search summary of the report] — [LFM2 Technical Report, arXiv 2511.23404](https://arxiv.org/abs/2511.23404v1)
- Memory Layers at Scale: trainable key-value lookup layers scaled to 128B memory parameters, pretrained on 1T tokens and compared against base models of up to 8B. Memory-augmented models outperform dense models with "more than twice the computation budget" and beat MoE when matched for compute and parameters, with gains "especially pronounced" on factual tasks [F] — [arXiv 2412.09764](https://arxiv.org/abs/2412.09764)
- Product-key memory provides sparse lookup over a very large key set at small compute cost and is the precursor of memory layers [K] — [Lample et al., arXiv 1907.05242](https://arxiv.org/abs/1907.05242)
- Titans adds a neural long-term memory module that learns to memorise at test time. It reports beating Transformers and modern linear recurrent models and scaling to context windows larger than 2M tokens. Published at NeurIPS 2025 [F] — [Titans, arXiv 2501.00663](https://arxiv.org/abs/2501.00663); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/119639)
- TTT layers make the hidden state a small model updated by self-supervised gradient steps during inference [K] — [arXiv 2407.04620](https://arxiv.org/abs/2407.04620)
- Background architectures [K]:
  - Mamba (selective SSM, linear-time) — [arXiv 2312.00752](https://arxiv.org/abs/2312.00752)
  - Mamba-2 (state-space duality) — [arXiv 2405.21060](https://arxiv.org/abs/2405.21060)
  - xLSTM — [arXiv 2405.04517](https://arxiv.org/abs/2405.04517)
  - RWKV-7 "Goose" — [arXiv 2503.14456](https://arxiv.org/abs/2503.14456)
  - LLaDA (masked diffusion LM competitive with autoregressive models at 8B) — [arXiv 2502.09992](https://arxiv.org/abs/2502.09992)

### Inferences
- **Most promising combination for Aptivra (SPECULATION, untested):**
  - A hybrid conv- or SSM-plus-attention backbone in the LFM2 or Hymba style, for long personal context at a fixed small state.
  - One or two memory layers, whose large sparse tables can live in RAM or flash with only a few rows touched per token. This "adds knowledge capacity cheaply" and echoes the retired v0 NVMe-expert idea, with stronger published support.
  - LoRA experts for skills.

  A memory layer also offers a *localised* write target: updating only selected memory rows, rather than dense weights, may reduce interference. That is a hypothesis to test, not an established result.
- Diffusion LMs: fit is **low** for v1. Evidence sits at 8B scale, and on-device toolchains (llama.cpp, Core ML, ExecuTorch) are built around autoregressive decoding.
- Titans and TTT: fit is **research-track**. They are the only published architectures whose weights genuinely change during use. That matches "continual learning" literally, but it brings safety and rollback problems: test-time-updated state must be versioned and revertible under the repo's Stabilität rules.

### Gaps
- No memory-layer result below about 1B base parameters was confirmed. LFM2-350M's exact on-phone latency and memory were not retrieved. RWKV-7, xLSTM and Mamba sub-1B benchmark numbers were not fetched.

## Part B1: Measured capability ceilings by size (about 135M, 360M and 1B against larger models)

### Takeaway
At 360M, current best-in-class models reach roughly MMLU 33-45 (format-dependent; random is 25) and GSM8K 3-30 (highly data-dependent). At 1B, they reach MMLU about 49 and GSM8K about 44. Function calling at 1B is weak (BFCL v2 25.7 against 67 at 3B). Sub-1B models reliably fail multi-step maths, tool calling, broad factual recall and long-context multi-hop reasoning.

### Cited Findings
- **SmolLM2-360M base** [F] — [SmolLM2-360M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct):
  - HellaSwag 54.5, ARC 53.0, PIQA 71.7, MMLU (cloze) 35.8, TriviaQA 16.9, GSM8K (5-shot) **3.2**.
  - Qwen2.5-0.5B: MMLU (cloze) 33.7, TriviaQA 4.3, GSM8K **33.4**.
- **SmolLM2-360M-Instruct**: IFEval 41.0, MT-Bench 3.66, MMLU (cloze) 32.8, BBH 27.3, GSM8K 7.43. Qwen2.5-0.5B-Instruct scores GSM8K 26.8 and MT-Bench 4.16 [F] — same source.
- **LFM2-350M**: MMLU 43.43, GPQA 27.46, IFEval 65.12, IFBench 16.41, GSM8K 30.1, MMMLU 37.99. Qwen3-0.6B: MMLU 44.93, IFEval 64.24 [F, search summary of a README revision; verify against the current card] — [LFM2-1.2B README](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/26bb21ffbfe2d9527e007758ddddb3700955d2af/README.md)
- **Llama 3.2 1B, 3B and Llama 3.1 8B (instruct)** [F] — [Llama-3.2-1B-Instruct model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct):

  | Benchmark | 1B | 3B | 8B |
  |---|---|---|---|
  | MMLU | 49.3 | 63.4 | 69.4 |
  | GSM8K (CoT) | 44.4 | 77.7 | 84.5 |
  | MATH | 30.6 | 48.0 | 51.9 |
  | GPQA | 27.2 | 32.8 | 32.8 |
  | IFEval | 59.5 | 77.4 | 80.4 |
  | BFCL v2 (tool calling) | **25.7** | 67.0 | 67.1 |
  | Nexus | 13.5 | 34.3 | 38.5 |
  | InfiniteBench En.MC | 38.0 | 63.3 | 72.2 |
  | NIH multi-needle | 75.0 | 84.7 | 98.8 |

- Qwen3-0.6B and 1.7B: a community table reports MMLU (0-shot CoT) of about 43.3 and 64.2. An aggregator lists conflicting figures for 1.7B (MMLU 50.99), so treat these as unverified [F, low quality] — [HF community README](https://huggingface.co/Tiiny/SmallThinker-4BA0.6B-Instruct/commit/1671d12c9079a000ec363db93be96e8f9d13cbec); [lmmarketcap](https://lmmarketcap.com/zh/upcoming/qwen3-1-7b)
- Frontier reference: GPT-4 reported 86.4% on MMLU (5-shot) in 2023 [K] — [GPT-4 Technical Report, arXiv 2303.08774](https://arxiv.org/abs/2303.08774)

### Inferences
- **The size gap is large and quantified.** Moving from 1B to 3B nearly doubles GSM8K (44 to 78) and nearly triples tool-calling (26 to 67). Against GPT-4-class MMLU (about 86), a 360M model at 33-45 sits much closer to chance (25) than to the frontier.
- **Data matters as much as size at 360M.** SmolLM2-360M (3.2) and LFM2-350M (30.1) differ about tenfold on GSM8K at near-identical size, as do SmolLM2-360M and Qwen2.5-0.5B (33.4). For the owner's own 360M pretraining run, the maths and reasoning data mix will matter more than the architecture choice.
- **Format caveat:** SmolLM2 reports MMLU in "cloze" form, while LFM2 and Llama use multiple-choice form. These numbers are not directly comparable, so the report should not rank across them.
- **Reliable sub-1B failure modes (evidenced):** multi-step arithmetic (GSM8K below 45 even at 1B), function or tool calling (BFCL 25.7 at 1B), long-context multi-needle and QA (InfiniteBench En.QA 20.3), broad factual recall (TriviaQA 16.9 at 360M), and graduate-level reasoning (GPQA about 27, near the 25% chance level).

### Gaps
- SmolLM2-135M and 1.7B tables were not on the fetched page (SmolLM2 paper [K]: [arXiv 2502.02737](https://arxiv.org/abs/2502.02737)). No agentic-benchmark (SWE-bench, tau-bench, GAIA) results for sub-1B models were found; they are likely near zero but unmeasured here. Current frontier (2026) MMLU-Pro and GPQA numbers were not fetched.

## Part B2: Known failure modes for small continually learning assistants

### Takeaway
The evidenced failure modes are: hallucination that rises when facts are trained into weights; long-term memory failure (about 30% accuracy drop even for frontier assistants); and forgetting and drift under sequential updates. Sycophancy and wrong-fact entrenchment are well documented in general LLMs. Sub-1B-specific measurements are scarce.

### Cited Findings
- Hallucination rises linearly as finetuned new-fact examples are learned [F] — [Gekhman et al.](https://arxiv.org/abs/2405.05904)
- Memory failures appear in knowledge updates and temporal reasoning, which are LongMemEval categories. There is a roughly 30% accuracy drop for commercial assistants [F] — [LongMemEval](https://arxiv.org/html/2410.10813v2)
- Sycophancy increases with model scale and instruction tuning, and simple synthetic-data intervention reduces it [K] — [Wei et al., arXiv 2308.03958](https://arxiv.org/abs/2308.03958)
- Forgetting needs active mitigation: consolidation papers add multi-timescale memories or dreaming specifically because naive updates forget [F, secondary] — [2606.03979 summary](https://hyper.ai/en/papers/2606.03979)

### Inferences
- **Memorising wrong user facts** is the dominant product risk. A misheard name SFT'd into weights cannot be cleanly deleted. In a KG it is a single row with provenance that can be edited, superseded or deleted, which also covers GDPR-style right-to-erasure. This is a strong argument for facts-in-KG and skills-in-weights.
- **Drift:** each sleep cycle must be gated on a frozen regression suite with a measured noise floor, and must be rollback-able. This is consistent with the owner's existing H3 noise-floor and EWC work (recent commit a14492a).
- **Safety:** a small model's refusal and safety behaviour can erode during personal finetuning. Re-running a safety eval set at each promotion gate is the minimal control. SPECULATION about magnitude at 360M, which was not measured here.

### Gaps
- No measured hallucination or sycophancy rates for 135M-1B models were found this session. No study of on-device continual LoRA drift over months of real usage was found.

## Part B3: Feasibility of "super intelligent", "smartest advisor" and "learns the entire user world", and what makes it best-in-class

### Takeaway
**"Super intelligent" at 100M-1B is not attainable.** Every measured reasoning benchmark places sub-1B models near chance on hard tasks and far below 3B-8B models, let alone frontier models. **"Smartest advisor" is attainable only as a hybrid system** with cloud escalation. **"Learns the entire user world"** is partly attainable, and in a narrow sense possibly better than frontier assistants, if "learning" means a structured, time-aware, editable personal memory plus retrieval rather than weights.

### Cited Findings
- 1B instruct: MMLU 49.3, GSM8K 44.4, GPQA 27.2, BFCL 25.7, against 8B at 69.4, 84.5, 32.8 and 67.1 [F] — [Llama 3.2 card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- Cascades match GPT-4 quality at up to 98% lower cost [F] — [FrugalGPT](https://arxiv.org/abs/2305.05176)
- Frontier assistants themselves lose about 30% on long-term interactive memory [F] — [LongMemEval](https://arxiv.org/html/2410.10813v2)
- Memory layers add factual capacity at small compute cost (evidenced at large scale) [F] — [arXiv 2412.09764](https://arxiv.org/abs/2412.09764)
- Hybrid architectures cut KV cache by 11.67x at higher accuracy [F] — [Hymba](https://arxiv.org/abs/2411.13676)

### Inferences
- **Claim verdicts**

  | Claim | Verdict | Evidence-based reason |
  |---|---|---|
  | "Super intelligent" | **Unattainable** at this size; hype | 1B GPQA ~27 (chance 25); GSM8K 44 at 1B, 3-30 at 360M |
  | "Smartest advisor" (standalone) | **Unattainable** | Tool-calling 25.7 at 1B; broad knowledge (TriviaQA 16.9) |
  | "Smartest advisor" (hybrid with private cloud escalation) | **Plausible** | Cascades retain frontier quality; local model handles personal context and redaction |
  | "Learns the entire user world" (in weights) | **Unattainable and unsafe** | Gekhman: new facts in weights raise hallucination; no deletion |
  | "Learns the entire user world" (KG plus retrieval plus skill adapters) | **Attainable for personal facts and habits**; possible edge over frontier | Frontier assistants drop ~30% on LongMemEval; a structured memory targets exactly that |
  | "Learns continuously" | **Attainable for style and skills via gated sleep-time LoRA**; research-grade for true test-time learning (Titans/TTT) | 2025-2026 consolidation papers; not shown at sub-1B |

- **What could make Aptivra genuinely best-in-class at its size** (each item is a design hypothesis to be measured):
  1. Best-in-class *personal memory*. Target LongMemEval-style metrics on the user's own data: knowledge-update correctness, temporal reasoning and abstention. This is the one axis where a small system can beat a frontier chat assistant.
  2. *Calibrated abstention and escalation.* Knowing when it does not know matters more at 360M than raw accuracy. Measure the calibration (ECE) of the escalation decision.
  3. *Program-aided arithmetic and planning* (a skill library and a symbolic planner) to bypass the GSM8K and tool-calling ceiling.
  4. *Hybrid long-context backbone* (LFM2- or Hymba-style) so the whole recent personal context fits on a phone.
  5. *Gated, rollback-able sleep consolidation* of style and skills, never raw facts, promoted only past a noise-floor-calibrated regression and safety gate.
  6. *Privacy-first escalation* with KG-informed redaction and a PCC-inspired provider choice (zero retention, attestation where available).
- **Unattainable without a cloud model at this scale:** frontier-level reasoning (GPQA or MATH class), reliable multi-step agentic tool use, broad up-to-date world knowledge without retrieval, and high-quality synthetic-data "dreaming."
- Given the owner's GCP budget constraint (₹1000 total, from memory notes), any cloud-teacher or cloud-escalation design needs a per-query cost model and a hard budget cap before adoption.

### Gaps
- No direct measurement exists of how a 360M model plus KG plus retrieval system scores on LongMemEval. That is the single most decision-relevant experiment, and it can run locally.
- No current (2026) frontier benchmark table was fetched for an up-to-date gap figure. GPT-4's 86.4 MMLU [K] understates today's frontier.
