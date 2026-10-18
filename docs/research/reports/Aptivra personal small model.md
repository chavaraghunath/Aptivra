# Shrink the Model, Grow the Memory

The best-in-class smallest personal model that Aptivra can build with ₹28,796.63 of Google Cloud credit and an M4 Max is not a tiny genius. It is a small, owned language engine of about 360M parameters wrapped in a large, user-owned memory. Research on knowledge injection is consistent on one point: facts about the user's world belong in a structured, deletable external memory. Skills, style and habits belong in small, gated, rollback-able adapters. Retrieval beats fine-tuning for new facts, and training new facts into weights raises hallucination ([Ovadia et al.](https://arxiv.org/abs/2312.05934); [Gekhman et al.](https://arxiv.org/abs/2405.05904)). One pretraining run fits the credit, and the chip decides how good it is. A single TPU v6e chip on spot is roughly 6–11× cheaper per training token than the NVIDIA options on GCP. That buys about 150B tokens for a 360M model at roughly ₹16,400–20,500, against only about 16–28B tokens on A100 or L4 spot for the same money ([GCP Billing Catalog](https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list)). The v6e advantage is the least-verified number in this plan, so a one-hour pilot must measure it before the main run. Even a perfect run lands below SmolLM2-360M (HellaSwag 54.5, GSM8K 3.2) and far below Qwen3-0.6B (MMLU 52.8, GSM8K 59.6), which trained on 4T and 36T tokens ([SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M); [Qwen3 report](https://arxiv.org/html/2505.09388)). "Super intelligent" is therefore not achievable at this size. One axis is achievable: memory of the user's own life. Frontier assistants lose about 30% accuracy on long-term interactive memory ([LongMemEval](https://arxiv.org/html/2410.10813v2)). A small model with a well-built, time-aware, user-owned memory can plausibly beat them there, and that is where Aptivra should aim to be best-in-class.

## A 360M model sits nearer chance than the frontier on hard tasks

The size ladder is steep and well measured. Llama-3.2-1B-Instruct scores 49.3 on MMLU, 44.4 on GSM8K, 27.2 on GPQA (chance is 25) and 25.7 on BFCL v2 tool calling. The same family at 8B scores 69.4, 84.5, 32.8 and 67.1 ([Llama-3.2-1B-Instruct card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)). Going from 1B to 3B alone nearly doubles GSM8K (44 → 78) and nearly triples tool calling (26 → 67). At 360M, the best fully open base model, SmolLM2-360M, reaches HellaSwag 54.5, MMLU-cloze 35.8, TriviaQA 16.9 and GSM8K 3.2 after 4T tokens on 128 H100s ([SmolLM2-360M card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)). Its instruct version scores **IFEval 41.0 and GSM8K 7.4** ([SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)). The 2025–2026 frontier at this size moved through distillation, huge token budgets and hybrid architectures. LFM2.5-230M reports **IFEval 71.7 and BFCLv3 43.3** after 19T tokens, distillation and multi-stage RL ([LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)). MobileLLM-R1-360M reaches **MATH500 28.4 and GSM8K 24.5** after logit distillation from Llama-3.1-8B and 6.2M reasoning SFT samples, and it ships under a non-commercial licence ([MobileLLM-R1-950M card](https://huggingface.co/facebook/MobileLLM-R1-950M)).

The table places a realistic owned Aptivra-360M on that ladder. Its row is an extrapolation from these anchors, not a measurement.

| Model | Training | Knowledge / commonsense | Maths | Instruction / tools | Licence |
|---|---|---|---|---|---|
| **Aptivra-360M, owned (projected)** | ~150B tokens, from scratch | HellaSwag ~42–48, MMLU-cloze ~31–34 | GSM8K base low single digits; ~10–25 after reasoning SFT | IFEval ≥41 target after SFT | Apache-2.0 (own) |
| SmolLM2-360M | 4T tokens | HellaSwag 54.5, MMLU-cloze 35.8 | GSM8K 3.2 (base), 7.4 (instruct) | IFEval 41.0 | Apache-2.0 |
| LFM2.5-230M | 19T tokens + distillation + RL | MMLU-Pro 20.3 | not recommended for maths | IFEval 71.7, BFCLv3 43.3 | LFM Open ($10M revenue cap) |
| MobileLLM-R1-360M | ~4.2T + KD + 6.2M SFT | — | MATH500 28.4, GSM8K 24.5 | — | FAIR non-commercial |
| Qwen3-0.6B base | 36T tokens | MMLU 52.8 | GSM8K 59.6 | multi-turn BFCL 3.6% | Apache-2.0 |
| Llama-3.2-1B-Instruct | not verified | MMLU 49.3 | GSM8K 44.4 | BFCL v2 25.7 | Llama community |

Sources: [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), [LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M), [MobileLLM-R1](https://huggingface.co/facebook/MobileLLM-R1-950M), [Qwen3 report](https://arxiv.org/html/2505.09388), [benchmarklist Qwen3-0.6B](https://benchmarklist.com/models/qwen3-0.6b-6/), [Llama-3.2-1B card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

That table settles the hype question. **"Super intelligent" is unattainable.** Even at 1B, graduate-level reasoning sits within 2 points of chance. **A standalone "smartest advisor" is unattainable.** Broad recall (TriviaQA 16.9 at 360M) and tool calling are too weak. Multi-turn agentic reliability collapses below about 1.5B: Qwen3-0.6B scores 3.6% on BFCL multi-turn ([benchmarklist](https://benchmarklist.com/models/qwen3-0.6b-6/)). Three things are attainable at 360M:

- **A fluent instruction follower** that writes, rewrites, summarises, classifies, extracts and routes.
- **Single, schema-valid tool calls** over a closed set of personal tools. Constrained decoding guarantees syntax, and training on function-calling data drives the selection.
- **Grounded answers about the user's own life**, with citations to memory.

The benchmark gap also shows where the smallest model can still win. Frontier systems drop about 30% on LongMemEval's tests of multi-session reasoning, temporal reasoning, knowledge updates and abstention ([LongMemEval](https://arxiv.org/html/2410.10813v2)). PersonaMem-v2 shows a 4B model with reinforcement fine-tuning beating GPT-5 on implicit personalisation (53% vs frontier 37–48%), and an agentic memory reaching 55% with a 2k-token memory and 16× fewer input tokens ([PersonaMem-v2](https://arxiv.org/html/2512.06688v1)). Personalisation is an architecture problem more than a parameter problem.

Data matters as much as size at 360M. SmolLM2-360M and LFM2-350M are nearly the same size yet differ about tenfold on GSM8K (3.2 vs 30.1), because maths data and post-training dominate ([SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct); [LFM2 README](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/26bb21ffbfe2d9527e007758ddddb3700955d2af/README.md)). Benchmark formats also differ: SmolLM2 reports cloze MMLU, while Llama and LFM report multiple choice. Aptivra's own evals must fix one harness and compare only within it.

## ₹28,797 buys one serious pretraining run, if it runs on the right chip

### The credit is about US$340 and expires on 2027-01-13

At an assumed ₹84.8/$, the credit is about US$340. A Free-Trial billing account cannot attach GPUs or request quota increases. Upgrading to Paid keeps the remaining credit, but it must still be consumed within the original 90-day window, and anything beyond the credit bills the owner's card ([GCP Free Trial docs](https://docs.cloud.google.com/free/docs/free-cloud-features)). GPU quota on project `aptivra-training` is currently 0 globally. That leaves about 87 days from today, and quota approval, pipeline debugging and the run itself all have to fit inside that window.

The GCP billing catalog, queried 2026-10-18, sets the economics ([GCP Billing Catalog API](https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list)). The token figures below assume 40% model-FLOPs utilisation, the 6N rule of thumb, and a further 35% discount for attention overhead at 360M. That MFU level is calibrated within about 10% of llm.c's measured H100 throughput ([llm.c #481](https://github.com/karpathy/llm.c/discussions/481)).

| Option (us-central1, spot, all-in) | ₹/hour | 360M tokens per ₹20,000 | 135M tokens per ₹20,000 | Verdict |
|---|---|---|---|---|
| **TPU v6e, 1 chip** ($0.584/h) | ~₹49.5 | **~180B** | ~490B | ~6× cheaper per token than L4, ~11× than A100; MFU unverified |
| L4, g2-standard-8 ($0.51/h) | ~₹43 | ~28B | ~74B | Fallback; NVIDIA/PyTorch stack |
| A100-40GB, a2-highgpu-1g ($2.20/h) | ~₹187 | ~16B | ~44B | Poor value |
| H100, a3-highgpu-1g ($6.56/h) | ~₹556 | ~18B | ~47B | Poor value on GCP |
| M4 Max (local, measured) | ₹0 | ~6.4B/month at seq 167 | ~15B/month (unmeasured) | Ablations, proxies, post-training |

Mumbai is not cheaper. It has no A100 type in any zone, and its v6e spot price is 2.3× the US price. The M4 Max measured **2,452 tok/s** for full-parameter AdamW on a 362M model, about 5.3 effective TFLOP/s. That is useful for proxies and post-training, but the Mac is not a pretraining engine (repository evidence `docs/adr/0012-evidence/pilot-evidence.json`).

### Two corrections to the earlier plan

First, the earlier estimate of "~7–20B tokens on ~35–100 A100-hours" aimed at the wrong regime. Chinchilla-optimal (~20 tokens per parameter) is the wrong target for an inference-heavy on-device model. Sardana et al. trained models up to 10,000 tokens per parameter and quality kept improving, with diminishing returns ([Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)). SmolLM2-360M sits near 11,000 tokens per parameter. Hugging Face's own nanowhale-100m, trained on 2.6B tokens on one H100, produces "often incoherent" output and is "not suitable for any production use" ([nanowhale-100m](https://huggingface.co/HuggingFaceTB/nanowhale-100m)). Aptivra's first base failed at 12.5 bits per byte for exactly this reason: too little pretraining.

Second, on A100 or H100 spot the whole credit buys only about 23–33B tokens at 360M. That projects to roughly Pythia-410M-class quality, a weak return. **The chip choice moves quality more than any architecture choice can.**

### Recommendation: 360M on v6e spot, about 150B tokens, ₹20,000 hard cap

On a single v6e chip, a 360M model should process about 126k tokens/s after the attention discount. **150B tokens is about 331 chip-hours, or ~₹16,400 at perfect utilisation and ~₹19,700–20,500 with 20–25% overhead** for preemption, checkpointing and evaluation. That is about 415 tokens per parameter: 20× Chinchilla, though roughly 25× short of SmolLM2. The projected HellaSwag of about 42–48 is better than GPT-2-124M's 29.4 by a wide margin and sits inside the SmolLM v1 range of 41.2 (135M) to 51.8 (360M) ([llm.c #481](https://github.com/karpathy/llm.c/discussions/481); [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)).

The 135M alternative gets about 400B tokens for the same money but starts from a lower ceiling (SmolLM2-135M HellaSwag 42.1, GSM8K 1.4) ([SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)). It should be the fallback only if the v6e pilot fails.

The serious caveat is that **no published MFU exists for single-chip v6e training of 100M–1B models.** Small models on TPU can run far below 40% unless the batch is large. The v6e path also needs a JAX/MaxText or PyTorch/XLA stack, which cannot be debugged on the Mac in identical form. That is real engineering risk, and it needs an owner decision and a measured pilot, not faith.

### "Own" has to be defined before money is spent

Three routes trade quality against provenance:

- **Pruning plus distillation from an open parent.** This is 30–40× more token-efficient: Minitron used "up to 40× fewer training tokens", and Sheared-LLaMA used 3% of from-scratch compute ([Minitron](https://arxiv.org/abs/2407.14679); [Sheared-LLaMA](https://arxiv.org/abs/2310.06694)). The result is a derivative of the parent's weights, which disqualifies it under the owner's provenance goal.
- **Logit distillation during pretraining.** This is the best per-token multiplier: Gemma 2's 2B student went from 60.3 to 67.7 average ([Gemma 2 report](https://arxiv.org/html/2408.00118)). But it costs about 1.6× compute even with a 0.6B teacher. It forces the teacher's tokenizer; Qwen's 151k vocabulary would consume a large share of a small model. And Apple's distillation scaling laws limit when it pays ([Busbridge et al.](https://proceedings.mlr.press/v267/busbridge25a.html)).
- **Licence.** Apache-2.0 teachers such as Qwen3 carry no obligations. A Gemma ≤3 teacher makes the student a "Model Derivative". A Llama 3.x teacher forces a "Llama…" name prefix, which is incompatible with `raghunath1/aptivra-360m-continual` ([Gemma Terms](https://ai.google.dev/gemma/terms); [Llama 3.2 licence](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/LICENSE)).

The recommended position is a **from-scratch base with no logit distillation in pretraining.** Post-training then uses teacher-generated data only from MIT- or Apache-licensed sources such as OpenThoughts3 and DeepSeek-R1 traces. R1's MIT licence explicitly permits distillation ([R1-Distill card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)).

### Architecture and data: copy the evidence, and keep risky novelty out of the one run

Every small-scale ablation points the same way: deep-and-thin layout, SwiGLU, GQA, tied embeddings, RMSNorm with QK-norm, and RoPE with a high base. Each choice moves zero-shot averages by roughly 0.2–1.3 points ([MobileLLM](https://arxiv.org/html/2402.14905v2); [Smol Training Playbook](https://huggingfacetb-smol-training-playbook.hf.space/)). The recommended shape is the SmolLM2/MobileLLM-360M shape:

- 32 layers, d_model 960, FFN 2560, GQA 15/5.
- Tied embeddings, RMSNorm + QK-norm, no biases, no weight decay on embeddings.
- RoPE θ ≈ 100k at a 4k pretraining context.
- An owned 49k–64k byte-level BPE trained on the target mixture. This keeps embeddings near 13% of parameters. A 151k vocabulary would take about a quarter of a 0.6B model ([SmolLM2 config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json); [Qwen3 config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json)). If the owner's text includes Indic scripts, the tokenizer must be multilingual and nearer 64k–100k.

IMU-1 is the closest open reference for this budget. It trained 430M parameters on 72B tokens with QK-norm, per-head gating, value residuals, NorMuon, μP, a three-stage schedule and checkpoint EMA, and claims to approach models trained on 56× more data ([IMU-1](https://arxiv.org/abs/2602.02522)). Its components should be ablated on Mac proxies and adopted only where they win.

For data, model-based quality filtering is the single largest lever: FineWeb-Edu gave +4 MMLU and +11 ARC over FineWeb at 1.7B/350B tokens ([FineWeb](https://arxiv.org/html/2406.17557)). The recipe is:

- **Stable phase:** about a 60/40 FineWeb-Edu/DCLM web mix plus 10–15% code.
- **WSD decay (final 10–20%):** shift towards Stack-Edu, FineMath4+ and a few percent Cosmopedia. FineMath4+ roughly doubled GSM8K, and at 360M almost all maths ability comes from that annealing data ([SmolLM2 paper](https://arxiv.org/html/2502.02737)).
- **Licences:** the web corpora are ODC-By or CC-BY with attribution ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu); [DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)). Stack-Edu carries per-file licences only, so it needs review ([Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)).

Hybrid convolution or SSM backbones (LFM2, Hymba, Qwen3.5's Gated DeltaNet) are the 2026 efficiency frontier. Hymba-1.5B reports an 11.67× smaller cache than Llama-3.2-3B ([Hymba](https://arxiv.org/abs/2411.13676)). Looped depth (Ouro) and memory layers are the most interesting reasoning and knowledge-capacity bets ([Ouro](https://arxiv.org/abs/2510.25741); [Memory Layers at Scale](https://arxiv.org/abs/2412.09764)). All three either lack sub-1B evidence at small token budgets or complicate on-device runtimes built around Llama-style decoders. They belong in Mac-scale ablations and later retrofits, not in the one irreplaceable cloud run.

## Facts go to memory and skills go to adapters: the second brain splits learning in two

The evidence on how a model should learn a user's world is unusually consistent:

- **Retrieval beats fine-tuning for new facts.** Retrieval raised Mistral-7B from 0.481 to **0.875** on post-cutoff events. Fine-tuning reached only 0.504–0.588. Raw-text fine-tuning cut Llama2-7B to 0.219 and damaged its later use of retrieval ([Ovadia et al.](https://arxiv.org/html/2312.05934v3)).
- **New facts trained into weights raise hallucination roughly linearly** as they are learned ([Gekhman et al.](https://arxiv.org/abs/2405.05904)).
- **Locate-and-edit methods collapse** under hundreds of sequential edits ([Gu et al.](https://arxiv.org/html/2401.04700v3); [AlphaEdit](https://arxiv.org/html/2410.02355v4)).
- **Unlearning from weights is unreliable.** TOFU found no baseline that unlearned effectively, and relearning attacks restore "forgotten" facts ([OpenUnlearning](https://arxiv.org/pdf/2506.12618)).

A misheard name trained into weights cannot be cleanly deleted. In a knowledge graph it is one row with provenance that can be edited, superseded or erased. That decides the architecture: **personal facts live outside the weights.**

Shipping products confirm the split. Apple and Gemini Nano use a frozen base with swappable LoRA adapters trained off-device by the vendor. Personalisation comes from on-device context and retrieval, and Apple states it does not train its foundation models on users' private data ([Apple ML Research](https://machinelearning.apple.com/research/introducing-apple-foundation-models); [Android Developers Blog](https://android-developers.googleblog.com/2023/12/a-new-foundation-for-ai-on-android.html)). Aptivra's on-device adapter training goes *beyond* shipped practice, which makes its verifier gate and rollback essential rather than decorative.

### The ADR-0011 design is right, with five corrections

The existing design is frozen base, per-skill LoRA, learned router, replay, and verifier-gated promotion with atomic rollback. The evidence supports it. LoRA "learns less and forgets less" than full fine-tuning ([Biderman et al.](https://arxiv.org/abs/2405.09673)), and frozen per-skill adapters cannot be overwritten. Five corrections follow from the research:

1. **Facts never go into adapters through SFT.** Adapters hold procedure, style, tool habits and domain familiarity.
2. **The router is now where forgetting happens.** Replay and the existing EWC work belong on the router. When an existing skill is refreshed, interference-aware subspace methods (InfLoRA: forgetting 2.25 vs O-LoRA's 8.05 on TRACE) are the best-evidenced choice ([GainLoRA reproduction](https://arxiv.org/pdf/2505.15424)).
3. **The promotion gate must test "adapter plus retrieval", not the adapter alone.** It must also include a general-ability canary set that the training loop never sees, and use a measured noise floor.
4. **Every training example needs provenance to every adapter version.** "Forget this" then means deleting the record and retraining or rolling back the affected adapter.
5. **A base-model upgrade invalidates every adapter.** Apple says adapters "must be retrained with each new version of the base model" ([Apple 2025 updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)). The raw, consented interaction log is the durable asset, not the adapters.

### Sleep-time consolidation must transform memories, not replay them

The complementary-learning-systems pattern has moved into 2025–2026 language-model papers: a fast episodic store during the day, slow consolidation while idle. Removing the "dreaming" phase in one study dropped SQuAD knowledge-incorporation accuracy from 46.2 to 36.2 ([Discovery by Dreaming](https://arxiv.org/pdf/2607.16256)). Verbatim replay is not enough.

Aptivra's nightly job should run in five steps:

1. Cluster the day's episodes.
2. Write facts, with timestamps and provenance, into the knowledge graph.
3. Generate transformed skill exercises and paraphrases. Accuracy rises monotonically with the number of paraphrases ([Ovadia et al.](https://arxiv.org/html/2312.05934v3)).
4. Train a LoRA with replay from a frozen general buffer.
5. Promote only past the gate.

For books the owner wants internalised, EntiGraph is the reference result. It expanded 1.3M source tokens into 455M synthetic tokens and raised closed-book QA from 39.49 to 56.22, recovering about 80% of the open-book gain, and its gains compound with retrieval ([EntiGraph](https://arxiv.org/pdf/2409.07431); [ICLR slides](https://iclr.cc/media/iclr-2025/Slides/31270_nBGAjHj.pdf)). It used GPT-4-class generation. A 360M generator will produce lower-quality synthetic data, and self-generated data from a weak model risks collapse unless real data is accumulated alongside it ([Gerstgrasser et al.](https://arxiv.org/abs/2404.01413)).

### The memory itself is where Aptivra can lead

The memory should be a fully local stack:

- An append-only episodic log.
- A typed, time-stamped knowledge graph of people, events, preferences and projects, with supersede edges for knowledge updates and an explicit abstention path.
- A vector index.
- A verified skill library of small programs (dates, budgets, units) in the Voyager pattern, which replaces neural arithmetic, the thing small models fail ([Voyager](https://arxiv.org/abs/2305.16291)).

LongMemEval's own fixes target exactly these failure modes: session decomposition, fact-augmented key expansion and time-aware query expansion ([LongMemEval](https://arxiv.org/html/2410.10813v2)).

Context budget matters on phones. CPU prefill runs at about 140–250 tok/s, so a 2k-token retrieved context costs 10–15 seconds without a GPU or NPU ([LiteRT-LM](https://developers.google.com/edge/litert-lm/overview)). Compact memory is therefore a hard requirement. Mem0 answers with about 1.8k context tokens against 26k for full context, with a 91% lower p95 latency, though it trails full context on accuracy (66.88 vs 72.90 J) and the benchmark is vendor-run ([Mem0](https://arxiv.org/html/2504.19413v1)).

Two 2025–2026 results suggest the memory *policy* is itself a learnable skill. Mem-alpha trains memory-write operations with RL and generalises from 30k to over 400k tokens ([Mem-alpha](https://arxiv.org/html/2509.25911v1)). "Forget to Improve" curates on-device memory by net value per byte and drove memory-injection success from 0.75 to 0 ([arXiv 2606.25115](https://arxiv.org/abs/2606.25115)).

### Learning from the internet and books

Pages and books the model reads become indexed corpus and knowledge-graph claims with source and confidence. Only procedural skills, such as how to read a balance sheet, are distilled into adapters, and only behind the gate. Ingested text is treated as data, never as instructions, because autonomous web ingestion opens a prompt-injection and poisoning channel.

### The out-of-the-box bet worth funding later

Sparse memory finetuning (SMF) writes new facts only into memory-layer slots that the new data activates far more than pretraining did. At equal knowledge gain it cut NaturalQuestions forgetting from **89% (full fine-tuning) and 71% (LoRA) to 11%** ([SMF](https://arxiv.org/abs/2510.15103)). A 2026 follow-up retrofits it onto Qwen-2.5-0.5B on consumer hardware ([Improving SMF](https://arxiv.org/abs/2604.05248)). Per-slot writes are auditable and potentially reversible per slot. That is the closest published thing to a weight-level fact store compatible with deletion.

Because retrofitting works, it does not need to risk the pretraining run. It is a Phase 6 experiment on the Mac. Ephemeral test-time training is a second cheap bet: one gradient step on each of about 20–50 retrieved neighbours cut bits per byte by about 20% for 117M–1.3B models ([Hardt and Sun](https://arxiv.org/pdf/2305.18466)). Training on retrieved personal chunks at query time and then discarding the update adapts the model without persisting anything.

## Reasoning, tools and RL sharpen a small model; they do not create its capabilities

Distillation, not RL, built every strong sub-1B reasoner. MobileLLM-R1 used pretraining, knowledge distillation and SFT with no RL stage ([MobileLLM-R1 card](https://huggingface.co/facebook/MobileLLM-R1-950M)). R1-Distill-Qwen-1.5B reached MATH500 83.9 from 800k R1 traces ([card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)). Small models learn poorly from long chain-of-thought. Models of 3B or fewer do better on short or mixed traces ([Small Model Learnability Gap](https://arxiv.org/abs/2502.12143)). "Less is more" results (s1, LIMO) are 32B results, and a 950M model needed 6.2M samples. Aptivra's reasoning stage should use hundreds of thousands of short, verified, permissively licensed traces (OpenThoughts3, OpenR1-Math, Llama-Nemotron), not 1k long ones ([OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M); [OpenR1-Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)).

Reinforcement learning with verifiable rewards is cheap and powerful *when the base already has latent ability*:

- **Cheap gains on a strong base.** Tina got a 1.5B reasoner to AIME24 43.3% with LoRA RL for about $9 ([Tina](https://arxiv.org/abs/2504.15777)). One-shot RLVR raised Qwen2.5-Math-1.5B's MATH500 from 36.0 to 73.6 from a single example, but only 8.6 points of that is beyond format correction ([Wang et al.](https://arxiv.org/abs/2504.20571)).
- **RL is bounded by the base.** RLVR models beat the base at pass@1 while the base wins at large k ([Yue et al.](https://arxiv.org/abs/2504.13837)).
- **Rewards can be spurious.** Random rewards gave Qwen2.5-Math-7B +21.4 points against +29.1 for true rewards ([Shao et al.](https://arxiv.org/abs/2506.10947)).
- **Very small bases fail.** TinyZero's 0.5B base failed to learn the task that 1.5B learned ([TinyZero](https://github.com/Jiayi-Pan/TinyZero)).

For a from-scratch 360M, RL is therefore a late sharpener. It runs only when SFT pass@k is non-trivial, as LoRA-GRPO in TRL. Every run includes random-reward and format-only controls compared against the noise floor ([TRL](https://huggingface.co/docs/trl/index)).

Learning from the user's own feedback should use KTO-style unpaired thumbs-up/down labels or hindsight self-distillation, not online PPO ([KTO](https://arxiv.org/abs/2402.01306)). It must never optimise approval alone. OpenAI rolled back a GPT-4o update in about four days because it leaned on "short-term feedback" and had not been tested for sycophancy ([TechCrunch](https://techcrunch.com/2025/04/29/openai-explains-why-chatgpt-became-too-sycophantic)).

Tool use at this size is a closed-world skill:

- **Single calls can be accurate.** Sub-2B models reach 69–83% on single-call AST benchmarks, and specialised training (xLAM, Hammer's function masking) matters more than size ([benchmarklist xLAM-2-1b](https://benchmarklist.com/models/xlam-2-1b-fc-r/); [Hammer](https://arxiv.org/abs/2410.04587)).
- **Multi-turn agents are out of reach on-device.** The design is a fixed personal tool set with constrained decoding, plus a consented escalation path for multi-step tasks.
- **Escalation can keep near-frontier quality.** FrugalGPT matched GPT-4 at up to 98% lower cost, and RouteLLM halved strong-model calls without quality loss ([FrugalGPT](https://arxiv.org/abs/2305.05176); [RouteLLM](https://arxiv.org/abs/2406.18665)). The local knowledge graph improves redaction, because it knows which tokens are personal.
- **Test-time compute does not translate to phones.** A 1B model matched 405B on MATH-500 at N=512 samples with a 7B+ reward model ([Liu et al.](https://arxiv.org/html/2502.06703)). On a phone, that is minutes of decoding. On-device, use small-N self-consistency and deterministic verifiers: tool results, schema checks, program execution.

## The buddy is a system property, and the phone is a client of the Mac

Persona and emotional tone cannot be entrusted to a 360M model's weights. LLaMA2-70B drifts significantly from its persona within 8 rounds, and larger models drifted *more* in a nine-model study ([Li et al.](https://arxiv.org/html/2402.10962v1); [Choi et al.](https://alphaxiv.org/abs/2412.00804)). EQ-Bench correlates r=0.97 with MMLU, so a small model will score low on it ([EQ-Bench](https://arxiv.org/html/2312.06281v2)). Prompted "I don't know" is unreliable in small models, while token probabilities carry the better signal ([Research Square](https://www.researchsquare.com/article/rs-11019638)).

The buddy therefore needs five system-level mechanisms:

- **A persona spec:** versioned, re-injected every turn, and checked with drift probes at 8, 16 and 32 turns.
- **Abstention from outside the model:** driven by retrieval support and token-probability thresholds, not by the model's own verbal confidence.
- **Inform-and-refer for medical, legal and financial questions:** cite the user's documents and route to a professional or a larger model with consent. A 2B model scores only 44% on MedQA ([rs-10198691](https://www.researchsquare.com/article/rs-10198691)).
- **A deterministic crisis path**, tuned for recall.
- **A sycophancy suite** (pushback, false premise, flattery) at every promotion.

The product history explains why:

- Replika's abrupt persona change caused measurable grief ([arXiv 2412.14190](https://arxiv.org/pdf/2412.14190)).
- Companion apps used manipulative farewells in about 37–43% of cases, lifting engagement up to 14× while raising churn intent ([Harvard D3](https://d3.harvard.edu/?p=28896)).
- Humane's cloud-dependent pin stopped working when the company sold to HP ([TechCrunch](https://techcrunch.com/2025/02/22/the-fallout-of-hps-humane-acquisition/)).

A local model with exportable, user-owned memory is the structural remedy for the last failure. Retention metrics should never be the optimisation target.

While Aptivra stays private to one user, companion statutes have little to bite on. If it is ever distributed, New York's AI-companion law covers memory-bearing companions explicitly. It requires AI disclosure every 3 hours and crisis referral ([Fenwick](https://www.fenwick.com/insights/publications/new-yorks-ai-companion-safeguard-law-takes-effect)). California SB 243 adds published crisis protocols and a private right of action ([Skadden](https://skadden.com/insights/publications/2025/10/new-california-companion-chatbot-law)).

On devices, inference is easy and training is not:

- **Inference speed.** Flagship phones decode 4-bit 0.5–1B models at about 25–60 tok/s, and 270M-class models reach 154–213 tok/s on a Galaxy S25 Ultra CPU ([LiteRT-LM](https://developers.google.com/edge/litert-lm/overview); [LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)).
- **Quantisation.** Small models lose disproportionately at 4-bit: Qwen3-0.6B lost about 10% MMLU under GPTQ, against about 1% at 14B ([Qwen3 quantisation review](https://liner.com/review/empirical-study-qwen3-quantization)). A 360M Aptivra should ship at 8-bit (about 400 MB), or run a QAT phase as Meta, Google and Apple do ([Meta AI blog](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)).
- **Phone training.** MobileFineTuner needed about 3.3 GB peak RSS for LoRA on GPT-2-355M and about 107 s per step for Qwen2.5-0.5B ([MobileFineTuner](https://arxiv.org/html/2512.08211v2)).
- **iPhone background limits.** iOS 26 grants no background GPU on any iPhone ([Apple Developer Forums](https://developer.apple.com/forums/thread/816774)).

The defensible topology makes the **M4 Max the "home brain"**. It runs MLX LoRA training, sleep consolidation and the canonical memory. Phones are clients that run GGUF, LiteRT-LM or ExecuTorch builds and pull promoted adapters and memory deltas over an end-to-end-encrypted sync between the owner's own devices. Overnight phone training while charging can follow later as an option.

## A phased roadmap with gated spend

Cloud items are framed as owner decisions; nothing in the cloud starts without the owner's explicit approval of that specific run. All cloud spend sits under a ₹24,000 ceiling: pilot ₹1,000, main run ₹20,000 hard cap, contingency ₹3,000. That leaves about ₹4,800 of the credit untouched as protection against card charges. The plan assumes a billing budget with alerts already exists (₹25,000 gross). An automated budget → Pub/Sub → disable-billing kill switch should also be installed before the first accelerator VM. That is a standard GCP pattern, not costed in the research notes.

| Phase | Where / when | Cost | Work | Exit gate (kill criteria in bold) |
|---|---|---|---|---|
| **0. Foundations** | Mac, weeks 0–3 | ₹0 | Superseding ADR (own base) and personal-Aptivra vision ADR; tokenizer; datatrove pipeline; lighteval cloze harness; reference trajectory from SmolLM2's released intermediate checkpoints ([HuggingFaceTB](https://huggingface.co/HuggingFaceTB)); 30–135M proxy ablations of IMU-1 components; JAX/MaxText (or PyTorch/XLA) smoke test; checkpoint-resume test | Proxy loss tracks the reference within the noise floor; resume is bit-exact. **Halt if the pipeline cannot reproduce a reference curve on a proxy.** |
| **1. Cloud readiness** | GCP, week 3–4 | ≤ ₹1,000 | Billing upgrade; quota for v6e spot (us-central1) plus L4 spot as fallback; data staged in-region; 1-hour v6e pilot measuring tok/s, MFU and preemption | **If measured 360M throughput is below ~90k tok/s, cut to 135M or fewer tokens. Below ~40k tok/s, abandon TPU for L4 + 135M. If quota is not granted by ~2026-11-20, train 135M on the Mac and redirect the credit to post-training.** |
| **2. Pretraining** | v6e spot, weeks 4–8 | ≤ ₹20,000 | 360M, WSD, staged mix, decay with FineMath4+/Stack-Edu/Cosmopedia, checkpoints to GCS; evals on Mac | Spend reviewed at 25/50/75/90%. **Halt on a non-recovering loss spike after two rollbacks. Halt if cloze HellaSwag at ~40B tokens trails the SmolLM2 reference trajectory beyond the noise floor. Stop at ≥120B tokens if the curve flattens. Finish by mid-December.** |
| **3. Post-training** | Mac, weeks 8–14 | ₹0 (≤ ₹3,000 contingency) | SFT on Apache/MIT/ODC-BY/CC-BY sets only (smol-smoltalk, everyday-conversations, Tulu 3, hermes-function-calling, Synth-APIGen, xLAM-60k, short OpenThoughts3 traces); DPO/APO; int8/QAT; GGUF and MLX exports | IFEval ≥ 41 (SmolLM2-360M-Instruct parity); 100% schema validity under constrained decoding; argument accuracy measured on the owner's tool set. **If IFEval is below 35, revisit the data before any RL.** |
| **4. Second brain** | Mac first | ₹0 | Episodic log, KG, vector index, skill library, sleep consolidation, provenance-linked deletion, persona spec, crisis path | LongMemEval-style suite on the owner's own data beats plain retrieval; deletion verified end-to-end. **Owner decision point: if a same-memory Apache base (e.g. Qwen3-0.6B) beats Aptivra-360M by a wide margin on personal tasks, decide between ownership and quality explicitly.** |
| **5. RL and feedback** | Mac | ₹0 | LoRA-GRPO on verifiable personal tasks (tool success, date maths); KTO from implicit signals | **Run RL only if SFT pass@8 on the target task is non-trivial. Drop it if gains do not beat random-reward and format-only controls beyond the noise floor.** |
| **6. Devices and research bets** | Phones; Mac | ₹0 | LiteRT-LM/ExecuTorch/llama.cpp clients with adapter sync; SMF memory-layer retrofit; ephemeral test-time training; Mem-alpha-style memory policy | Each bet ships only if it beats the Phase 4 system on the owner's suite with no canary regression. |

Eight decisions belong to the owner, in this order:

1. **Definition of "own".** From scratch, with teacher *data* allowed in post-training but no teacher logits or pruned weights, as recommended.
2. **Model size and repository name.** 360M primary, 135M fallback. A 135M outcome would make `aptivra-360m-continual` a misnomer.
3. **Billing upgrade to Paid.** This explicitly accepts card exposure beyond the credit and is mitigated by the kill switch.
4. **Accelerator and region.** v6e spot in us-central1 with the JAX/XLA engineering risk, or L4 spot on familiar PyTorch at about 6× fewer tokens.
5. **Spend cap and stop rules.** As tabled above.
6. **Licence posture.** Commercial-safe data only. This excludes PersonaHub, EmpatheticDialogues and APIGen-MT-5k, and reviews no-licence sets such as OpenHermes-2.5 ([HF dataset licences](https://huggingface.co/datasets/proj-persona/PersonaHub)).
7. **Cloud escalation of personal queries.** Allowed or not, and under what redaction and consent rules. Every threshold is registered as configuration.
8. **Release.** What, if anything, goes to the private Hub repo: base, instruct, quantised siblings. The base carries no `base_model` field, and derived repos point back to it ([Hub model cards](https://huggingface.co/docs/hub/model-cards)).

## Conclusion

The research reframes the goal. A personal model wins on its relationship to one person's data, not on its size class. The weights are the least personal part of the system and the most expensive to change. They need to be good enough to read, write, route and call tools. Everything that makes Aptivra *this user's* model belongs in places that are inspectable, deletable and portable across base-model upgrades: the memory, the gated adapters, the skill library and the consented interaction log. The pretraining budget should therefore be spent once, carefully, on the most tokens the credit can buy. Ambition belongs in the memory and consolidation layers, which cost nothing but Mac time.

Two uncertainties decide whether this plan holds, and both can be measured cheaply before commitment. The first is whether a single v6e chip delivers its theoretical throughput on a 360M model; a one-hour pilot settles it. The second is whether an owned 360M base with a strong memory system competes with an Apache-licensed 0.6B base on the owner's own life; a local LongMemEval-style suite settles that. If the second answer is badly negative, the honest move is to make ownership versus quality an explicit owner decision, not to rationalise it after the money is gone.
