# Small Language Models (~100M–1B): Architecture, Tokenizer and Pretraining Data — State of the Art 2023–2026

Research notes for choosing the architecture, tokenizer and data mixture of a new small base model trained from scratch for on-device use. Research date: 2026-10-18. All figures carry a citation. Numbers I derived by arithmetic are labelled **(computed)**. Anything recalled from memory but not re-verified in this session is in **Gaps**, not in Cited Findings.

---

## 1. Which small models lead their size class, and what are their exact configs and training-token counts?

### Takeaway
At 100M–1B parameters the leaders are trained far past Chinchilla-optimal (roughly 2T–36T tokens) on heavily filtered or educational data. The 2023 generation (TinyLlama, Pythia, OPT) is superseded. Three base-model families are open and documented well enough to reproduce: **SmolLM2** (135M / 360M / 1.7B, Apache-licensed weights, fully open data), **Qwen3-0.6B** (strongest published 0.6B base on knowledge and math, 36T tokens, data closed) and **MobileLLM** (Meta; deep-thin research recipe, non-commercial licence). Newer 2025 entrants are Gemma 3 270M/1B, MobileLLM-R1/Pro and LFM2 (a hybrid convolution–attention design).

### Cited Findings

**SmolLM2 (Hugging Face, Feb 2025)**
- SmolLM2-1.7B: 24 layers, d_model 2048, FFN 8192, 32 attention heads, SwiGLU, tied embeddings, vocabulary 49,152, RoPE θ=10k raised to 130k for the 8k context extension, 2M tokens/batch, Llama-2-style architecture, about 11T training tokens on 256 H100s — [SmolLM2 paper, arXiv 2502.02737](https://arxiv.org/html/2502.02737)
- SmolLM2-360M: trained on 4T tokens. SmolLM2-135M: trained on 2T tokens. Both use GQA and a WSD schedule with 20% decay and peak LR 3.0e-3 — [SmolLM2 paper](https://arxiv.org/html/2502.02737)
- SmolLM2-360M config: LlamaForCausalLM, 32 layers, hidden 960, intermediate 2560, 15 heads / 5 KV heads, vocabulary 49,152, max_position_embeddings 8192, rope_theta 100,000, RMSNorm eps 1e-5, tied embeddings — [config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json)
- SmolLM2-135M config: 30 layers, hidden 576, intermediate 1536, 9 heads / 3 KV heads, vocabulary 49,152, 8192 positions, rope_theta 100,000, tied embeddings — [config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json)
- SmolLM2-1.7B base scores vs Llama3.2-1B vs Qwen2.5-1.5B:

  | Benchmark | SmolLM2-1.7B | Llama3.2-1B | Qwen2.5-1.5B |
  |---|---|---|---|
  | HellaSwag | 68.7 | 61.2 | 66.4 |
  | ARC (avg) | 60.5 | 49.2 | 58.5 |
  | PIQA | 77.6 | 74.8 | 76.1 |
  | CommonsenseQA | 43.6 | 41.2 | 34.1 |
  | Winogrande | 59.4 | 57.8 | 59.3 |
  | OpenBookQA | 42.2 | 38.4 | 40.0 |
  | MMLU-Pro (held out) | 19.4 | 11.7 | 13.7 |
  | TriviaQA | 36.7 | 28.1 | 20.9 |
  | GSM8K (5-shot) | 31.1 | 7.6 | 61.7 |
  | MATH (4-shot) | 11.6 | 3.3 | 34.3 |
  | HumanEval | 22.6 | 18.9 | 37.2 |

  — [SmolLM2 paper, Table 4](https://arxiv.org/html/2502.02737)
- The paper gives no full per-benchmark tables for 135M or 360M in the portion read. It only claims they are state of the art for their size — [SmolLM2 paper](https://arxiv.org/html/2502.02737)

**SmolLM3 (Hugging Face, Jul 2025), 3B: above the target range but the best-documented recipe**
- Llama-style decoder with tied embeddings, GQA with 4 groups, and NoPE (RoPE removed) in every 4th layer. Uses intra-document masking and no weight decay on embeddings (following OLMo 2) — [SmolLM3 blog](https://huggingface.co/blog/smollm3)
- About 11.2T pretraining tokens, then 100B long-context tokens in two stages (RoPE θ raised to 1.5M, then 5M; trained to 64k, 128k at inference via YaRN), then 35B reasoning mid-training tokens. Global batch 2.36M tokens, sequence length 4096, peak LR 2e-4, AdamW (0.9, 0.95), weight decay 0.1, WSD with 2000 warmup steps and linear decay over the last 10%. Trained on 384 H100s for 24 days — [SmolLM3 blog](https://huggingface.co/blog/smollm3)
- Web/code/math mix: Stage 1 (0–8T) 85/12/3; Stage 2 (8–10T) 75/15/10; Stage 3 (10–11.1T) 63/24/13. Multilingual share of web is 12% in every stage — [SmolLM3 blog](https://huggingface.co/blog/smollm3)
- Reasoning results, think vs no-think mode: AIME25 36.7 vs 9.3; LiveCodeBench 30.0 vs 15.2; GPQA-Diamond 41.7 vs 35.7. Hugging Face says the model beats other 3B models and is competitive with Qwen3-4B and Gemma3-4B, but the post gives no numeric base table — [SmolLM3 blog](https://huggingface.co/blog/smollm3)

**Qwen3-0.6B / 1.7B (Alibaba, May 2025)**
- Qwen3-0.6B-Base config: 28 layers, hidden 1024, intermediate 3072, 16 Q heads / 8 KV heads, head_dim 128 (so attention width 2048 > hidden 1024), SiLU, rope_theta 1,000,000, max_position 32,768, tied embeddings, vocab_size 151,936, RMSNorm eps 1e-6, no attention bias — [config.json](https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json)
- Report Table 1: 0.6B and 1.7B both have 28 layers, 16/8 heads, tied embeddings and 32K context. The tokenizer is byte-level BPE with a 151,669-token vocabulary. Qwen3 drops Qwen2's QKV bias and adds QK-Norm for stability — [Qwen3 Technical Report, arXiv 2505.09388](https://arxiv.org/html/2505.09388)
- Pretraining: about 36T tokens in 119 languages. Stage 1 is general (about 30T at 4,096 seq len). Stage 2 is reasoning (about 5T of higher-quality STEM, code, reasoning and synthetic data). Stage 3 is long context (hundreds of billions of tokens at 32,768) — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388)
- Qwen3-0.6B has 0.44B non-embedding parameters (secondary model-hosting page restating the official card) — [friendli.ai Qwen3-0.6B-Base](https://friendli.ai/models/Qwen/Qwen3-0.6B-Base)
- Base model scores (report Table 8):

  | Benchmark | Qwen2.5-0.5B | Qwen3-0.6B | Gemma-3-1B | Qwen2.5-1.5B | Qwen3-1.7B |
  |---|---|---|---|---|---|
  | MMLU | 47.50 | 52.81 | 26.26 | 60.90 | 62.63 |
  | MMLU-Pro | 15.69 | 24.74 | 9.72 | 28.53 | 36.76 |
  | GSM8K | 41.62 | 59.59 | 2.20 | 68.54 | 75.44 |
  | MATH | 19.48 | 32.44 | 3.66 | 35.00 | 43.50 |
  | EvalPlus | 31.85 | 36.23 | 8.98 | 44.80 | 52.70 |
  | MGSM | 12.07 | 30.99 | 1.74 | 32.82 | 50.71 |

  — [Qwen3 Technical Report, Table 8](https://arxiv.org/html/2505.09388)

**Gemma 3 (Google, Mar 2025; 270M added Aug 2025)**
- Gemma 3 1B: 302M embedding + 698M non-embedding parameters, no vision encoder. Trained on 2T tokens (4B: 4T; 12B: 12T; 27B: 14T). Uses the 262k Gemini 2.0 tokenizer. Interleaves 5 local (sliding-window 1024) layers per global layer. Global RoPE base 1M, local 10k. QK-norm replaces soft-capping. Context 32K for 1B (128K for the larger sizes). All sizes are trained by knowledge distillation (256 teacher logits sampled per token) — [Gemma 3 Technical Report, arXiv 2503.19786](https://arxiv.org/html/2503.19786)
- Gemma 3 1B PT scores: HellaSwag 62.3, ARC-C 38.4, ARC-E 73.0, BoolQ 63.2, PIQA 73.8, SIQA 48.9, WinoGrande 58.2, BBH 28.4, DROP 42.4. The report gives no PT MMLU/GSM8K for 1B. IT 1B scores: MMLU 38.8, GSM8K 62.8, MATH 48.0, HumanEval 41.5 — [Gemma 3 report, Tables 9 & 18](https://arxiv.org/html/2503.19786)
- Conflict: Qwen3's report evaluates Gemma-3-1B base at MMLU 26.26 and GSM8K 2.20 — [Qwen3 report](https://arxiv.org/html/2505.09388). Gemma's own report omits these for the 1B PT model. Treat cross-lab baseline numbers with caution (harness and prompt differences).
- Gemma 3 270M: about 170M embedding parameters + about 100M transformer-block parameters, 256k vocabulary. On a Pixel 9 Pro the INT4 model used 0.75% battery over 25 conversations. The blog gives no training-token count or numeric IFEval — [Google Developers Blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)

**MobileLLM family (Meta)**
- MobileLLM (ICML 2024) final configs: 125M = 30 layers, 9 heads / 3 KV heads, dim 576, FFN 1536 (124.6M). 350M = 32 layers, 15 heads / 5 KV heads, dim 960, FFN 2560 (345.3M). Final models trained on 1T tokens (exploration runs used 0.25T) on 32 A100s with LR 2e-3 cosine — [MobileLLM, arXiv 2402.14905](https://arxiv.org/html/2402.14905v2). SmolLM2-135M/360M reuse exactly these shapes (compare the configs above).
- Zero-shot commonsense average: MobileLLM-125M 46.3, LS-125M 47.0, vs Pythia-160M 42.5, OPT-125M 42.6, RWKV-169M 43.6. MobileLLM-350M 51.3, LS-350M 52.1, vs Pythia-410M 46.6, RWKV-430M 47.0, BLOOM-560M 44.2 — [MobileLLM Table 3](https://arxiv.org/html/2402.14905v2)
- MobileLLM-R1-950M (Sep 2025, reasoning-focused): reported as 22 layers, 24 heads / 6 KV heads, dim 1536, MLP 6144, 4K context (32K after post-training). This comes from secondary sources; I did not see the official config — [Ritvik Rastogi summary](https://ritvik19.medium.com/papers-explained-469-mobilellm-r1-00d456b46e5e). About 2T curated tokens were resampled into a 4.2T-token pretraining run — [MobileLLM-R1, arXiv 2509.24945 (via opentrain.ai)](https://www.opentrain.ai/papers/mobilellm-r1-exploring-the-limits-of-sub-billion-language-model-reasoners-with-o--arxiv-2509.24945). Licence: FAIR Noncommercial Research License v1 — [OSAI index](https://osai-index.eu/model/mobilellm)
- MobileLLM-Pro (Oct/Nov 2025, about 1.08B params): pretrained on under 2T fully open-source tokens with distillation from a Llama 4-Scout teacher, 128k context. Meta claims about +5.7% over Gemma 3 1B and +7.9% over Llama 3.2 1B on average. ARC-c is 52.62 vs 38.40 (Gemma 3 1B) and 38.28 (Llama 3.2 1B). Licence is FAIR non-commercial — [MobileLLM-Pro model card](https://huggingface.co/facebook/MobileLLM-Pro-base/blob/main/README.md); [MobileLLM-Pro Technical Report, arXiv 2511.06719](https://arxiv.org/pdf/2511.06719)

**LFM2 (Liquid AI, Jul 2025): hybrid, non-Transformer-only**
- 350M/700M/1.2B share a 16-block layout: 10 double-gated short-range convolution blocks + 6 GQA blocks, 32K context, 10T training tokens. Actual parameter counts are 354M, 742M and 1.17B — [LFM2-1.2B model card](https://huggingface.co/LiquidAI/LFM2-1.2B); [LFM2 Technical Report, arXiv 2511.23404](https://arxiv.org/pdf/2511.23404)
- Vendor-reported post-trained MMLU / GPQA / IFEval / IFBench: 350M 43.43/27.46/65.12/16.41; 700M 49.9/28.48/72.23/20.56; 1.2B 55.23/31.47/74.89/20.7 — [Liquid AI blog](https://liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)

**Phi-4-mini (Microsoft, Feb 2025), 3.8B: above range, reference for synthetic-heavy data**
- 32 layers, hidden 3072, 24 Q / 8 KV heads, 200,064 vocabulary (o200k_base), tied embeddings, 128K context via LongRoPE, fractional RoPE (25% of head dim position-agnostic). Trained on 5T tokens of filtered web plus synthetic data (reused from Phi-4) with more math, code and reasoning — [Phi-4-Mini Technical Report, arXiv 2503.01743](https://arxiv.org/html/2503.01743)
- Reported scores: MMLU 67.3 (5-shot), GSM8K 88.6 (8-shot CoT), MATH 64.0, ARC-C 83.7, HellaSwag 69.1. The report does not clearly say whether these are base or instruct scores — [Phi-4-mini report Table 7](https://arxiv.org/html/2503.01743)

**MiniCPM (OpenBMB, 2024)**
- 1.2B and 2.4B non-embedding-parameter models. Introduced the WSD (Warmup-Stable-Decay) scheduler and "model wind tunnel" hyperparameter search on small proxies. Found a compute-optimal data:model ratio "much higher than Chinchilla optimal." The authors claim parity with 7B–13B models — [MiniCPM, arXiv 2404.06395](https://arxiv.org/abs/2404.06395)

### Inferences
- **Tokens per parameter (computed)** run far above Chinchilla's about 20: SmolLM2-135M ≈ 14,800 (2T/135M), SmolLM2-360M ≈ 11,100, SmolLM2-1.7B ≈ 6,500, Gemma 3 1B ≈ 2,000, LFM2-350M ≈ 28,000, MobileLLM-125M ≈ 8,000. The data budget, not parameter count, is the main lever for on-device models, because inference cost is fixed by size.
- For a fully open, reproducible base model at 135M–360M, the SmolLM2/MobileLLM shape (30–32 layers, dim 576–960, 3:1 GQA) is the evidence-backed default. Qwen3-0.6B's 28 layers × 1024 with a 151k vocabulary is the stronger 0.6B, but its data is closed and about 26% of its parameters are embeddings (computed below).
- Distillation from a large teacher (Gemma 3, MobileLLM-Pro) is now standard at the frontier of this class. A from-scratch project without teacher logits should expect to trail those models at equal token count.

### Gaps
- Official per-benchmark tables for SmolLM2-135M/360M (MMLU-cloze, ARC, HellaSwag, GSM8K) were not in the portion of the paper read. The model cards likely carry them; not verified.
- Gemma 3 270M layer count, width, head counts and training tokens are not in the blog. I did not fetch its config.json. (From memory, unverified: 18 layers, 640 hidden, 6T tokens. Verify before use.)
- Qwen2.5-0.5B config (24 layers, 896 hidden, 14/2 heads, 18T tokens) is recalled, not verified this session.
- **Superseded models, not re-verified this session.** TinyLlama-1.1B (22 layers, 2048 hidden, 32/4 heads, 32k vocab, about 3T tokens, 2024). Pythia (160M–12B, 300B tokens of the Pile, 2023). OpenELM (Apple, 2024, layer-wise scaling, 270M–3B, about 1.8T tokens). Phi-3-mini (3.8B, 3.3T tokens, 32k vocab). These figures are from memory; cite the primary sources before reusing. All are superseded by SmolLM2/Qwen3/Gemma 3 at equal size.
- I did not read the official MobileLLM-R1 config.json or its benchmark table. The R1 architecture numbers above are secondary-source.
- No 2026-dated small base model was found in these searches. The release survey may be incomplete for 2026.

---

## 2. Which architecture choices matter at small scale, and what is the evidence?

### Takeaway
At ≤1B parameters, the consensus with measured evidence is:
- deep-and-thin layout (about 30 layers at 125M–360M)
- SwiGLU
- GQA (KV heads about 1/3–1/2 of Q heads)
- tied input/output embeddings
- RMSNorm with QK-norm
- RoPE with a high base (100k–1M), plus later extension

Each choice moves zero-shot averages by roughly 0.2–1.3 points. The biggest structural decision is how much of the parameter budget goes to embeddings versus Transformer blocks.

### Cited Findings
- **Depth vs width (MobileLLM, about 125M, 0.25T tokens).** 12 layers × 768 → 43.9 avg; 30 layers × 512 → 44.8 (+0.9); 42 layers × 448 → 44.5. At 350M, 15 → 32 layers: 48.7 → 49.8 (+1.1). The paper calls these "significantly better," but the measured gain is about 1 point — [MobileLLM Tables 10–11](https://arxiv.org/html/2402.14905v2)
- **SwiGLU vs vanilla FFN:** +1.3 avg at both 125M (42.6 → 43.9) and 350M (47.4 → 48.7) — [MobileLLM Table 10](https://arxiv.org/html/2402.14905v2)
- **Embedding sharing (tying):** at 125M, sharing saves about 16M parameters (11.8%) at a cost of −0.2 avg (44.8 → 44.6). Reinvesting the saving in depth (32 layers) gives 45.0 at 125M parameters. Embeddings are over 20% of a 125M model's parameters — [MobileLLM Table 1](https://arxiv.org/html/2402.14905v2)
- **GQA:** +0.4 at 125M (44.6 → 45.0), +0.7 at 350M (49.2 → 49.9) when the saved parameters are reinvested. Head ablation: 16Q/4KV gives 44.7 vs 16/16 at 44.6 (125M), and 49.4 vs 49.6 (350M) — [MobileLLM Table 10, App. D](https://arxiv.org/html/2402.14905v2). SmolLM3 ablations at 3B: GQA (4 groups) matches MHA quality while shrinking the KV cache — [SmolLM3 blog](https://huggingface.co/blog/smollm3)
- **Layer sharing (immediate block-wise repeat, MobileLLM-LS):** +1.1 at 0.25T tokens and +0.7 / +0.8 at 1T tokens (125M / 350M). It adds no parameters, only latency — [MobileLLM Tables 3 & 10](https://arxiv.org/html/2402.14905v2)
- **Normalization/stability:** Qwen3 removes QKV bias and adds QK-Norm "for stable training" — [Qwen3 report](https://arxiv.org/html/2505.09388). Gemma 3 replaces soft-capping with QK-norm — [Gemma 3 report](https://arxiv.org/html/2503.19786). SmolLM3 found that removing weight decay from embeddings improves stability without hurting quality — [SmolLM3 blog](https://huggingface.co/blog/smollm3)
- **RoPE settings:**
  - SmolLM2-135M/360M ship rope_theta 100k with 8k positions — [135M config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/raw/main/config.json); [360M config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M/raw/main/config.json)
  - SmolLM2-1.7B used 10k, raised to 130k for the 8k extension — [SmolLM2 paper](https://arxiv.org/html/2502.02737)
  - Qwen3-0.6B uses 1M at 32k — [config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json)
  - Gemma 3 uses 1M on global layers and 10k on local layers — [Gemma 3 report](https://arxiv.org/html/2503.19786)
  - SmolLM3 removes RoPE in every 4th layer (NoPE), which improved long-context results with no short-context loss — [SmolLM3 blog](https://huggingface.co/blog/smollm3)
  - Phi-4-mini applies RoPE to only 75% of each head's dimensions — [Phi-4-mini report](https://arxiv.org/html/2503.01743)
- **Context length / KV-memory designs:**
  - Gemma 3 uses a 5:1 local (1024-token sliding window) to global ratio to cut KV memory — [Gemma 3 report](https://arxiv.org/html/2503.19786)
  - LFM2 replaces 10 of 16 attention blocks with short convolutions — [LFM2 card](https://huggingface.co/LiquidAI/LFM2-1.2B)
  - SmolLM3 uses intra-document masking to stabilise long-context training — [SmolLM3 blog](https://huggingface.co/blog/smollm3)

### Inferences
- **Embedding share of total parameters (computed, vocab × d_model with tied embeddings):**

  | Model | Embedding params | Share of total |
  |---|---|---|
  | SmolLM2-135M | 49,152 × 576 ≈ 28.3M | ≈ 21% |
  | SmolLM2-360M | 49,152 × 960 ≈ 47.2M | ≈ 13% |
  | Qwen3-0.6B | 151,936 × 1024 ≈ 155.6M | ≈ 26% (matches the 0.44B non-embedding figure) |
  | Gemma 3 1B | 302M (reported) | ≈ 30% |
  | Gemma 3 270M | 170M (reported) | ≈ 63% |
  | Phi-4-mini | 200,064 × 3072 ≈ 615M | ≈ 16% |

  At ≤360M, a 150k–262k vocabulary consumes most of the budget. Gemma 3 270M is effectively a 100M-parameter Transformer with a large lookup table.
- **KV cache per token (computed, bf16, 2 × layers × kv_heads × head_dim × 2 bytes):**

  | Model | Per token | At context |
  |---|---|---|
  | SmolLM2-135M (head_dim 64, 3 KV, 30 layers) | ≈ 23 KB | ≈ 189 MB at 8k |
  | SmolLM2-360M (head_dim 64, 5 KV, 32 layers) | ≈ 41 KB | ≈ 335 MB at 8k |
  | Qwen3-0.6B (head_dim 128, 8 KV, 28 layers) | ≈ 115 KB | ≈ 3.7 GB at 32k |

  On phones the KV budget, not the weights, can limit usable context. This argues for 3:1–4:1 GQA, sliding-window/hybrid layers, or a modest context target (4k–8k).
- **Latency caveat.** Deep-thin layouts raise sequential kernel launches per token. On some mobile runtimes 30+ thin layers may decode slower than 16–24 wider layers at equal parameters. LFM2 (16 blocks) and Qwen3 (28) are less extreme than MobileLLM. Benchmark on the target device; this is an inference, not measured here.
- **Recommended default for a 300–400M from-scratch base (inference from the evidence above):**
  - about 30–32 layers, d_model 960–1024, SwiGLU FFN ≈ 2.67–3× d_model
  - GQA 15/5 or 16/4–16/8
  - tied embeddings, RMSNorm + QK-norm, no biases, no weight decay on embeddings
  - RoPE θ ≈ 100k at a 4k–8k pretraining context, extended later; optionally NoPE in every 4th layer

### Gaps
- No controlled small-scale (≤500M) ablation of QK-norm, NoPE or the local:global ratio was found. The evidence comes from 3B+ models or is asserted without small-model tables.
- OpenELM's layer-wise (non-uniform) scaling evidence was not fetched.
- I found no published on-device latency comparison of deep-thin vs wide-shallow at equal parameters.

---

## 3. What vocabulary size suits a small model, given the embedding cost?

### Takeaway
Scaling-law work shows the compute-optimal vocabulary grows with model size, and most large models are under-vocabularied. At 100M–400M, every 50k of vocabulary at d=576–1024 costs 29–51M parameters **(computed)**, so small open models cluster at **~49k (SmolLM2)** when English-centric. Large vocabularies (151k Qwen, 200k Phi, 262k Gemma) are paid for with tied embeddings and justified by multilingual coverage, better compression, or distillation from a teacher sharing that tokenizer.

### Cited Findings
- "Scaling Laws with Vocabulary" (Tao et al., NeurIPS 2024): trained 33M–3B models on up to 500B characters. Three methods (IsoFLOPs, derivative estimation, parametric loss fit) agree that optimal vocabulary grows with compute and that most LLMs use too-small vocabularies. Llama2-70B's optimum is ≥216K vs its actual 32K. At 3B / 2.3e21 FLOPs, 32K → 43K raised ARC-Challenge from 29.1 to 32.0 — [arXiv 2407.13623](https://arxiv.org/abs/2407.13623)
- Vocabulary sizes in practice:
  - SmolLM2: 49,152 — [SmolLM2 paper](https://arxiv.org/html/2502.02737)
  - Qwen3: 151,669 tokens (config vocab_size 151,936) — [Qwen3 report](https://arxiv.org/html/2505.09388); [config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/raw/main/config.json)
  - Gemma 3: 262k — [Gemma 3 report](https://arxiv.org/html/2503.19786)
  - Gemma 3 270M: 256k — [Google blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)
  - Phi-4-mini: 200,064 (o200k_base) — [Phi-4-mini report](https://arxiv.org/html/2503.01743)
  - MobileLLM's embedding analysis uses a 32k vocabulary — [MobileLLM](https://arxiv.org/html/2402.14905v2)
- Multilingual: Qwen3 covers 119 languages with its 151k byte-level BPE — [Qwen3 report](https://arxiv.org/html/2505.09388). SmolLM3 keeps 12% multilingual web data in every stage — [SmolLM3 blog](https://huggingface.co/blog/smollm3). Google pitches Gemma 3 270M's 256k vocabulary as enabling fine-tuning on rare and specific tokens — [Google blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)

### Inferences
- For an English(+code)-first personal on-device model at ~360M, a 32k–64k byte-level BPE trained on the target mixture is defensible. It keeps embeddings at ≤15% of parameters. Tao et al.'s trend implies the optimum at ~100M–400M lies well below 100k. Their exact fitted optimum for these sizes was not extracted, so the range is an inference.
- If the user's on-device text is multilingual (e.g. Indic scripts), a 49k English-centric tokenizer will fragment it badly. In that case, choose a vocabulary of about 64k–100k trained on a multilingual mixture, or reuse an existing tokenizer and accept the embedding cost.
- Reusing an existing tokenizer (e.g. SmolLM2's 49k) has a practical benefit: tooling compatibility and the option of distilling from a same-tokenizer teacher.

### Gaps
- The exact compute-optimal vocabulary Tao et al. predict for ~100M–500M models was not extracted (the abstract only gives the 70B example).
- I found no direct small-model study measuring tokenizer fertility for Indic or other non-Latin languages vs vocabulary size.

---

## 4. Data quality vs quantity: what did FineWeb, FineWeb-Edu, DCLM, Cosmopedia, synthetic data, Stack-Edu, FineMath and WSD annealing each measurably contribute?

### Takeaway
Model-based quality filtering is the single largest data lever. FineWeb-Edu gives +4 MMLU and +11 ARC over FineWeb at 1.7B / 350B tokens. FineWeb-Edu favours knowledge and reasoning (MMLU, ARC), while DCLM favours HellaSwag, so the best recipes mix them, about 60/40. Domain-filtered code (Stack-Edu) and math (FineMath) add +4–8 points on code and roughly 2× GSM8K. They are most effective when concentrated in a WSD decay/annealing phase at the end of training.

### Cited Findings
- **FineWeb:** about 15T tokens from 96 Common Crawl snapshots. **FineWeb-Edu:** about 1.3T tokens, keeping pages with educational score ≥3 (0–5 scale). Scores come from a linear regressor on Snowflake-arctic-embed-m embeddings, trained on 410k of 460k Llama-3-70B-Instruct annotations (F1 82% at threshold 3). At 1.71B params / 350B tokens, FineWeb-Edu raised MMLU 33% → 37% and ARC 46% → 57% vs FineWeb — [FineWeb paper, arXiv 2406.17557](https://arxiv.org/html/2406.17557)
- **DCLM:** DCLM-Pool is 240T Common Crawl tokens. The DCLM-Baseline 7B model, trained on 2.6T tokens, reaches 64% 5-shot MMLU: +6.6 over MAP-Neo with 40% less compute, and comparable to Llama 3 8B (66%) with 6.6× less compute. The authors conclude "model-based filtering is key" — [DCLM, arXiv 2406.11794](https://arxiv.org/abs/2406.11794)
- **FineWeb-Edu vs DCLM vs mix (SmolLM2 ablation, 350B tokens):**

  | Benchmark | FW-Edu | DCLM | 60/40 mix |
  |---|---|---|---|
  | MMLU | 37.5 | 35.5 | 37.0 |
  | ARC | 57.5 | 53.5 | 56.0 |
  | HellaSwag | 60.1 | 62.3 | 62.2 |

  The mix keeps most of each dataset's strength — [SmolLM2 paper Table 1](https://arxiv.org/html/2502.02737)
- **Stack-Edu:** about 125B tokens across 15 languages, filtered from about 450B (Python 50.6B → 21.8B). MultiPL-E gains: Python 20.7 → 25.6, C++ 16.7 → 24.8, JS 18.2 → 22.4, Java 17.6 → 22.7 — [SmolLM2 paper](https://arxiv.org/html/2502.02737)
- **FineMath:** FineMath4+ is 10B tokens (6.7M docs); FineMath3+ is 34B. FineMath4+ gives about 2× GSM8K and about 6× MATH vs InfiMM-WebMath. InfiMM-WebMath beats OpenWebMath on GSM8K (14% vs 10%) — [SmolLM2 paper](https://arxiv.org/html/2502.02737)
- **Multi-stage schedule (SmolLM2-1.7B, 11T):**

  | Stage | Tokens | Web | Code | Math | Other |
  |---|---|---|---|---|---|
  | 1 | 0–6T | FW-Edu/DCLM 60/40 | 10% StarCoderData | none | |
  | 2 | 6–8T | 75% | 20% | 5% OpenWebMath | |
  | 3 | 8–10T | FW-Edu/DCLM shifts to 40/60 | switches to Stack-Edu + Jupyter | ~10% | |
  | 4 (decay) | 10–11T | 58% (DCLM-heavy) | 24% Stack-Edu | 14% (InfiWebMath-3+, FineMath 4+, …) | 4% Cosmopedia v2 |

  The 8k extension then adds 40% long documents (Dolma books 20%, DCLM 10%, FW-Edu 10%). Optimizer: AdamW (0.9, 0.95), 2000 warmup steps, LR 5e-4 stable, decay to 0 over the final 10% — [SmolLM2 paper](https://arxiv.org/html/2502.02737)
- **WSD schedule:** introduced by MiniCPM to allow continuous training and cheap data-scaling studies, since checkpoints from the stable phase can be decayed on demand — [MiniCPM](https://arxiv.org/abs/2404.06395). It is used by SmolLM2 (10–20% decay) and SmolLM3 (10% decay) — [SmolLM2](https://arxiv.org/html/2502.02737); [SmolLM3](https://huggingface.co/blog/smollm3)
- **Staged "quality at the end" is universal.** Qwen3 Stage 2 adds about 5T tokens of higher-quality STEM/code/reasoning/synthetic data — [Qwen3 report](https://arxiv.org/html/2505.09388). SmolLM3 raises code and math from 12%/3% to 24%/13% across stages — [SmolLM3 blog](https://huggingface.co/blog/smollm3)
- **Quality beats quantity at sub-billion scale:** MobileLLM-R1 reports that only about 2T high-quality tokens (resampled to 4.2T) were sufficient for strong reasoning — [MobileLLM-R1 abstract via opentrain.ai](https://www.opentrain.ai/papers/mobilellm-r1-exploring-the-limits-of-sub-billion-language-model-reasoners-with-o--arxiv-2509.24945). MobileLLM-Pro trained on under 2T fully open tokens (with distillation) and beat Gemma 3 1B (2T tokens) — [MobileLLM-Pro card](https://huggingface.co/facebook/MobileLLM-Pro-base/blob/main/README.md)
- **Synthetic textbook data:** Phi-4-mini's 5T corpus combines quality-classifier-filtered web data with synthetic data reused from Phi-4 — [Phi-4-mini report](https://arxiv.org/html/2503.01743). SmolLM2 uses Cosmopedia v2 (synthetic textbooks) only at 4% in the decay stage — [SmolLM2 paper](https://arxiv.org/html/2502.02737)

### Inferences
- **Practical recipe for a from-scratch 135M–400M model, derived from SmolLM2:**
  - Stable phase: FineWeb-Edu/DCLM about 60/40 web plus 10–15% code.
  - Final 10–20% (WSD decay): shift to Stack-Edu, FineMath4+, InfiWebMath and a few percent Cosmopedia v2.
  - Budget: 1–4T tokens. SmolLM2-360M used 4T; MobileLLM-R1 suggests ~2T well-curated tokens can be enough.
- FineWeb-Edu is only about 1.3T tokens, so multi-trillion budgets require either epoching (MobileLLM-R1 resampled ~2× without reported harm) or the larger FineWeb-Edu score-2 / DCLM pools.
- At ≤360M, measurable GSM8K/MATH ability comes almost entirely from decay-phase math data. Expect low absolute GSM8K for 135M–360M base models regardless.

### Gaps
- I did not retrieve the measured impact of Cosmopedia alone on small-model benchmarks (the original Cosmopedia/SmolLM v1 blog).
- I did not retrieve the original Phi-1 "Textbooks Are All You Need" numbers (1.3B, about 7B tokens, HumanEval about 50.6%; recalled, unverified).
- I found no isolated ablation of WSD vs cosine at ≤500M in the sources read. MiniCPM's own curves were not extracted.
- The DCLM fastText classifier details (OH-2.5 + ELI5 positives) are recalled, not confirmed this session.

---

## 5. What licences cover the main open pretraining datasets, and what restrictions apply to synthetic data generated by other models?

### Takeaway
The core web corpora are permissive with attribution: FineWeb/FineWeb-Edu and SmolLM-Corpus are **ODC-By 1.0**, also subject to Common Crawl terms. The licence risk sits in code (per-file upstream licences and opt-outs in The Stack lineage) and in synthetic data. That data inherits the generating model's terms (e.g. Mixtral for Cosmopedia, Llama-3-70B annotations for FineWeb-Edu scores, Qwen3-32B traces in SmolLM3 SFT). The open-weight small models themselves range from Apache-2.0 (SmolLM2) to non-commercial (MobileLLM-R1/Pro).

### Cited Findings
- FineWeb and FineWeb-Edu: ODC-By v1.0, plus Common Crawl terms of use — [FineWeb paper](https://arxiv.org/html/2406.17557)
- SmolLM-Corpus (cosmopedia-v2, python-edu, fineweb-edu-dedup subsets): tagged odc-by on the Hub — [smollm-corpus tree](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/tree/c074f3d3783ef8c321b40fd89088e5955cd05bad)
- FineWeb-Edu's quality labels were generated by Llama-3-70B-Instruct — [FineWeb paper](https://arxiv.org/html/2406.17557). SmolLM3's SFT reasoning traces were generated by Qwen3-32B — [SmolLM3 blog](https://huggingface.co/blog/smollm3)
- MobileLLM-R1 and MobileLLM-Pro weights: FAIR Noncommercial Research License — [OSAI index](https://osai-index.eu/model/mobilellm); [MobileLLM-Pro card](https://huggingface.co/facebook/MobileLLM-Pro-base/blob/main/README.md)

### Inferences
- For a personal, non-redistributed model, ODC-By corpora pose minimal risk. If the model is published (the HF release plan), keep a dataset attribution list in the model card.
- Distilling from or generating data with Llama-family models brings Llama licence terms, including naming and acceptable-use clauses. Qwen (Apache-2.0 for most sizes) and Mixtral (Apache-2.0) are lower-friction teachers. Verify each teacher's current licence before generating synthetic data.

### Gaps
- Not verified this session (recalled, must be checked on the dataset cards before adoption):
  - DCLM-Baseline licence (believed CC-BY-4.0)
  - Stack-Edu / The Stack v2 terms (per-file licences, opt-out, SWH terms)
  - FineMath licence (believed ODC-By)
  - Cosmopedia licence and its Mixtral-8x7B-Instruct generation
  - Dolma licence (ODC-By)
  - Gemma 3 terms of use and their clauses on "model derivatives" trained on Gemma outputs (relevant if Gemma is used as a teacher)
  - Qwen3 Apache-2.0 status
- I found no authoritative legal analysis of copyright status for synthetic text generated by open-weight models. This remains an open legal question.
