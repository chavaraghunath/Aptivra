# Pretraining budget and distillation: what ~₹28,000 (~US$330) of GCP credit plus an M4 Max 36GB buys

Research date: 2026-10-18. Prices come from the Google Cloud Billing Catalog API (`cloudbilling.googleapis.com/v1/services/{6F81-5844-456A Compute Engine, E000-3F24-B8AA Cloud TPU}/skus`, currency USD). I queried it on 2026-10-18, and every SKU had `effectiveTime` 2026-10-18T07:00:00Z. The public pricing pages (https://cloud.google.com/compute/gpus-pricing, https://cloud.google.com/tpu/pricing) render their tables in JavaScript, so a plain fetch returns no numbers. The catalog API is the machine-readable source behind them. Implied exchange rate: ₹28,000 / $330 ≈ ₹84.8/$ (assumption).

**Labelled global assumptions used in the arithmetic**
- A1. Training FLOPs per token ≈ 6N, where N is the parameter count (the standard Kaplan/Chinchilla approximation). This ignores attention FLOPs. For a 360M model at sequence length 2048 (L=32, d=960), attention adds about 12·L·d·T ≈ 0.75 GFLOP/token, which is **about 35% on top of 6N**. Real token counts for 135M/360M models are therefore up to about 25–35% lower than the 6N figures below.
- A2. Model FLOPs utilisation (MFU) is 40%. llm.c reports up to ~60% on A100 SXM, 49% on A100 PCIe and ~47% for 1.6B on H100 (cited below), so 40% is deliberately conservative for a well-tuned single-node bf16/FlashAttention stack. Small models (135M) usually run at lower MFU than 1B because kernels are small.
- A3. Peak dense bf16 throughput per accelerator: A100 312 TFLOPS, H100 SXM 989 TFLOPS, L4 121 TFLOPS, TPU v5e 197 TFLOPS, TPU v6e 918 TFLOPS. The v6e figure is cited. The others are NVIDIA/Google datasheet values from my knowledge that I did **not** re-fetch this session.
- A4. VM shapes: a2-highgpu-1g = 1×A100 40GB, 12 vCPU, 85 GB RAM. a2-ultragpu-1g = 1×A100 80GB, 12 vCPU, 170 GB. a3-highgpu-1g = 1×H100, 26 vCPU, 234 GB. g2-standard-8 = 1×L4, 8 vCPU, 32 GB. These are from my knowledge of GCP machine types; the GCP GPU docs page was truncated when fetched and did not confirm them. Disk, network and egress costs are excluded.

---

## 1. Compute-optimal (Chinchilla, ~20 tokens/param) versus deliberate over-training of small models

### Takeaway
Chinchilla-optimal is the wrong target for a model meant for on-device inference. Every recent strong small model trains at 1,000–10,000+ tokens per parameter, and quality keeps improving across that range, though with diminishing returns per token. A ₹28k budget can reach roughly 100–1,000 tokens/param for a 135M–360M model. That is far past Chinchilla but well short of SmolLM2/Qwen3.

### Cited Findings
- "Beyond Chinchilla-Optimal" (Sardana et al.) changes the Chinchilla law to account for inference demand. It concludes that for heavy inference use (about 1B requests), models should be "smaller and longer than Chinchilla-optimal". The authors trained 47 models at up to **10,000 tokens/param**, and quality kept improving across the whole range. Scaling laws fitted at typical ratios **overestimate** the benefit of extra tokens at extreme ratios, which is the diminishing-returns signal. — [arXiv 2401.00448](https://arxiv.org/abs/2401.00448)
- SmolLM2 token counts: 135M on **2T tokens**, 360M on **4T tokens**, 1.7B on **~11T tokens**. The 1.7B run cost "around 1e23 FLOPs, or $250,000 USD worth of GPU compute" on 256 H100s. The authors call extended training "a worthwhile trade-off" despite leaving Chinchilla-optimal, and they aimed to stay near a 4–5 epoch repetition threshold. — [SmolLM2 paper, arXiv 2502.02737](https://arxiv.org/html/2502.02737)
- Hardware from the model cards: SmolLM2-135M used 64 H100s and SmolLM2-360M used 128 H100s, both in bf16 with nanotron. Wall-clock time is not stated. — [SmolLM2-135M card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [SmolLM2-360M card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- All Qwen3 models, including 0.6B, were trained on a **36T-token** corpus in 119 languages, under Apache 2.0. — [Qwen3 tech report, arXiv 2505.09388](https://arxiv.org/html/2505.09388)
- Gemma 2 trained its 2B and 9B students on "more than 50× the compute-optimal quantity" of tokens (2B on 2T, 9B on 8T), using distillation. — [Gemma 2 report, arXiv 2408.00118](https://arxiv.org/html/2408.00118)
- Evidence of diminishing returns in the GPT-2 124M class: llm.c 124M on 10B tokens scores HellaSwag 29.9, OpenAI GPT-2 124M scores 29.4, and GPT-3 Small 124M on 300B tokens scores 33.7. A 30× token increase gave about +3.8 HellaSwag points. — [llm.c discussion #481](https://github.com/karpathy/llm.c/discussions/481)
- GPT-2 1.6B in llm.c crosses OpenAI GPT-2 1.5B at about 25K steps (≈26B tokens). It reaches HellaSwag 57.7 at 100K steps and 62.7 at 330K steps, and about 61% for a 400B-token run before instability. — [llm.c discussion #677](https://github.com/karpathy/llm.c/discussions/677)

### Inferences
- Chinchilla tokens (20N): 135M → 2.7B tokens; 360M → 7.2B; 1B → 20B. SmolLM2-360M's 4T tokens is about 11,100 tokens/param, roughly 555× Chinchilla. SmolLM2-135M is about 14,800 tokens/param.
- From the llm.c numbers, returns per token are clearly sublinear at fixed model size (≈log-like): 10B→300B tokens gave +3.8 HellaSwag at 124M. Data quality (FineWeb-Edu-style filtering) probably matters as much as token count at this scale. The SmolLM2 paper treats data curation as central (see adjacent notes).

### Gaps
- I did not retrieve a published per-token quality curve (loss or HellaSwag versus tokens) for SmolLM2-135M/360M intermediate checkpoints. Such checkpoints may exist on the Hub but were not checked.
- I did not fetch the "Scaling Laws for Precision" paper (Kumar et al. 2024). Its finding that over-trained models degrade more under post-training quantisation is relevant to on-device int4 deployment but is unverified here.
- I did not fetch the Llama 3.2 1B training-token count (9T, per the task brief).

---

## 2. Google Cloud GPU/TPU prices (on-demand and spot), us-central1 versus asia-south1 (Mumbai)

### Takeaway
At list price, a **single TPU v6e chip on spot (~$0.58/h in us-central1) is by far the cheapest FLOP on GCP**, about 4× cheaper per FLOP than the next option. Among NVIDIA GPUs, L4 spot is cheapest per FLOP, and H100 spot is about equal to A100 spot. Mumbai is generally **not** cheaper. It has **no A100 accelerators in any zone**, and its v6e spot price ($1.36) is 2.3× the US price.

### Cited Findings
GPU accelerator SKUs, USD per GPU-hour, **excluding** the vCPU/RAM that A2/A3/G2 VMs also bill — [GCP Cloud Billing Catalog API, Compute Engine service 6F81-5844-456A, queried 2026-10-18](https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list)

| GPU | us-central1 on-demand | us-central1 spot | asia-south1 on-demand | asia-south1 spot |
|---|---|---|---|---|
| A100 40GB | 2.9339 | 1.7603 | 3.4369 | 0.3096 (*see below*) |
| A100 80GB | 3.9281 | 2.3568 | 4.7180 | 0.4246 (*see below*) |
| H100 80GB | 9.7966 | 5.8485 | 10.1884 | 5.5003 |
| H200 141GB | 9.3117 | 5.586 | 11.1741 | 6.0491 |
| L4 | 0.5600 | 0.3352 | 0.5830 | 0.2969 |

vCPU/RAM SKUs (us-central1, on-demand / spot): A2 core $0.031611 / $0.01897 per vCPU-h and A2 RAM $0.004237 / $0.002542 per GiB-h. A3 core $0.025498 / $0.01524 and A3 RAM $0.00222 / $0.001325. G2 core $0.024988 / $0.01495 and G2 RAM $0.002927 / $0.001751. — same catalog query.

TPU SKUs, USD per chip-hour — [GCP Billing Catalog, queried 2026-10-18](https://cloud.google.com/billing/docs/reference/rest/v1/services.skus/list)

| TPU | us-central1 on-demand | us-central1 spot | us 1-yr CUD | asia-south1 on-demand | asia-south1 spot |
|---|---|---|---|---|---|
| v5e | 1.20 | 0.3593 | 0.84 | 1.248 | 0.1123 |
| v6e (Trillium) | 2.70 | 0.5838 | 1.89 | 3.24 | 1.3556 |
| v5p | 4.20 | 0.9261 | 2.94 | 4.212 | 0.3791 |

- TPU v6e: **918 peak BF16 TFLOPS** and **32 GB HBM** per chip. — [GCP TPU v6e docs](https://docs.cloud.google.com/tpu/docs/v6e)
- On 2026-10-18, `gcloud compute accelerator-types list` for asia-south1 showed only nvidia-l4, nvidia-tesla-t4, nvidia-h100-80gb, nvidia-h100-mega-80gb, nvidia-h200-141gb, nvidia-rtx-pro-6000 and TPU `ct6e` (v6e) in zones b and c. **No A100 accelerator type exists in any asia-south1 zone.** The Mumbai A100 SKU prices are therefore unusable, and the very low Mumbai A100 spot rows look like catalogue artefacts. — direct gcloud query on project aptivra-training (primary observation, not a web source)
- GCP Spot prices "can change up to once every 30 days" and carry "up to 91% off" on-demand. — [GCP VM instance pricing page](https://cloud.google.com/compute/vm-instance-pricing)
- Non-GCP context (comparison blogs, not provider pages; dated Feb–Aug 2026): H100 per GPU-hour about $1.40–2.10 on Vast.ai, about $1.99–3.29 on RunPod and about $2.49–3.99 on Lambda. A100 80GB about $0.70–2.00 on Vast.ai, about $0.89–1.64 on RunPod and about $1.29–2.79 on Lambda. — [jarvislabs 2026 comparison](https://jarvislabs.ai/ai-faqs/best-cloud-gpu-providers-2026), [findstack](https://findstack.com/resources/runpod-vs-vast-ai-vs-lambda-labs), [keenable H100 price summary](https://select.keenable.ai/r/effective-nvidia-h100-cloud-prices/0DRsK7gE-5uMsmQ7IiUM1A). llm.c priced an 8×A100 80GB node at ~$14/h on Lambda in 2024. — [llm.c #481](https://github.com/karpathy/llm.c/discussions/481)

### Inferences (arithmetic, using A4 VM shapes)
- a2-highgpu-1g (A100 40GB), us-central1: on-demand 2.9339 + 12×0.031611 + 85×0.004237 = **$3.67/h**; spot 1.7603 + 12×0.01897 + 85×0.002542 = **$2.20/h**.
- a2-ultragpu-1g (A100 80GB): on-demand **$5.03/h**; spot **$3.02/h**.
- a3-highgpu-1g (H100): on-demand 9.7966 + 26×0.025498 + 234×0.00222 = **$10.98/h**; spot **$6.56/h**.
- g2-standard-8 (L4): on-demand **$0.85/h**; spot **$0.51/h**.
- $330 therefore buys about 90 h of A100-40 on-demand, 150 h of A100-40 spot, 30/50 h of H100 (on-demand/spot), 387/646 h of L4, 275/918 h of TPU v5e, or **122/565 h of TPU v6e**.
- Non-GCP H100 at about $2–3/h is roughly 2–3× cheaper than GCP H100 spot ($6.56/h all-in). The owner's credit is GCP-only, so this is context, not an option.

### Gaps
- I could not render the official HTML pricing tables, so the "Last updated" date on those pages is unknown. The catalog `effectiveTime` (2026-10-18) is the dating evidence used.
- Spot availability and preemption rates for A100/H100/v6e in us-central1 are not published. Spot obtainability is unverified.
- I did not verify whether A3 High 1-GPU shapes are available on spot/flex-start only.

---

## 3. Realistic training throughput (tokens/s, MFU) and tokens per dollar

### Takeaway
Published single-node reproductions show 40–60% MFU in bf16. That is about 180k tok/s per A100 and about 490k tok/s per H100 for a GPT-2-124M-class model. Converted to tokens per dollar at GCP list prices, a **360M model gets ~0.06–0.10B tokens/$ on A100/H100/L4 versus ~1.05B tokens/$ on TPU v6e spot**. The owner's M4 Max measured about 2,450 tok/s full-parameter training on a 362M model, about 5.3 effective TFLOP/s.

### Cited Findings
- llm.c GPT-2 124M ran on 8×A100 80GB SXM for ~90 min and ~$20 on 10B FineWeb tokens (≈18,865 steps × 524,288 tokens). It reached "up to ~60%" MFU on A100 SXM. A single A100 40GB PCIe ran at 177,968 tok/s with 49.0% MFU. 8×H100 ran at ~3.91M tok/s (≈489k tok/s per GPU), about 2.1× faster than 8×A100, finishing in ~40 min. — [llm.c #481](https://github.com/karpathy/llm.c/discussions/481)
- llm.c GPT-2 1.6B ran on 8×H100 for ~24 h and $672, covering ~33.6B tokens (32K steps × 1,048,576). Throughput was ~381,690 tok/s per node (≈47.7k per GPU) at ~47% MFU. llm.c was ~19% faster per iteration than PyTorch. — [llm.c #677](https://github.com/karpathy/llm.c/discussions/677)
- Local measurement on Apple M4 Max 36GB (MPS), full-parameter AdamW training of a 361,821,120-parameter model (SmolLM2-360M shape) at sequence length 167: **2,452 tok/s** (micro-batch 8), 2,190 (mb 4), 1,773 (mb 2) and 1,388 (mb 1). LoRA (8.7M trainable) reached 2,845 tok/s at mb 8. Forward-only (frozen) reached ~10,000 tok/s. — repository evidence `/Users/raghunath/code/personal/Aptivra/docs/adr/0012-evidence/pilot-evidence.json` (primary, local)

### Inferences (arithmetic; A1–A4 apply)
- Sanity check of the model: H100 at 40% MFU on a 124M model gives 989e12×0.4 / (6×124e6) ≈ 532k tok/s, against the measured ~489k/GPU. The model is calibrated to within about 10%.
- Throughput = peak × 0.40 / (6N). Tokens/$ = throughput × 3600 / ($/h).

| Accelerator (us-central1) | $/h OD / spot | 135M tok/s | 360M tok/s | 1B tok/s | 360M tokens/$ (OD / spot) |
|---|---|---|---|---|---|
| A100 40GB (a2-highgpu-1g) | 3.67 / 2.20 | 154k | 57.8k | 20.8k | 0.057B / 0.094B |
| A100 80GB (a2-ultragpu-1g) | 5.03 / 3.02 | 154k | 57.8k | 20.8k | 0.041B / 0.069B |
| H100 (a3-highgpu-1g) | 10.98 / 6.56 | 488k | 183k | 65.9k | 0.060B / 0.10B |
| L4 (g2-standard-8) | 0.85 / 0.51 | 59.8k | 22.4k | 8.1k | 0.094B / 0.16B |
| TPU v5e (1 chip) | 1.20 / 0.36 | 97.3k | 36.5k | 13.1k | 0.11B / 0.37B |
| TPU v6e (1 chip) | 2.70 / 0.58 | 453k | 170k | 61.2k | 0.23B / **1.05B** |

- What $330 buys in tokens (6N, 40% MFU). Subtract about 25–35% at 135M/360M for attention overhead (A1).
  - **135M**: A100-40 spot 83B; H100 spot 89B; L4 spot 139B; v5e spot 322B; **v6e spot 922B**; v6e on-demand 199B.
  - **360M**: A100-40 spot 31B; H100 spot 33B; L4 spot 52B; v5e spot 121B; **v6e spot 346B**; v6e on-demand 75B.
  - **1B**: A100-40 spot 11B; H100 spot 12B; L4 spot 19B; v5e spot 43B; **v6e spot 125B**; v6e on-demand 27B.
- M4 Max: 6 × 361.8e6 × 2,452 ≈ **5.3 TFLOP/s effective**. Running 24/7, that is 2,452 × 86,400 × 30 ≈ **6.4B tokens/month** for a 360M model, or ~19B tokens in 90 days. Scaling by FLOPs suggests about 15–17B tokens/month for 135M, but this is not measured. At seq 167 this throughput is not pretraining-shaped (pretraining uses 2k context, which adds attention cost). The Mac roughly matches ~$20–$45 of L4-spot compute per month (6.4B / 0.16B per $ ≈ $40). It is useful for ablations, small-scale proxies and continued training, but it is not the main pretraining engine.
- These are list-price, perfect-utilisation upper bounds. Spot preemption, checkpointing, data-loader stalls, evaluation and debugging overhead should be budgeted at 20–40% (assumption).

### Gaps
- I found no published MFU for single-chip TPU v6e training of 100M–1B models. Small models on TPU may get much lower MFU than 40% unless batch size is large (via MaxText/JAX or PyTorch/XLA). **The v6e advantage is the most important number here and also the least verified.** It should be measured with a ~1-hour pilot before committing the budget.
- The SmolLM2 model cards and paper give no tokens/s or MFU figures. The SmolLM training logs were not found.
- Pythia's throughput figures were not retrieved.

---

## 4. Which model size and token count fit the budget, and the expected quality

### Takeaway
The realistic target is a **135M–360M model trained on ~50–300B high-quality tokens**. With FineWeb-Edu-class data that should land **clearly above GPT-2 124M and plausibly near SmolLM-v1 135M/360M (HellaSwag ~41–52)**, below SmolLM2-360M (54.5) and far below Qwen3-0.6B on knowledge and maths. A 1B model is under-trained at this budget (≤125B tokens even on v6e spot) and is a poor choice for an on-device model.

### Cited Findings
- SmolLM2-360M base (zero-shot): HellaSwag **54.5**, ARC-avg 53.0, PIQA 71.7, MMLU-cloze 35.8, CommonsenseQA 38.0, TriviaQA 16.9, Winogrande 52.5, OpenBookQA 37.4, GSM8K (5-shot) 3.2. On the same table, Qwen2.5-0.5B scores 51.2 / 45.4 / 69.9 / 33.7 / 31.6 / 4.3 / 54.1 / 37.4 / 33.4 and SmolLM-360M (v1) scores 51.8 / 50.1 / 71.6 / 34.4 / 35.3 / 9.1 / 52.8 / 37.2 / 1.6. — [SmolLM2-360M card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- SmolLM2-135M base: HellaSwag **42.1**, ARC 43.9, PIQA 68.4, MMLU-cloze 31.5, CSQA 33.9, TriviaQA 4.1, Winogrande 51.3, OBQA 34.6, GSM8K 1.4. SmolLM-135M (v1) scores HellaSwag 41.2 and ARC 42.4. — [SmolLM2-135M card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- Qwen3-0.6B-Base: MMLU 52.81, MMLU-Redux 51.26, GSM8K 59.59, MATH 32.44, BBH 41.47, MBPP 36.60 (36T tokens). — [Qwen3 report](https://arxiv.org/html/2505.09388)
- GPT-2 124M HellaSwag 29.4 (OpenAI) / 29.9 (llm.c, 10B tokens); GPT-3 Small 33.7 (300B tokens). — [llm.c #481](https://github.com/karpathy/llm.c/discussions/481)

### Inferences
- **Option A, 135M on TPU v6e spot**: ~600–900B tokens (after the A1 attention discount), about 4,400–6,700 tokens/param. That is within 2–3× of SmolLM2-135M's 2T. SmolLM-v1-135M (41.2 HellaSwag) and SmolLM2-135M (42.1) differ by under 1 point despite different token counts, so on good data a 135M model should plausibly reach HellaSwag ~38–42. This depends entirely on v6e MFU and spot availability, both unverified.
- **Option B, 360M on TPU v6e spot**: ~250–350B tokens, about 700–950 tokens/param. That is below SmolLM-v1-360M's token budget, which I recall as 600B (from the SmolLM v1 blog, **not re-verified this session**). A plausible range is HellaSwag ~45–50 and MMLU-cloze ~32–34. That is better than GPT-2 124M by a wide margin but below SmolLM2-360M.
- **Option C, 360M on NVIDIA (A100/H100 spot)**: only ~25–33B tokens (~80 tokens/param). A plausible range is HellaSwag ~35–40, roughly Pythia-410M-class or below. This is a weak result for $330.
- **No option reaches Qwen3-0.6B** on MMLU/GSM8K. Its 52.8 MMLU and 59.6 GSM8K come from 36T tokens plus heavy maths/code data, about 100–1000× this budget's tokens.
- The honest framing for a "personal on-device learning" goal: $330 buys a **research-grade, owned 135M–360M base** that is clearly worse than freely downloadable SmolLM2/Qwen3 weights of the same size. The value is ownership and control of the data/licence chain, not benchmark quality.

### Gaps
- I did not fetch Pythia-160M/410M HellaSwag numbers, so the "Pythia-class" comparisons above are qualitative.
- Quality predictions for options A–C are extrapolations from the cited anchors, not measurements.

---

## 5. Knowledge distillation during pretraining: gains, teacher overhead, licences

### Takeaway
Logit distillation from a stronger teacher is the best-evidenced per-token quality multiplier at this scale. Gemma 2 reports +7.4 points average for a 2B student at 500B tokens, and lower perplexity at 200M–1B. It costs about 1.5–2.6× student compute for a 0.6B–1.7B teacher, and Apple's distillation scaling laws say it pays only when the teacher already exists, which it does here. Licences decide which teachers are usable: an **Apache-2.0 teacher (Qwen3) carries no obligations on the student**, a **Gemma (pre-Gemma-4) teacher makes the student a "Model Derivative"** bound by Gemma terms, and a **Llama 3.x teacher forces a "Llama…" name prefix plus "Built with Llama"**.

### Cited Findings
- Gemma 2 ablation: a 2B student on 500B tokens (10× compute-optimal) with a 7B teacher scored **60.3 → 67.7** average over 3 benchmarks (from scratch versus distilled). Perplexity at 200M/400M/1B went from 23/19/17 (scratch) to **21/17/15** (distilled), and the gain "remains as the model size is scaled". The loss is cross-entropy between teacher and student per-token distributions over the vocabulary. — [Gemma 2 report](https://arxiv.org/html/2408.00118)
- Distillation Scaling Laws (Busbridge et al., ICML 2025) say distillation beats supervised training "when many students are involved or a teacher already exists", but only up to a compute threshold that grows with student size. "If only one student is to be distilled and a teacher also requires training, supervised learning is generally preferable." The law was fitted for students with cross-entropy > 2.3, so it extrapolates below that. — [PMLR v267](https://proceedings.mlr.press/v267/busbridge25a.html), [Apple ML research](https://machinelearning.apple.com/research/distillation-scaling-laws)
- Qwen3's small models (0.6B, 1.7B, …) use "Strong-to-Weak Distillation" in **post-training** (off-policy then on-policy from larger teachers). The report does not say the base models were pre-trained with distillation. Qwen3 is Apache 2.0. — [Qwen3 report](https://arxiv.org/html/2505.09388)
- Gemma Terms of Use (last modified 2026-04-01) define "Model Derivatives" as including "any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or Output of Gemma … including distillation methods that use intermediate data representations or methods based on the generation of synthetic data Outputs by Gemma for training that model". "Outputs are not deemed Model Derivatives". Google "claims no rights in Outputs". Distributing Model Derivatives requires passing on the use restrictions (Prohibited Use Policy) as enforceable terms, giving recipients a copy of the agreement, marking modified files, and including a NOTICE file. The page also says **Gemma 4 uses a separate Apache-2.0-based licence**. — [Gemma Terms](https://ai.google.dev/gemma/terms)
- Llama 3.2 Community License §1.b.i: "If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include 'Llama' at the beginning of any such AI model name". Redistribution also requires a licence copy and a prominent "Built with Llama" notice. A 700M MAU threshold triggers a separate licence. — [Llama 3.2 LICENSE, meta-llama/llama-models](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/LICENSE)

### Inferences (arithmetic; labelled)
- Teacher overhead per token ≈ teacher forward 2·N_T over student train 6·N_S (assumption A1-style).
  - Teacher Qwen3-0.6B: student 360M → 1 + 1.2/2.16 = **1.56×** total cost; student 135M → **2.48×**.
  - Teacher Qwen3-1.7B: student 360M → **2.57×**; student 135M → **5.2×**.
  - Teacher 4B: student 360M → **4.7×**; student 135M → **10.9×**.
  - So at a fixed $330, online distillation from a 1.7B teacher cuts a 360M student's tokens from ~346B to ~135B (v6e spot). It is worth it only if the per-token gain exceeds the ~2.6× token loss. The Gemma 2 evidence suggests distilled models at N tokens beat scratch at N tokens, but I found no published "distilled at N/2.6 versus scratch at N" comparison at this scale.
- Offline alternative: precompute teacher top-k logits once and reuse them (Gemma 3 is reported to sample 256 logits per token; **not verified this session**). Storage at k=64 with int32 id + fp16 prob (6 B) = 384 B/token, which is **~3.8 TB per 10B tokens** (assumption). That is a real GCS cost and I/O burden. Storing on the M4 Max (36GB RAM, local SSD) is impractical beyond ~1–2B tokens.
- Tokenizer constraint: logit distillation needs the student to share the teacher's vocabulary (or a cross-tokenizer method). A Qwen3 teacher implies Qwen's ~151k vocabulary. For a 135M model, a 151k×576 embedding is ~87M parameters, about 65% of the parameter budget. This is a real architectural cost of choosing a Qwen teacher (assumption; vocabulary size from general knowledge, not fetched).
- Licence ranking for an "own weights" goal: **Apache-2.0 teacher (Qwen3, Gemma 4) > Gemma ≤3 (derivative, terms pass-through) > Llama 3.x (name prefix "Llama…", attribution)**. A Llama-distilled student could not be called "Aptivra-…" without the "Llama" prefix.

### Gaps
- I did not fetch the MiniLLM (reverse-KL), DistilBERT or Gemma 3 report figures, so their measured gains are not cited here.
- I found no source quantifying distillation gains specifically at 100–400M with a 0.6–2B teacher under a fixed dollar budget.
- Whether "Outputs are not Model Derivatives" still lets Gemma-generated synthetic text (as data, not logits) go into a clean-licence student is contradictory on its face. The definition explicitly includes "synthetic data Outputs by Gemma for training that model". This needs legal review.

---

## 6. Pruning plus distillation (Minitron, Sheared-LLaMA) versus from-scratch; are the weights "own"?

### Takeaway
Pruning a larger open model and then distilling is ~30–40× more token-efficient than training from scratch, so it is the highest quality-per-dollar route. The result, however, is legally and conceptually a derivative of the parent's weights, not "own" weights.

### Cited Findings
- Minitron derived 8B and 4B models from Nemotron-4 15B with "up to 40× fewer training tokens per model" than from scratch, retraining on "<3%" of the original data. It reports 1.8× compute savings across the family and up to 16% MMLU improvement over from-scratch models of the same size. — [arXiv 2407.14679](https://arxiv.org/abs/2407.14679)
- Sheared-LLaMA pruned LLaMA2-7B to 1.3B and 2.7B using "only 3% of compute compared to training such models from scratch". The results outperform Pythia, INCITE, OpenLLaMA and TinyLlama at equal size. — [arXiv 2310.06694](https://arxiv.org/abs/2310.06694)
- Under the Gemma Terms, "modifications to Gemma" and "works based on Gemma" are Model Derivatives. — [Gemma Terms](https://ai.google.dev/gemma/terms). Under Llama 3.2, derivative works carry the licence, "Built with Llama" and the "Llama" name prefix. — [Llama 3.2 LICENSE](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/LICENSE)

### Inferences
- Pruning SmolLM2-360M (Apache 2.0, per its card's licence, **not re-checked this session**) or Qwen3-0.6B (Apache 2.0) down to ~135–250M and distilling on ~5–20B tokens would fit easily in $330. It would probably beat any from-scratch run at the same size. The weights would be Apache-2.0 derivatives: legally permissive, needing only licence and NOTICE retention, but not independently originated.
- If "own base model" means **provenance independence** (the stated rationale in the owner's 2026-10-18 decision), pruning disqualifies itself. Logit distillation from an Apache teacher is an intermediate case: the weights are freshly initialised, but their behaviour is transferred. Plain from-scratch training is the only fully clean option. This is a definitional choice the owner must make explicitly.

### Gaps
- I found no legal analysis of whether Apache-2.0 attaches to a student trained on Apache-2.0 teacher logits. Apache-2.0 has no output or derivative-model clause comparable to Gemma's, so the practical risk appears low, but this is an inference, not a cited finding.
- Absolute token counts and benchmark tables for Minitron and Sheared-LLaMA were not retrieved (abstract-level only).

---

## 7. Free-trial constraint: GPUs blocked, quota 0, upgrade process

### Takeaway
A Free-Trial billing account **cannot attach GPUs or request quota increases**. The owner must upgrade to a Paid billing account (remaining credit is kept) and then request GPU/TPU quota, which starts at **0**. The credit must also be **used within 90 days of trial signup**. That is a hard deadline that shapes the plan as much as the dollar amount does.

### Cited Findings
- "While your billing account is a non-billable Free trial account, you can't do any of the following: Add GPUs to your VM instances. Use Google Cloud Marketplace. Request a quota increase…" — [GCP Free Trial docs, last updated 2026-10-13](https://docs.cloud.google.com/free/docs/free-cloud-features)
- On manual upgrade to Paid: "You gain access to all Google Cloud products and services, including … GPUs"; "Your remaining credit is available … but they must be consumed within the original 90 days period from your sign-up date"; billing is then enabled for usage beyond the credit. If the account is not upgraded before 90 days, or the credit is spent, "your Free Trial billing account will be closed and all of its associated projects and resources will be stopped." — [same](https://docs.cloud.google.com/free/docs/free-cloud-features)
- "GPUs and TPUs are not included in the Free Tier offer. You are always charged for GPUs and TPUs that you add to VM instances." — [same](https://docs.cloud.google.com/free/docs/free-cloud-features)
- On project `aptivra-training` (gcloud, 2026-10-18): global quota **GPUS_ALL_REGIONS = 0**; asia-south1 NVIDIA_T4_GPUS = 0 and PREEMPTIBLE_NVIDIA_T4_GPUS = 0. Billing is linked (`billingEnabled: True`), but this command does not reveal whether the account is still Free-Trial or Paid. — direct gcloud observation (primary, local)

### Inferences
- Required sequence: (1) upgrade the billing account to Paid in Console → Billing. (2) In IAM & Admin → Quotas, request `GPUS_ALL_REGIONS` ≥ 1, plus the per-region GPU or preemptible-GPU quota (for example `PREEMPTIBLE_NVIDIA_L4_GPUS` / A100 / H100), or a TPU v6e spot quota in the target zone. (3) Wait for approval. New accounts are often granted small quotas or asked for justification (this is general experience, not cited).
- **Risk control:** after upgrading, any spend beyond the credit bills the card. A billing budget plus alert, and ideally a programmatic cap via a budget → Pub/Sub → disable-billing function, should be in place before the first GPU VM. This is a standard GCP pattern, not fetched here.
- The 90-day clock means a ~565-hour v6e-spot plan needs ~24 days of continuous running. That is feasible, but only if quota approval and pipeline debugging happen early.
- **Budget conflict:** the owner's stored project memory records a "₹1000 total on project aptivra-training" constraint with per-run approval. That conflicts with the ₹28,000 premise of this brief, and it should be reconciled before any spend.

### Gaps
- I did not fetch the official quota-request documentation (Cloud Quotas "request a quota adjustment"). Typical approval times and new-account GPU limits are unverified.
- I did not verify whether TPU (ct6e) spot quota is grantable to newly upgraded accounts, or whether TPU usage is also blocked in Free Trial. The trial text names GPUs explicitly, not TPUs.
