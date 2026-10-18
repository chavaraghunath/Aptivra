# On-Device Runtimes and Hardware Limits for 100M-1B Language Models (Inference, Retrieval, On-Device LoRA)

Research date: 2026-10-18. Scope: Android phones, iPhones, Macs, and x86 laptops; models of 0.1B-1B (with nearby 1.5B-2B data points where no smaller measurement exists). Every number names its device, model, quantization and runtime where the source gives them. If a source omits one of these, the bullet says so.

Method note: about 20 searches and fetches. Primary sources: vendor model cards and docs (Google LiteRT-LM, Qualcomm AI Hub, Meta, Apple ML Research), Google Play Console help, and arXiv papers. Several numbers came through search-engine summaries of pages that were not fetched in full, and these are flagged. Search-engine summaries can garble table rows, so treat flagged numbers as "verify before use".

## Runtimes: architectures, quantization formats, platforms, and training/LoRA support

### Takeaway
Every mainstream runtime is inference-first. LiteRT-LM (Google), ExecuTorch (Meta/PyTorch), llama.cpp/GGUF, MLC-LLM, MLX and Qualcomm Genie/QNN all ship working 4-bit inference for 0.5-1B models on phones. On-device LoRA training, though, is product-grade only on Mac (MLX). On phones it exists only as research frameworks (MobileFineTuner, PockEngine), research extensions of ExecuTorch's training runtime, or forks (a llama.cpp LoRA fork). Apple's on-device adapters are trained off-device and only loaded on the phone.

### Cited Findings

**Google LiteRT-LM / AI Edge (successor path to the MediaPipe LLM Inference API)**
- LiteRT-LM backend matrix: CPU on Android, iOS, macOS, Windows, Linux and IoT; GPU on Android, iOS, macOS, Windows and Linux; NPU on Android, plus a "🚀" marker for Windows that the page doesn't define. A JavaScript/Web API is marked Early Preview and uses WebGPU. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- Model families named: Gemma (Gemma 4 E2B/E4B, Gemma 3n, Gemma 3 1B, FunctionGemma), EmbeddingGemma 2, Llama, Phi-4 and Qwen (Qwen2.5-0.5B/1.5B, Qwen3-0.6B). The overview page makes no LoRA statement. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- Quantization formats on the Gemma 3 1B LiteRT community card: `dynamic_int4` (including a QAT int4 variant, block size 128) and `dynamic_int8`. CPU inference runs through the LiteRT XNNPACK delegate with 4 threads. — [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) (via search summary; not fetched in full)
- EmbeddingGemma 2 ships in sizes useful for retrieval and memory indexing: 270M text-only (165 MB), 440M text+vision (388 MB) and 740M multimodal (485 MB). — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)

**PyTorch ExecuTorch**
- Meta's quantized Llama 3.2 1B/3B models run on ExecuTorch with the Arm CPU backend and KleidiAI kernels. The scheme is 4-bit groupwise weights (group size 32) with 8-bit per-token dynamic activations in transformer linears, 8-bit per-channel classifier and embedding, and up to 8K context. Meta said NPU support was "in progress", not yet benchmarked. — [Meta AI blog: quantized Llama 3.2](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)
- ExecuTorch supports on-device training by lowering both forward and backward graphs. Its paper validated this on a classification model on Android, not an LLM and not on iOS. Separating weights from the program lets LoRA adapters share foundation weights at deploy time. — [ExecuTorch paper, arXiv 2605.08195](https://arxiv.org/pdf/2605.08195) (via search summary)
- LoRA training on ExecuTorch exists only as research. ZeroLock extended the ExecuTorch training runtime with an Android interface and an on-device AdamW that updates LoRA parameters. — [ZeroLock, arXiv 2608.07974](https://arxiv.org/pdf/2608.07974) (via search summary)
- Common production path: fine-tune with QAT on PC or Colab (Unsloth plus torchao), export a `.pte`, and deploy it for phone inference. — [Unsloth docs: deploy LLMs on phone](https://docs.unsloth.ai/new/deploy-llms-phone)

**llama.cpp / GGUF**
- llama.cpp runs on iPhone through CPU and Metal GPU at F16, Q8 and Q4 (Q4 ≈ 4.5 bits/weight), and was measured with Qwen2-0.5B, Llama-3.2-1B and others. — [Challenging GPU Dominance, arXiv 2505.06461](https://arxiv.org/html/2505.06461v1)
- Android GPU inference runs through the Vulkan backend. For example, a Flutter plugin advertises Vulkan acceleration built on a March 2026 llama.cpp release. — [llama_flutter_android on pub.dev](https://pub.dev/documentation/llama_flutter_android/0.2.6/)
- Upstream llama.cpp training is limited. A December 2025 post says it "relies exclusively on full fine-tuning" over raw text tokens. LoRA fine-tuning comes from a fork, QVAC-fabric-llm.cpp, which targets mobile and non-CUDA GPUs; this is a vendor claim and is not merged upstream as far as this search found. Snapshot b8008 contains `ggml-opt.h` and `examples/training/finetune.cpp`. — [HF blog: QVAC fabric-llm finetune](https://huggingface.co/blog/qvac/fabric-llm-finetune); [b8008 source mirror](https://git.mitjafelicijan.com/r/llmnpc/files)

**MLC-LLM**
- In a 2024 measurement study, MLC-LLM (GPU) beat llama.cpp's GPU generation throughput by about 4% on average. That study used a 7B 4-bit model. — [alphaXiv summary of arXiv 2403.12844](https://www.alphaxiv.org/overview/2403.12844) (secondary summary); [arXiv 2410.03613](https://arxiv.org/pdf/2410.03613)
- MLC packages use q4f16_1 quantization, as in `Llama-3.2-1B-Instruct-q4f16_1-MLC`. — [NVIDIA forum, MLC speculative decoding](https://forums.developer.nvidia.com/t/boosting-llm-inference-speed-using-speculative-decoding-in-mlc-llm-on-nvidia-jetson-agx-orin/314470)

**Apple MLX / Core ML / Foundation Models framework**
- `mlx-lm` supports LoRA and QLoRA fine-tuning on Apple silicon, for example `mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit --train ...` with text or chat JSONL data. — [MLX LoRA guide (insiderllm)](https://insiderllm.com/guides/fine-tuning-mac-lora-mlx/); [mlx-examples lora](https://github.com/arpitingle/mlx-examples/blob/main/lora)
- Apple Foundation Models framework (2025): the roughly 3B on-device model accepts rank-32 LoRA adapters trained with Apple's Python toolkit, off-device. "Adapters must be retrained with each new version of the base model." — [Apple ML Research: Foundation Models 2025 updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)

**Qualcomm AI Hub / QNN / Genie**
- Qualcomm publishes Llama-v3.2-1B-Instruct precompiled for its Genie runtime (QNN-based, NPU) in w4 and w4a16 variants with a 4096 context. — [qualcomm/Llama-v3.2-1B-Instruct](https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct) (via search summary)
- Qualcomm's 8B model card says Genie support "will be deprecated soon". The 1B card's status is unconfirmed. — [qualcomm/Llama-v3-8B-Instruct](https://huggingface.co/qualcomm/Llama-v3-8B-Instruct) (via search summary)

**Research training frameworks**
- MobileFineTuner (arXiv 2512.08211, Dec 2025) is open-source C++ with no Python runtime. It supports full fine-tuning and LoRA on commodity phones, with memory-efficient attention, activation checkpointing, gradient accumulation, parameter sharding and an energy-aware scheduler. It was evaluated on GPT-2 124M/355M, Gemma3-270M/1B and Qwen2.5-0.5B. — [MobileFineTuner, arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)
- PockEngine (MIT, MICRO 2023) does sparse backpropagation with compile-time graph pruning. It reports 5.6x memory saving for backprop on Jetson Orin and fine-tunes LLaMA2-7B on Jetson AGX Orin at 550 tokens/s, 7.9x faster than PyTorch. — [PockEngine project page](https://hanlab.mit.edu/projects/pockengine); [arXiv 2310.17752](https://arxiv.org/abs/2310.17752v1)
- PocketLLM uses derivative-free (zeroth-order) optimization and fine-tuned OPT-1.3B on an OPPO Reno 6 with about 6.5 GB of memory. — [PocketLLM, arXiv 2407.01031](https://arxiv.org/html/2407.01031v1)

### Inferences
- For a cross-platform deployment format, the realistic candidates are GGUF (llama.cpp: CPU everywhere, Metal, Vulkan), LiteRT-LM `.litertlm` (CPU/GPU everywhere, NPU on Android) and ExecuTorch `.pte` (CPU with KleidiAI/XNNPACK, vendor delegates). MLX is Mac-only but is the only mature on-device LoRA trainer found.
- On phones, on-device LoRA training needs a custom stack, either MobileFineTuner-style C++ or an ExecuTorch training extension. No vendor runtime offers a supported phone LoRA-training API as of this research.
- Apple's own design pattern is a frozen, heavily quantized base plus small LoRA adapters trained off-device that must be retrained when the base changes. This is direct prior art for a "personal adapter" design, and it implies that personal adapters are invalidated whenever the base model is updated.

### Gaps
- **ONNX Runtime GenAI**: the search budget ran out before it was covered. Support for architectures, quantization (int4 RTN/AWQ via the model builder), the QNN execution provider, and training (ORT has an on-device training API) is unverified here.
- **MediaTek NeuroPilot**: no source fetched. Whether the Dimensity 9300 NPU is open to third-party LLM apps, and at what speed, is unverified.
- **Core ML**: stateful KV-cache and int4 palettization specifics, plus iPhone LLM numbers, were not fetched.
- **MNN**: on-device training support and numbers were not fetched. MNN appears only as an inference framework in the NPU study.
- No primary documentation was fetched for the MediaPipe LLM Inference API's LoRA support, which historically allowed LoRA on GPU for Gemma and Phi-2.

## Measured decode/prefill speed and memory for 0.1B-1B models on representative devices

### Takeaway
On 2024-2026 flagships, a 4-bit 0.5-1B model decodes at roughly 30-65 tok/s, with GPU or NPU prefill in the 1,000-2,600 tok/s range. Decode is bound by memory bandwidth, so CPU and GPU decode are similar and sometimes the CPU wins for ≤1B models. Macs are about 4-8x faster, with about 230-260 tok/s decode for a 1B Q4 model on an M4 Pro. No primary measurements were found for Snapdragon 7-series or x86 laptop CPUs.

### Cited Findings

**Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3), LiteRT-LM**
- Gemma3-1B (1,005 MB file): CPU 177 prefill / 33 decode tok/s; GPU 1,191 prefill / 24 decode tok/s. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- Gemma3-1B by quantization on the S24 Ultra: GPU `dynamic_int4` QAT, 2048 context, ≈2,585 prefill / 56 decode tok/s with 4.50 s TTFT; GPU `dynamic_int8`, 1280 context, ≈1,191 prefill / 24 decode; CPU `dynamic_int4` (4 threads, XNNPACK), ≈138 prefill / 50 decode. — [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT). This came via a search summary that warned the table layout was partly garbled, so verify before citing.
- Qwen2.5-0.5B (521 MB): CPU 251 prefill / 30 decode tok/s. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- Gemma-3n-E2B (2,965 MB): CPU 111/16; GPU 816/16. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)

**Samsung Galaxy S25 Ultra (Snapdragon 8 Elite), LiteRT-LM**
- FunctionGemma (270M class, 289 MB): CPU 2,238 prefill / 154 decode tok/s. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- Qwen2.5-1.5B (1,598 MB): CPU 298/34; GPU 1,668/31. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)

**Other recent devices, LiteRT-LM**
- Vivo X300 Pro (Dimensity 9500 per vendor; the SoC is not named on the page), Qwen3-0.6B (586 MB): CPU 165 prefill / 9 decode tok/s; GPU 580 prefill / 21 decode tok/s. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- Gemma-4-E2B (2.58 GB, above scope but gives the 2026 device ceiling):
  - S26 Ultra: CPU 557/47 tok/s, peak 1,733 MB; GPU 3,808/52, 676 MB.
  - iPhone 17 Pro: CPU 532/25, 607 MB; GPU 2,878/56, 1,450 MB.
  - MacBook Pro M4 Max: CPU 901/42; GPU 7,835/160.
  - Intel Lunar Lake laptop: CPU 435/30, 3,505 MB; GPU 3,751/48.
  - Raspberry Pi 5: CPU 133/8.
  - — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)

**Snapdragon 8 Elite / 8 Elite Gen 5 NPU, Qualcomm Genie**
- Llama-v3.2-1B-Instruct on Snapdragon 8 Elite QRD: w4 ≈33.7 tok/s decode, TTFT ≈0.08-2.6 s; w4a16 ≈53.6 tok/s, TTFT ≈0.04-1.19 s. On 8 Elite Gen 5 QRD: w4 ≈36.9 tok/s, w4a16 ≈64.0 tok/s. The TTFT range runs from a 128-token prompt to a full 4,096-token prompt. — [qualcomm/Llama-v3.2-1B-Instruct](https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct) (via search summary)

**OnePlus 12 (Snapdragon 8 Gen 3), ExecuTorch + KleidiAI**
- Llama 3.2 1B/3B, 4-bit (QAT+LoRA or SpinQuant) vs BF16: decode 2.5x faster and prefill 4.2x faster on average; model size −56% and memory −41% on average. Relative results were reproduced on the Samsung S24+ (1B, 3B) and S22 (1B). The blog gives absolute tok/s only in an image table. — [Meta AI blog](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)

**iPhone 15 Pro (A17 Pro, 8 GB), llama.cpp**
- Llama-3.2-1B at F16, 2 CPU threads: 17 tok/s, against 12.8 tok/s with Metal GPU. At Q4 with multithreaded CPU, CPU beat GPU by 1.33x on Llama-3.2-1B and 1.31x on Qwen2-0.5B. Going from F16 to Q4 gave 1.5-2.5x speedups. — [arXiv 2505.06461](https://arxiv.org/html/2505.06461v1)
- Matrix multiplication was 87.6% of prefill time and 76.2% of decode time for Llama-3.2-1B F16. The 7B/8B models at F16/Q8 hit mmap allocation failures on the 8 GB phone. — [arXiv 2505.06461](https://arxiv.org/html/2505.06461v1)

**Apple M-series Macs**
- Llama 3.2 1B Q4 decode (tg128) on an M4 Pro: llama.cpp ≈230 tok/s, MLX ≈258 tok/s. — [BaseRT, arXiv 2607.00501](https://arxiv.org/pdf/2607.00501) (via search summary)
- Gemma3n-E2B on a MacBook Pro M3 (CPU, LiteRT-LM): 233 prefill / 28 decode tok/s. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)

**NPU vs CPU on Snapdragon (SM8650 to SM8850), Qwen2.5-1.5B**
- Prefill: GENIE/QNN on NPU 1,463.7 tok/s vs llama.cpp 115.1 tok/s. Decode was about 23 tok/s on both NPU paths. CPU decode reached about 70 tok/s. — [Is Your NPU Ready for LLMs?, arXiv 2607.05475](https://www.alphaxiv.org/abs/2607.05475)

### Inferences
- For a 0.5-1B model at 4-bit on any 2024+ flagship, plan on about 25-60 tok/s decode. That is well above reading speed, so interactive chat is not the bottleneck. Prefill, which sets the cost of RAG context, is where the GPU or NPU pays off: about 1,000-2,600 tok/s versus about 140-250 tok/s on CPU. A 2K-token retrieved context therefore costs about 1-2 s on GPU/NPU but about 10-15 s on CPU.
- Decode is memory-bandwidth-bound, so going from 8-bit to 4-bit roughly doubles decode speed. CPU decode can match or beat the GPU for ≤1B models; the S24 Ultra Gemma3-1B CPU decode of 33 tok/s exceeds its GPU decode of 24.
- Model size shrinks well below 1B, while the fixed per-token overheads stay. FunctionGemma 270M reaches 154 tok/s decode on CPU on the S25 Ultra, so a 100-300M model is "free" for background tasks such as classification, routing and memory tagging.

### Gaps
- No primary measurement was found for **Snapdragon 7-series** (for example 7 Gen 3 or 7s Gen 3) running a 0.5-1B model. Inferring from bandwidth (LPDDR5 vs LPDDR5X), expect roughly half to two-thirds of flagship decode. This is unverified.
- No primary measurement was found for the **iPhone 16 Pro** specifically. The closest data points are the iPhone 15 Pro (llama.cpp) and the iPhone 17 Pro (LiteRT-LM, Gemma 4 E2B).
- **Dimensity 9300**: no 0.5-1B measurement was found.
- **x86 laptop CPU**: only Lunar Lake with Gemma-4-E2B (30 tok/s CPU decode). No 0.5-1B data on a generic x86 laptop was found.
- MLC-LLM phone tok/s for 1B models was not found. The llama.cpp Apple-silicon thread (GitHub discussion #4167) was not reachable through search. A secondary site cites M4 Max 40-core Q4_0 7B at 83 tok/s ([markaicode](https://markaicode.com/benchmarks/llamacpp-qwen-3-m4-max-throughput-benchmark/)), but that source admits to removing figures it could not verify.
- FP16 phone numbers for 0.5-1B are thin: only the iPhone 15 Pro llama.cpp F16 figure (17 tok/s).

## Quantization quality loss for small models (QAT vs PTQ; SpinQuant, QuaRot, AWQ, GPTQ)

### Takeaway
Small models lose meaningfully more quality at 4-bit than large ones. In the strongest number found, Qwen3-0.6B dropped about 10% on MMLU under 4-bit GPTQ, against about 1% for Qwen3-14B. QAT, usually with a LoRA or adapter recovery step, is the established remedy: Meta uses it for Llama 3.2 1B, Google for Gemma 3 int4, and Apple for its 2-bit on-device model. Rotation-based PTQ (SpinQuant) is the next-best option when QAT is not possible.

### Cited Findings
- Qwen3 quantization study: under 4-bit GPTQ, Qwen3-14B loses about 1% on MMLU while Qwen3-0.6B loses about 10%. — [An Empirical Study of Qwen3 Quantization (review)](https://liner.com/review/empirical-study-qwen3-quantization) (secondary review of the arXiv paper)
- Smaller models such as Llama-3.2-1B and Qwen1.5-0.5B show "a more pronounced rise in PPL" under quantization. The FP16 WikiText-2 PPL baseline for Qwen1.5-0.5B is 13.52. — [CALM, arXiv 2512.16282](https://arxiv.org/pdf/2512.16282) (via search summary)
- Smaller models "can suffer severe accuracy drops at 4-bit", while 70B models stay stable. AWQ tends to outperform GPTQ for weight-only quantization. — [generalcompute benchmark write-up](https://www.generalcompute.com/blog/quantized-vs-full-precision-gptq-awq-fp8-benchmarks-across-model-sizes?agent=true) (secondary; this is a general claim, not specific to 0.5-1B)
- Math-reasoning study (Qwen2.5-0.5B/1.5B-Instruct): AWQ and GPTQ variants were not consistently worse than full precision across difficulty levels, so the effect depends on the task. — [Quantization Meets Reasoning, arXiv 2505.11574](https://arxiv.org/pdf/2505.11574)
- Meta, Llama 3.2 1B/3B: QAT+LoRA (QLoRA) reaches "competitive accuracy to the BF16 model"; SpinQuant, a PTQ method with learned rotations calibrated on WikiText, is less accurate than QAT+LoRA. Both use 4-bit groupwise weights (g=32) with 8-bit dynamic activations. — [Meta AI blog](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)
- Apple's on-device model (about 3B) uses 2-bit QAT weights with learnable clipping, 4-bit QAT embeddings and an 8-bit KV cache. After LoRA recovery it shows about −4.6% on MGSM and +1.5% on MMLU versus uncompressed. KV-cache sharing at a 5:3 block split cuts KV memory by 37.5%. — [Apple ML Research 2025](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
- NPU-side activation quantization (W4A16 to lower-bit activations) can give about 2x NPU throughput but "often causes accuracy loss", and is more viable for 3B than 1B models. — [arXiv 2607.05475](https://www.alphaxiv.org/abs/2607.05475)
- Google ships a QAT int4 variant of Gemma 3 1B for LiteRT. — [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT)

### Inferences
- For a self-pretrained model of about 360M, plain 4-bit PTQ (Q4_0/RTN) is risky. Budget for either (a) 8-bit weights, which at 360M cost only about 360-400 MB and are still fast, or (b) a QAT phase at the end of pretraining, mirroring Meta, Google and Apple. Evaluate both against an FP16 baseline on the project's own evals, not only on perplexity.
- 8-bit dynamic activations combined with 4-bit group-32 weights is the best-documented CPU-friendly mobile scheme (ExecuTorch/KleidiAI). Lower-bit activations for NPUs are where small models break.

### Gaps
- No primary numbers were found for **QuaRot** on 0.5-1B models, and no direct SpinQuant-vs-QAT accuracy deltas, because Meta's are in images.
- No head-to-head of AWQ vs GPTQ vs GGUF k-quants (Q4_K_M, Q5_K_M, Q8_0) on sub-1B models. The Qwen3 study appendix is the likely source and was not fetched.
- No data on how a LoRA adapter trained on an FP16 base behaves when applied to a 4-bit base on device, which is the base-quantization mismatch problem.

## NPU realities: third-party access, speed and constraints

### Takeaway
The only meaningful third-party LLM NPU path is on Snapdragon, through Qualcomm QNN/Genie (AI Hub precompiled models) and LiteRT-LM's Android NPU backend. Google Pixel adds a TPU path in LiteRT for embeddings. On iPhone, third-party code reaches the Neural Engine only through Core ML, with no direct LLM runtime. NPUs win prefill by about 10x but often do not beat the CPU on decode, and they waste energy unless tuned.

### Cited Findings
- On Qwen2.5-1.5B (Snapdragon SM8650 to SM8850), NPU prefill through GENIE reached 1,463.7 tok/s against 115.1 tok/s for llama.cpp. MNN keeps attention on the CPU and gets about 2x lower prefill than full-graph offload. NPU decode is "often slower than the CPU", and about 70 tok/s CPU decode was reported. — [arXiv 2607.05475](https://www.alphaxiv.org/abs/2607.05475)
- NPU constraints:
  - NPUs need static shapes, so the growing KV context underutilizes them.
  - The NPU can use more energy per token than the CPU during decode because it stalls on memory.
  - RPC polling can consume up to 30% of system energy. Changing the polling interval from 9,999 µs to 5-20 µs cuts energy by more than 30%.
  - Up to 40% of energy is lost to scheduling overhead. Tuning gives up to 54.8% energy reduction.
  - — [arXiv 2607.05475](https://www.alphaxiv.org/abs/2607.05475)
- Qualcomm Genie, Llama 3.2 1B: 53.6 tok/s on 8 Elite and 64.0 tok/s on 8 Elite Gen 5 (w4a16), against 33.7 and 36.9 tok/s for w4. — [qualcomm/Llama-v3.2-1B-Instruct](https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct)
- LiteRT-LM lists an NPU backend for Android only, with none on iOS or macOS. EmbeddingGemma 2 text embedding takes 8.3 ms on the Pixel 11 Pro TPU, against 27.1 ms on CPU and 25.9 ms on GPU on the S26 Ultra. — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- Meta's ExecuTorch Llama 3.2 NPU support was "in progress" at the 2024 launch. — [Meta AI blog](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)

### Inferences
- Treat the NPU as a prefill and embedding accelerator for long RAG contexts and memory-index embedding, not as a decode accelerator for sub-1B models. NPU binaries are SoC-specific (QNN context binaries per chipset), which multiplies the build and test matrix. CPU int4/int8 should remain the universal fallback.
- An embedding model of about 270M on an NPU or GPU at roughly 8-26 ms per text makes continuous on-device memory indexing cheap enough to run opportunistically.

### Gaps
- MediaTek NeuroPilot / Dimensity 9300 NPU access for third-party LLMs: not verified.
- Samsung Exynos NPU access: not covered.
- Apple Neural Engine LLM throughput for third-party Core ML models (0.5-1B): no numbers fetched.

## On-device fine-tuning: feasibility, memory, time per step, energy, thermal, and background execution limits

### Takeaway
On-device LoRA training of 0.1-1B models on phones is feasible but slow. MobileFineTuner measured roughly 1.2-8.2 GB peak RSS (rank 8, sequence 128) and step times of minutes. Full epochs ran tens to 140+ hours, consuming 13-440 kJ. That makes phone training an overnight, charging-only, incremental job, not an interactive one. On a Mac, MLX LoRA is practical today. iOS gives no iPhone background GPU access even in iOS 26, so iPhone background training would be CPU-only and slower.

### Cited Findings

**Phones: MobileFineTuner (2025)**
- Devices: Huawei P50 Pro (Kirin 9000, 8 GB), Huawei nova 9 Pro (Snapdragon 778G, 8 GB), iQOO 15 (Snapdragon 8 Elite Gen 5, 16 GB) and MacBook Air M2 (16 GB). — [arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)
- Peak RSS for LoRA (rank 8, alpha 32, sequence 128, batch 8), MMLU task:

  | Model | Peak RSS |
  |---|---|
  | GPT2-124M | 1,246 MB |
  | GPT2-355M | 3,311 MB |
  | Qwen2.5-0.5B | 2,888 MB |
  | Gemma3-270M | 4,042 MB (large vocabulary) |
  | Gemma3-1B | 8,209 MB |

  — [arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)
- Time and energy per run (PEFT, sequence 128):

  | Model, task | Time | Energy |
  |---|---|---|
  | GPT2-124M, MMLU | 36.3 h | 90 kJ |
  | GPT2-355M, MMLU | 97.9 h | 251 kJ |
  | Qwen2.5-0.5B, MMLU | 25.9 h | 76 kJ |
  | Qwen2.5-0.5B, PIQA | 39.3 h | 114 kJ |
  | Gemma3-270M, MMLU | 49.4 h | 118 kJ |
  | Gemma3-1B, MMLU | 142.8 h | 438 kJ |
  | GPT2-124M, ARC-C | 4.5 h | 13 kJ |

  The Gemma3-1B run uses 438 kJ, about 122 Wh, which is roughly 6-8 full charges of a 4,000-5,000 mAh battery. That ratio is this note's arithmetic, not the paper's. — [arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)
- Qwen2.5-0.5B on PIQA: 107.4 s per step with MobileFineTuner against 489.2 s with Termux+PyTorch. Peak RSS was 2,396 MB against 3,314 MB. — [arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)
- On the 8 GB phones, Gemma3-270M needed all four memory optimizations to avoid OOM, and Qwen2.5-0.5B and GPT2-355M needed attention optimization plus checkpointing. The 16 GB iQOO 15 needed none. — [arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)
- With energy-aware scheduling on the nova 9 Pro, battery fell below 60% at step 53, about 4 hours in. The scheduler then stretched the step interval from 0.081 h to 0.164 h. The paper reports no thermal or temperature data. — [arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)
- The paper's rule of thumb is about 16 GB of RAM per 1B parameters for FP16 full training. It describes typical 2025 phones as having 4-16 GB of RAM and 3,000-5,000 mAh batteries. — [arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)
- Real-world case: Qwen2.5-0.5B with LoRA rank 8 (sequence 256) was fine-tuned on device with 8,000 personal QA pairs per student (28 students, 3 months of smartwatch data). Gains were reported only qualitatively. — [arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)

**Edge GPU: PockEngine**
- Fine-tuning iteration time fell from about 7 s to under 1 s on Jetson Orin. LLaMA2-7B fine-tunes at 550 tok/s on Jetson AGX Orin, 7.9x faster than PyTorch, with 5.6x backprop memory savings. — [MIT News](https://news.mit.edu/2023/technique-enables-ai-edge-devices-keep-learning-over-time); [PockEngine project](https://hanlab.mit.edu/projects/pockengine)

**Mac: MLX**
- The original MLX LoRA example ran at about 250 tok/s on an M1 Max 32 GB, for a 7B-class model with the older `lora.py`. — [mlx-examples lora README](https://github.com/arpitingle/mlx-examples/blob/main/lora) (via search summary)

**Background-execution limits**
- iOS 26 `BGContinuedProcessingTask`: Apple DTS states that background GPU access is "only supported on iPads with an M3 or better (and not supported on any iPhone)". Developers check `BGTaskScheduler.supportedResources.contains(.gpu)`. — [Apple Developer Forums 816774](https://developer.apple.com/forums/thread/816774); [thread 801229](https://developer.apple.com/forums/thread/801229)
- One developer reports a CPU/ML job running 4-5x slower in a `BGContinuedProcessingTask` than in the foreground on iOS 26. — [Apple Developer Forums 807957](https://developer.apple.com/forums/thread/807957)
- On iOS 26.2, backgrounding an app with in-flight Metal work revokes the GPU and crashes the app through a ggml/Metal abort (whisper.cpp). — [whisper.cpp issue 3531](https://github.com/ggml-org/whisper.cpp/issues/3531)

### Inferences
- On a phone, an 8 GB device can LoRA-train a model of about 360M at sequence 128-256 only with checkpointing and memory-efficient attention. A 1B model at 8.2 GB peak needs a 12-16 GB phone. A personal model of about 360M fits the 8 GB tier.
- The MobileFineTuner timings are full-dataset epochs. Personal-memory updates of a few hundred to a few thousand examples per night scale down proportionally. At about 100 s per step (batch 8, 0.5B model), 100 steps is about 3 hours, which suits a charging-plus-idle overnight window, not daytime use.
- The design should be checkpoint-resumable. Background tasks are killed, iOS removes GPU access on backgrounding and can crash Metal jobs, and step-level checkpoints are the only way to make training idempotent under these OS constraints.
- On iPhone, either train in the foreground (for example, while the app is open and charging) or accept CPU-only background training at a 4-5x slowdown. A practical alternative is to train on the user's Mac with MLX and sync adapters.

### Gaps
- Primary Apple documentation for `BGProcessingTask` (`requiresExternalPower`, `requiresNetworkConnectivity`, typical granted runtime of minutes, system discretion) was not fetched. It is widely documented, but no number is cited here.
- Android WorkManager constraints (`setRequiresCharging`, `setRequiresDeviceIdle`, the 10-minute worker execution limit, and long-running work via foreground service plus Android 14+ foreground-service types such as `dataSync`/`specialUse` and their time limits) were not fetched. Needs a primary Android docs citation.
- No measured thermal-throttling curves for phone training were found. MobileFineTuner reports none.
- No MLX LoRA numbers for 0.1-1B models on M-series were found, nor measured seconds per step and peak memory. These need a local measurement on the project Mac.
- llama.cpp on-device LoRA training: upstream status is unclear, and only a fork (QVAC) claims mobile LoRA.

## Storage and RAM budgets on typical phones in 2025-2026, and app-store size limits

### Takeaway
Typical 2025 phones have 4-16 GB of RAM; flagships have 12-16 GB and mid-range phones 6-8 GB. A 0.5-1B 4-bit model needs about 0.5-1 GB of storage and roughly 0.6-1.7 GB of runtime memory. Both stores cap the base install at about 200 MB for frictionless cellular download, so model weights must ship as Play asset packs (up to 1.5 GB each) or as a post-install download.

### Cited Findings
- Typical 2025 phones have 4-16 GB of RAM and 3,000-5,000 mAh batteries. — [MobileFineTuner, arXiv 2512.08211v2](https://arxiv.org/html/2512.08211v2)
- Model file sizes in LiteRT-LM:

  | Model | File size |
  |---|---|
  | Qwen2.5-0.5B | 521 MB |
  | Qwen3-0.6B | 586 MB |
  | Gemma3-1B | 1,005 MB |
  | FunctionGemma | 289 MB |
  | EmbeddingGemma 2 270M | 165 MB |

  Peak CPU memory for the 2.58 GB Gemma-4-E2B ranges from 607 MB (iPhone 17 Pro, CPU) to 1,733 MB (S26 Ultra, CPU). — [LiteRT-LM overview](https://developers.google.com/edge/litert-lm/overview)
- An 8 GB iPhone 15 Pro could not mmap 7B/8B F16/Q8 models. — [arXiv 2505.06461](https://arxiv.org/html/2505.06461v1)
- Google Play size limits:
  - 200 MB for the base module and each feature module.
  - 1.5 GB per asset pack.
  - 4 GB cumulative for on-demand and fast-follow asset packs (30 GB in the Level Up program or for XR).
  - 8 GB total compressed download maximum.
  - Apps above 200 MB show a non-blocking dialog on mobile data.
  - — [Play Console Help 9859372](https://support.google.com/googleplay/android-developer/answer/9859372)
- App Store: 4 GB maximum uncompressed app size and an 80 MB executable limit. A 200 MB cellular-download threshold applies on iOS 12+, with larger apps prompting for Wi-Fi. — [codemia summary](https://codemia.io/knowledge-hub/path/max_size_of_an_ios_application) (secondary; not confirmed against Apple's own documentation in this session)

### Inferences
- A model of about 360M at 4-bit is roughly 200-250 MB, and at 8-bit roughly 400 MB. That puts even the 4-bit build at about the 200 MB cellular threshold, so ship weights as a Play asset pack (1.5 GB each) or as a background download (iOS On-Demand Resources or Background Assets) rather than in the base binary. A 270M embedding model at about 165 MB adds to that.
- Leave headroom: Android and iOS kill memory-heavy background apps. A 1B training peak of 8.2 GB is unrealistic on anything below a 16 GB phone, while inference for a 0.5-1B 4-bit model at about 0.6-1.5 GB fits the 6-8 GB mid-range tier.

### Gaps
- No authoritative 2025-2026 market-share breakdown of phone RAM tiers (for example, percent of active Android devices with 8 GB or more) was found.
- Apple's own documentation of the 200 MB cellular limit and the Background Assets framework size limits was not fetched.
- iOS per-app memory limits (jetsam), including the `increased-memory-limit` entitlement, were not fetched. They are critical for 1B-class models on 6-8 GB iPhones.
