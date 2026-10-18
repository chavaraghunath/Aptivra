# Hugging Face Ecosystem as the Source for a Small, Private, On-Device Personal Assistant Model (2024-2026)

Research date: 2026-10-18. Licence, parameter-count, creation-date and download figures marked "(Hub API)" were read live from `https://huggingface.co/api/datasets/<id>` or `https://huggingface.co/api/models/<id>` on that date. The citation for each is the Hub page of that repo. Licence tags on the Hub are self-declared by the uploader. They are not legal review.

---

## 1. Open end-to-end recipes and tooling from Hugging Face. Which are reusable for a ~360M model, and what compute did published runs need?

### Takeaway
Hugging Face publishes a complete, Apache-2.0, end-to-end stack: datatrove (data), nanotron (pretraining), lighteval (evaluation), TRL (post-training), PEFT, plus written playbooks. The parts that transfer directly to a 360M model are the SmolLM2-360M recipe (4T tokens, 128 H100, nanotron) and the 1B ablation protocol from the Smol Training Playbook (45B tokens, ~1.5 days on one 8xH100 node). SmolLM3's 3B run cost ~437k H100-hours including ablations, which is far beyond a small team. Treat it as a blueprint for decisions (data stages, GQA, NoPE, tied embeddings, APO, merging), not as a run to repeat.

### Cited Findings

**SmolLM3 "blueprint" (3B, July 2025): the most fully documented open recipe**
- SmolLM3 is 3B parameters. Base model and instruct/reasoning model are both on the Hub. Training configs are at `huggingface.co/datasets/HuggingFaceTB/smollm3-configs`, code and eval are at `github.com/huggingface/smollm`, and the tools named are nanotron (training), datatrove (data) and lighteval (evaluation). Training logs are on W&B (`wandb.ai/huggingface/SmolLM3-training-logs`) — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)
- Pretraining ran ~11.2T tokens in 3 stages (web/code/math): Stage 1 0-8T at 85/12/3, Stage 2 8-10T at 75/15/10, Stage 3 10-11.1T at 63/24/13. 12% of web data was multilingual throughout — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)
- Hyperparameters were a global batch of 2.36M tokens, sequence length 4096, LR 2e-4, AdamW (0.9, 0.95), weight decay 0.1, and a WSD schedule with 2000 warmup steps — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)
- The blog gives compute as 384 H100s for 24 days — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3). The playbook gives the main run as 384 GPUs for 30 days (~276,480 GPU-hours), ablations and debugging as ~161,280 GPU-hours, and the total as ~437,760 GPU-hours. Evaluation used just under 10,000 GPU-hours — [Smol Training Playbook](https://huggingfacetb-smol-training-playbook.hf.space/). **The two sources conflict (24 vs 30 days)**; the playbook figure probably includes the restart and recovery period.
- Long context used an extra 100B tokens in two stages: 4k→32k (RoPE theta 1.5M), then 32k→64k (theta 5M). YaRN extends inference to 128k — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)
- Reasoning mid-training used 35B tokens (OpenThoughts3-1.2M plus a subset of Llama-Nemotron-Post-Training v1.1) for 4 epochs (~140B tokens). SFT used 1.8B tokens (1B non-reasoning, 0.8B reasoning) over 4 epochs. Alignment used APO on the Tulu 3 preference mixture, with reasoning pairs where Qwen3-32B is "chosen" and Qwen3-0.6B is "rejected". The final model is a MergeKit linear merge: 0.9 APO model soup plus 0.1 mid-training checkpoint, which recovered long-context RULER scores — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)
- The chat template supports tool calling through `xml_tools` / `python_tools` arguments — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)
- Results: the thinking vs non-thinking modes score AIME25 36.7 vs 9.3, LiveCodeBench 30.0 vs 15.2, and GPQA-Diamond 41.7 vs 35.7. Non-thinking mode beats Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct. Qwen3-4B generally scores higher than SmolLM3 — [HF blog: SmolLM3](https://huggingface.co/blog/smollm3)
- SmolLM3-3B is Apache-2.0, 3.08B parameters, ~751k downloads (Hub API) — [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B). Intermediate checkpoints (`SmolLM3-3B-checkpoints`) and an ONNX export (`SmolLM3-3B-ONNX`) are also published (Hub API) — [HuggingFaceTB models](https://huggingface.co/HuggingFaceTB)
- Other HF datasets support the same recipe: `HuggingFaceTB/smollm3-blueprint` (2025-07) and `HuggingFaceTB/training-guide-nanotron-configs` (2025-10) (Hub API) — [HuggingFaceTB datasets](https://huggingface.co/HuggingFaceTB)

**SmolLM2 (135M / 360M / 1.7B; Nov 2024, paper Feb 2025): the closest size match**
- SmolLM2-360M is Apache-2.0. It was trained on 4T tokens (FineWeb-Edu, DCLM, The Stack plus newly filtered sets) on **128 H100 GPUs** with **nanotron**, in bf16 — [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M). SmolLM2-135M was trained on 2T tokens — [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M). The paper is arXiv 2502.02737 — [github.com/huggingface/smollm](https://github.com/huggingface/smollm)
- Base 360M vs Qwen2.5-0.5B: HellaSwag 54.5 vs 51.2, ARC 53.0 vs 45.4, PIQA 71.7 vs 69.9, MMLU-cloze 35.8 vs 33.7, TriviaQA 16.9 vs 4.3, **GSM8K 3.2 vs 33.4** — [SmolLM2-360M card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- Instruct 360M vs Qwen2.5-0.5B-Instruct: IFEval 41.0 vs 31.6, MT-Bench 3.66 vs 4.16, GSM8K 7.43 vs 26.8. Post-training was SFT (public plus curated data), then DPO on UltraFeedback. Argilla datasets including Synth-APIGen-v0.1 add rewriting, summarisation and function calling — [SmolLM2-360M card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- Intermediate checkpoints for all three sizes were released 2025-02-26 (Hub API) — [HuggingFaceTB](https://huggingface.co/HuggingFaceTB)
- The exact GPU-hours for the 360M run were not found in sources reached (see Gaps).

**Smol Training Playbook (30 Oct 2025): the decision manual**
- The playbook is book-length, covers the full SmolLM3 pipeline, and is a follow-up to the Ultra-Scale Playbook. Its "Training Compass" asks first whether you should train your own model at all — [Smol Training Playbook](https://huggingfacetb-smol-training-playbook.hf.space/); secondary summary: [The Neuron](https://www.theneuron.ai/explainer-articles/hugging-faces-new-playbook-reveals-the-messy-bug-filled-secrets-to-training-world-class-llms)
- Ablation baseline: a 1B Llama-3.2-style model on 45B tokens, about 1.5 days on one 8xH100 node at 42k tok/s/GPU. Global batch 1.5M tokens, seq 4096, 20k steps, data mix FineWeb-Edu / Stack-Edu-Python / FineMath-3plus at 0.7/0.2/0.1 — [Playbook](https://huggingfacetb-smol-training-playbook.hf.space/)
- Architecture lessons from 1B ablations:
  - GQA with 2, 4 or 8 groups ≈ MHA. MQA and 16-KV-head GQA underperformed.
  - Tied embeddings: a 1.2B tied model matched a 1.46B untied model with 18% fewer parameters.
  - NoPE every 4th layer matched RoPE on short context.
  - Document masking is neutral at short context but needed for long-context extension.
  - Over 80% of documents in FineWeb-Edu, DCLM, FineMath and Python-Edu are under 2k tokens.
  — [Playbook](https://huggingfacetb-smol-training-playbook.hf.space/)
- MMLU multiple-choice format only became non-random after ~6T tokens for SmolLM2-1.7B. Small-model ablations therefore need cloze/CF evaluation formats — [Playbook](https://huggingfacetb-smol-training-playbook.hf.space/)
- Framework sizes (core / total lines of code): TorchTitan 7k/9k, Nanotron 15k/66k, DeepSpeed 94k/194k, Megatron-LM 93k/269k — [Playbook](https://huggingfacetb-smol-training-playbook.hf.space/)
- Secondary report (single source, unverified): a full run was restarted after ~1T tokens because GPUs in a parallel group shared the same random seed — [The Neuron](https://www.theneuron.ai/explainer-articles/hugging-faces-new-playbook-reveals-the-messy-bug-filled-secrets-to-training-world-class-llms)

**Ultra-Scale Playbook (Feb/Mar 2025)**
- Over 4,000 scaling experiments on up to 512 GPUs; a later revision says over 4,100 (16k including test runs). It covers 5D parallelism and ZeRO, and pairs with `picotron` (educational) and `nanotron` (production) — [Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/); [InfoQ](https://www.infoq.com/news/2025/03/huggingface-ultra-scale-playbook)
- The guide says there is no universal scaling recipe because it depends on your infrastructure — [InfoQ](https://www.infoq.com/news/2025/03/huggingface-ultra-scale-playbook)

**nanowhale-100m (Apr 2026): the smallest HF from-scratch reference run**
- ~110M parameters (41M embedding), built on the DeepSeek-V4 architecture and trained from scratch: MLA, a 4+1-expert top-2 MoE, Hyper-Connections, 1 MTP layer, and context 2048. Pretraining used 2.6B tokens of FineWeb-Edu. SFT used `smol-smoltalk` (460K conversations, ~72.7M tokens). Compute was **1x H100 80GB**. Code is at `github.com/huggingface/nanowhale`. Licence is Apache-2.0. The card says it is **"not suitable for any production use"**, its outputs are often incoherent, and it reports no standard benchmarks — [HuggingFaceTB/nanowhale-100m](https://huggingface.co/HuggingFaceTB/nanowhale-100m)

**TRL (post-training)**
- The current docs (v1.16.0) list these trainers:
  - Stable: SFT, DPO, KTO, GRPO, RLOO, Reward, Distillation.
  - Experimental: Online DPO, Async GRPO, GMPO, A2PO, CPO, ORPO, TPO, GKD, GOLD, Async Distillation, IW-OPD, SDFT, SDPO.
  - Integrations: DeepSpeed and PEFT.
  — [TRL docs](https://huggingface.co/docs/trl/index)
- Relevant posts: "TRL v1" (31 Mar 2026), co-located vLLM for GRPO (Jun 2025), Liger GRPO (May 2025), OpenEnv (Oct 2025) — [TRL docs](https://huggingface.co/docs/trl/index)

**Open-R1 (Jan–Jun 2025)**
- Open-R1 is a reproduction of DeepSeek-R1. The repo is Apache-2.0 and is now marked **"no longer maintained"**; it points users to TRL for GRPO, SFT and distillation — [github.com/huggingface/open-r1](https://github.com/huggingface/open-r1)
- Scripts: `sft.py`, `grpo.py`, `generate.py` (Distilabel), and `decontaminate.py` (8-gram). Recipes target **1 node of 8xH100 80GB**. Generating with full R1 used 2 nodes; generating with a distilled R1 runs on 1 H100 — [open-r1](https://github.com/huggingface/open-r1)
- OpenR1-Distill-7B vs DeepSeek-R1-Distill-Qwen-7B: AIME24 52.7 vs 51.3, MATH-500 89.0 vs 93.5, GPQA-D 52.8 vs 52.4, LCB 39.4 vs 37.4 — [open-r1](https://github.com/huggingface/open-r1)
- Open-R1 did not train any model below 7B. Its 1.5B numbers are LightEval reproductions of DeepSeek's models, e.g. R1-Distill-Qwen-1.5B AIME24 30.7 — [open-r1](https://github.com/huggingface/open-r1)

**smol-course v2 (2025)**
- Apache-2.0, built around SmolLM3 and SmolVLM2. Units: instruction tuning, evaluation, preference alignment (DPO), VLMs, RL (Oct 2025), synthetic data (Nov 2025). It states "minimal GPU requirements"; v1 notebooks are kept in `v1/` — [github.com/huggingface/smol-course](https://github.com/huggingface/smol-course)

**SmolLM repository**
- Apache-2.0. `text/` holds SmolLM1/2/3, `vision/` holds SmolVLM, and `tools/` holds `smol_tools`, `smollm_local_inference` and `smolvlm_local_inference` — [github.com/huggingface/smollm](https://github.com/huggingface/smollm)

### Inferences
- For a 360M from-scratch model, the most defensible template is the SmolLM2-360M recipe: nanotron config, FineWeb-Edu + DCLM + Stack-Edu + FineMath, SFT on SmolTalk, then DPO. SmolLM3's choices apply as upgrades: GQA, tied embeddings (very significant at 360M, where embeddings are a large share of parameters), NoPE plus document masking, WSD schedule, multi-stage annealing, APO and checkpoint merging.
- Rough compute scaling: 128 H100 for SmolLM2-360M at 4T tokens is far past Chinchilla-optimal (~7B tokens). A budget team can run a Chinchilla-to-10x regime (10-100B tokens) on a single 8xH100 node in days, using the playbook's figure of 42k tok/s/GPU for a 1B model as an upper-bound throughput anchor. This is an extrapolation, not a measured figure.
- nanowhale-100m shows the single-GPU floor and how poor the result is: 2.6B tokens produces incoherent output. Token budget, not architecture novelty, dominates at this scale.
- Open-R1 / GRPO recipes are validated at 1.5B-7B. GRPO on 360M is unproven in HF sources. Distillation (SFT on traces, then TRL Distillation/GKD/GOLD trainers) is the safer reasoning path for very small models.

### Gaps
- No primary source found for SmolLM2-360M total GPU-hours or wall-clock time (the card gives only "128 H100"). The arXiv paper 2502.02737 should hold it but was not fetched.
- The playbook's post-training and infrastructure sections (offset >100k characters) were not read, so its SFT/APO/GRPO/on-policy-distillation lessons are not captured here.
- datatrove, lighteval, nanotron and PEFT docs were not individually fetched. Their roles are cited only through the SmolLM3 blog and the TRL docs.
- No official Transformers.js v4 release notes were reached (see section 4).

---

## 2. Post-training datasets on the Hub for a small assistant: licences and commercial safety

### Takeaway
A fully commercially usable post-training mix can be built from Apache-2.0, MIT, ODC-BY and CC-BY-4.0 sources:
- **Chat:** SmolTalk / SmolTalk2 new subsets, smol-smoltalk, UltraChat-200k, Tulu 3 SFT and preference mixtures, WildChat.
- **Reasoning:** OpenThoughts3, OpenR1-Math, Llama-Nemotron post-training.
- **Tools:** Hermes function calling, Synth-APIGen, xLAM-60k.
- **Personas:** Nemotron-Personas.
- **Companion dialogue:** SODA, Synthetic-Persona-Chat.

**Non-commercial and avoid for an owned commercial model:** PersonaHub (CC-BY-NC-SA-4.0), EmpatheticDialogues (CC-BY-NC-4.0), APIGen-MT-5k (CC-BY-NC-4.0). Magpie-Llama sets carry the Llama 3.1 licence. Several popular sets (OpenHermes-2.5, Mixture-of-Thoughts, magpie-ultra, personachat) declare **no licence**, which is a legal risk rather than permission. Separately from the dataset licence, synthetic data inherits the terms of the model that generated it.

### Cited Findings (licence from Hub API unless stated; ⚠ = non-commercial or restrictive; ❓ = no licence declared)

**HF-native SFT and preference data**
- **SmolTalk2** (Jul 2025, modified Oct 2025): ~8.61M rows, 87.7 GB.
  - Splits: Mid 4.78M rows (~35.2B tokens), SFT 3.38M rows (~19.3B tokens) from 25 datasets, Preference 447k rows.
  - New subsets are **Apache-2.0**: aya-Qwen3-32B, multi-turn-reasoning-if, smolagents-toolcalling-traces, everyday-convs-reasoning, multilingual8, systemchats, table-gpt, and the think preference mix. For the other subsets the card says to check the original dataset's licence.
  - Sources include Llama-Nemotron, OpenThoughts3, SmolTalk, LongAlign-64k, s1k-1.1, Aya, Table-GPT, Tulu 3 Personas IF, Mixture-of-Thoughts, hermes-function-calling-v1, OpenHermes-2.5 and the Tulu 3 8B preference mixture.
  - Samples containing emojis were removed, and the data is decontaminated against benchmarks.
  — [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)
- **SmolTalk** (v1, used for SmolLM2): new subsets (Smol-Magpie-Ultra, Smol-constraints, Smol-rewrite, Smol-summarize) are **Apache-2.0**; the remaining subsets follow their original licences — [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- **smol-smoltalk**: **Apache-2.0**, 100K-1M rows. HF used it for the nanowhale-100m SFT (460K conversations) — [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk); [nanowhale-100m](https://huggingface.co/HuggingFaceTB/nanowhale-100m)
- **everyday-conversations-llama3.1-2k**: **Apache-2.0**, 1K-10K rows. It covers greetings and small talk, which is directly relevant to "buddy" chat — [HuggingFaceTB/everyday-conversations-llama3.1-2k](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k). A think-variant was added 2025-09: `smoltalk2_everyday_convs_think` (Hub API).
- **Tulu 3 SFT mixture**: **ODC-BY**, 100K-1M rows — [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture). **Tulu 3 8B preference mixture**: **ODC-BY** — [allenai/llama-3.1-tulu-3-8b-preference-mixture](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture). **Tulu 3 personas instruction-following**: **ODC-BY** — [allenai/tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following)
- **UltraChat 200k**: **MIT** — [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k). **UltraFeedback binarized**: **MIT** — [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized); the original is also **MIT** — [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)
- **OpenHermes-2.5**: ❓ **no licence in metadata**, 1M-10M rows — [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- **Magpie**:
  - Magpie-Llama-3.1-Pro-300K-Filtered is ⚠ **llama3.1** (Llama Community Licence terms) — [Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered)
  - Magpie-Qwen2.5-Pro-1M is ❓ no licence — [Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1)
  - argilla/magpie-ultra-v1.0 is ❓ no licence — [argilla/magpie-ultra-v1.0](https://huggingface.co/datasets/argilla/magpie-ultra-v1.0)
- **Real user chat logs**:
  - WildChat-1M and WildChat-4.8M are **ODC-BY** — [allenai/WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M); [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
  - LMSYS-Chat-1M has ❓ no licence tag and is gated (auto) — [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)

**Reasoning**
- **OpenThoughts3-1.2M**: **Apache-2.0** — [open-thoughts/OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M). **OpenThoughts-114k**: **Apache-2.0** — [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
- **OpenR1-Math-220k**: **Apache-2.0**, ~126k downloads — [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k). HF also released a verified subset, `HuggingFaceTB/OpenR1-Math-220k-default-verified` (Sep 2025, Hub API).
- **Mixture-of-Thoughts**: ❓ **no licence declared**. ~350k R1-distilled traces (math 93.7k, code 83.1k, science 173k). Science traces come from Llama-Nemotron and keep only those where no Qwen model pre-processed the prompts — [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts)
- **Llama-Nemotron-Post-Training-Dataset**: **CC-BY-4.0** — [nvidia/Llama-Nemotron-Post-Training-Dataset](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset). **Nemotron-Post-Training-Dataset-v2**: **CC-BY-4.0**, gated (auto) — [nvidia/Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)

**Function calling and tool use**
- **hermes-function-calling-v1**: **Apache-2.0**, ~57k downloads, updated Jan 2026 — [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- **xLAM function-calling-60k**: **CC-BY-4.0**, gated (auto-approve) — [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **APIGen-MT-5k** (multi-turn agentic): ⚠ **CC-BY-NC-4.0 (NON-COMMERCIAL)** — [Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)
- **Synth-APIGen-v0.1** (used for SmolLM2 function calling): **Apache-2.0** — [argilla/Synth-APIGen-v0.1](https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1)
- **smolagents tool-calling traces** come inside SmolTalk2 under **Apache-2.0** — [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)

**Personas**
- **PersonaHub**: ⚠ **CC-BY-NC-SA-4.0 (NON-COMMERCIAL, share-alike)** — [proj-persona/PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub)
- **Nemotron-Personas-USA**: **CC-BY-4.0**, 1M-10M rows, updated Dec 2025 — [nvidia/Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA). India and Japan versions are also **CC-BY-4.0**. They are produced with NeMo Data Designer using a probabilistic graphical model — [Nemotron-Personas-India](https://huggingface.co/datasets/nvidia/Nemotron-Personas-India); [Nemotron-Personas-Japan](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan). The plain `nvidia/Nemotron-Personas` ID returned an error from the API on the research date.

**Empathetic and companion dialogue**
- **EmpatheticDialogues**: ⚠ **CC-BY-NC-4.0 (NON-COMMERCIAL)** — [facebook/empathetic_dialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues). A third-party LLM-format repack is tagged Apache-2.0 — [Estwld/empathetic_dialogues_llm](https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm). **This tag is likely invalid**: a derivative cannot relicense NC source data (see Inferences).
- **SODA**: **CC-BY-4.0**, 1M-10M social dialogues — [allenai/soda](https://huggingface.co/datasets/allenai/soda)
- **Synthetic-Persona-Chat**: **CC-BY-4.0** — [google/Synthetic-Persona-Chat](https://huggingface.co/datasets/google/Synthetic-Persona-Chat)
- **PersonaChat (truecased)**: ❓ no licence — [bavard/personachat_truecased](https://huggingface.co/datasets/bavard/personachat_truecased)

**Pretraining corpora (for a from-scratch model)**
- **FineWeb-Edu**: ODC-BY, ~424k downloads — [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- **FineWeb-2** (multilingual): ODC-BY — [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
- **FinePDFs**: ODC-BY, updated Apr 2026 — [HuggingFaceFW/finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)
- **FineMath**: ODC-BY — [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- **SmolLM-Corpus**: ODC-BY — [HuggingFaceTB/smollm-corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
- **Cosmopedia**: Apache-2.0 — [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
- **DCLM-baseline-1.0**: CC-BY-4.0 — [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
- **Stack-Edu**: ❓ no dataset-level licence, only per-file `detected_licenses` / `license_type` fields — [HuggingFaceTB/stack-edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)
- HF lists the full SmolLM3 pretraining set as a collection — [smollm3-pretraining-datasets](https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets-685a7353fdc01aecde51b1d9)

### Inferences
- **Commercial-safe core for "personal smart expert" and "private buddy"**, all permissive: SmolTalk2 Apache subsets + smol-smoltalk + everyday-conversations + Tulu 3 SFT, preference and personas (ODC-BY) + UltraChat-200k / UltraFeedback (MIT) + OpenThoughts3 / OpenR1-Math (Apache) + Llama-Nemotron (CC-BY) + hermes-function-calling / Synth-APIGen (Apache) + xLAM-60k (CC-BY) + Nemotron-Personas (CC-BY) + SODA / Synthetic-Persona-Chat (CC-BY) + WildChat (ODC-BY). CC-BY and ODC-BY require attribution in the model card.
- **Exclude** PersonaHub, EmpatheticDialogues (including the "Apache" repack, which cannot relicense NC source data) and APIGen-MT-5k from any model the owner may commercialise. **Review** the ❓ sets (OpenHermes-2.5, Mixture-of-Thoughts, magpie-ultra, Magpie-Qwen, personachat, LMSYS) before use, because "no licence" defaults to all rights reserved. SmolTalk2's own composite licence inherits these upstream ambiguities for its non-new subsets.
- An "empathetic buddy" capability without EmpatheticDialogues can be built from SODA + Synthetic-Persona-Chat + everyday-conversations + Nemotron-Personas-seeded synthetic dialogues generated by an Apache-2.0 teacher such as Qwen3. This mirrors how SmolTalk2 regenerated responses with Qwen3-32B.
- SmolTalk2's emoji-removal filter is a deliberate style choice by HF. A "buddy" persona may want to keep some emoji, so the filter should be a configurable decision, not an inherited default.

### Gaps
- Licence terms of the *generator models* behind synthetic sets (e.g. GPT-4 for OpenHermes, Llama-3.1-405B for magpie-ultra) were not verified from primary cards in this session.
- The Hub API exposed no licence for xLAM-60k's earlier versions or for `stingning/ultrachat` (the original UltraChat). The API returned an error for the latter.
- Dataset sizes in tokens are available only for SmolTalk2. The others give only Hub size categories.

---

## 3. Newest small models on the Hub (2025-2026): how they rank and what each did differently

### Takeaway
The 2026 sub-1B frontier is dominated by:
- **Qwen3.5-0.8B** (Feb 2026, Apache-2.0, hybrid Gated-DeltaNet + attention, multimodal, 262k context)
- **LiquidAI LFM2.5** (230M/350M/1.2B, convolution + GQA hybrids, 19T tokens, strong on IFEval/BFCL, **but LFM Open Licence with a $10M revenue cap**)
- **Gemma 4 E2B** (Mar 2026, now **Apache-2.0**, unlike Gemma 3's custom terms)
- **IBM Granite 4.0 350M / 1B** (Apache-2.0, Oct 2025)
- **PleIAs Baguettotron / Monad** (321M / 57M, Apache-2.0)

HF's own newest text model is still SmolLM3-3B (Jul 2025). Its only newer from-scratch release is the educational nanowhale-100m. The Open LLM Leaderboard was **retired in March 2025** and has no official HF small-model successor, so ranking now relies on model-card tables and third-party arenas.

### Cited Findings

**Leaderboards**
- The Open LLM Leaderboard was officially retired in March 2025 (announced by Clémentine Fourrier) because its benchmarks were becoming obsolete. Community alternatives pointed to are Chatbot Arena, the OpenVLM Leaderboard and RewardBench — [HF community post](https://huggingface.co/posts/burtenshaw/596060753193040); [Open LLM Leaderboard Space](https://huggingface.co/spaces/open-llm-leaderboard). Leaderboard v2 (2024) had added MMLU-Pro, GPQA and other benchmarks — [DeepLearning.AI The Batch](https://www.deeplearning.ai/the-batch/hugging-face-overhauls-open-llm-leaderboard-with-tougher-benchmarks)
- HF has since published benchmark viewers (`HuggingFaceTB/llm-benchmarks-viewer`, `post-training-benchmarks-viewer`, Oct 2025, Hub API) — [HuggingFaceTB](https://huggingface.co/HuggingFaceTB)
- The Hub model card spec has a new, simpler eval-results metadata format, which replaces model-index — [Hub docs: model cards](https://huggingface.co/docs/hub/model-cards)

**Model table (Hub API: licence, exact parameter count, creation date)**

| Model | Params | Created | Licence | Notes |
|---|---|---|---|---|
| [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M) | 229.7M | 2026-06 | ⚠ LFM Open v1.0 | 14 layers (8 double-gated conv + 6 GQA), 32k ctx, 19T tokens, distilled from LFM2.5-350M + multi-stage RL; IFEval 71.7, IFBench 38.4, BFCLv3 43.3, MMLU-Pro 20.3; 213 tok/s decode on Galaxy S25 Ultra, 42 tok/s on Raspberry Pi 5; not recommended for math/code/creative writing |
| [HuggingFaceTB/nanowhale-100m](https://huggingface.co/HuggingFaceTB/nanowhale-100m) | ~110M | 2026-04 | Apache-2.0 | educational DeepSeek-V4-style MoE; not for production |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 5.12B stored (an "E2B" effective-parameter model) | 2026-03 | **Apache-2.0** (Hub tag), not gated | licence change from Gemma 3 |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 0.87B | 2026-02 | Apache-2.0 | see detail below |
| [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | 1.17B | 2026-01 | ⚠ LFM Open v1.0 | ~115k downloads |
| [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron) / [PleIAs/Monad](https://huggingface.co/PleIAs/Monad) | 321M / 57M | 2025-11 | Apache-2.0 | |
| [ibm-granite/granite-4.0-350m](https://huggingface.co/ibm-granite/granite-4.0-350m), [-h-350m](https://huggingface.co/ibm-granite/granite-4.0-h-350m), [-1b](https://huggingface.co/ibm-granite/granite-4.0-1b) | 352M / 340M / 1.63B | 2025-10 | Apache-2.0 | |
| [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) | 268M | 2025-10 | ⚠ Gemma terms | function-calling specialist |
| [facebook/MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M), [MobileLLM-Pro](https://huggingface.co/facebook/MobileLLM-Pro) | 950M / 1.08B | 2025-09 | ⚠ **FAIR non-commercial research** | gated |
| [Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) | 0.75B | 2025-09 | Apache-2.0 | safety classifier |
| [google/vaultgemma-1b](https://huggingface.co/google/vaultgemma-1b) | 1.04B | 2025-09 | ⚠ Gemma terms | |
| [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | 268M | 2025-08 | ⚠ Gemma terms, gated (manual) | |
| [LiquidAI/LFM2-350M / 700M / 1.2B](https://huggingface.co/LiquidAI/LFM2-350M) | 354M / 742M / 1.17B | 2025-07 | ⚠ LFM Open v1.0 | |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3.08B | 2025-07 | Apache-2.0 | |
| [tiiuae/Falcon-H1-0.5B-Instruct](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct) | 0.52B | 2025-05 | ⚠ Falcon LLM licence (custom) | |
| [allenai/OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) | 1.48B | 2025-04 | Apache-2.0 | fully open data |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 0.75B | 2025-04 | Apache-2.0 | ~30M downloads, still in the Hub's trending top list |
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | 1.0B | 2025-03 | ⚠ Gemma terms | |
| [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | 3.84B | 2025-02 | MIT | |
| [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1.24B | 2024-09 | ⚠ Llama 3.2 Community Licence, gated | |
| [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 362M | 2024-10 | Apache-2.0 | |

**What each did differently (where a primary card was read)**
- **Qwen3.5-0.8B**:
  - Architecture: 24 layers arranged as 6×(3 Gated DeltaNet + 1 Gated Attention), MTP, a 248k vocabulary, a vision encoder (image and video input) and **262k native context**.
  - Thinking mode is off by default; the card warns of thinking loops.
  - Scores: MMLU-Pro 29.7 (non-think) / 42.3 (think), IFEval 52.1 / 44.0, BFCL-V4 25.3 (think), OCRBench 79.1.
  — [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- **LFM2.5-230M** (comparison table from Liquid's card): LFM2.5-230M / LFM2.5-350M / Qwen3.5-0.8B / Gemma 3 1B scores are IFEval 71.7 / 77.0 / 59.9 / 63.5, BFCLv3 43.3 / 44.1 / 35.1 / 16.6, MMLU-Pro 20.3 / 20.0 / 37.4 / 14.0, and τ²-Retail 13.7 / 17.8 / 6.1 / 6.4 — [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M). These are vendor-reported numbers.
- **LFM Open License v1.0** is Apache-2.0-based, but commercial rights end once the licensee's annual revenue reaches $10M. Derivatives and fine-tunes remain subject to the licence, and there is no copyleft — [Liquid AI LFM licence](https://liquid.ai/lfm-license); [Liquid docs](https://docs.liquid.ai/lfm/help/model-license). The licence text says "$10,000,000 or more" while the FAQ says "exceeds" — [Liquid docs](https://docs.liquid.ai/lfm/help/model-license)
- **SmolLM2 vs Qwen2.5-0.5B at 360M**: SmolLM2 wins on knowledge and commonsense tasks but loses badly on GSM8K (3.2 vs 33.4) — [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- **Secondary listicles** (low reliability) call Qwen3 the "safest all-round default" and SmolLM3 "the most genuinely open model in this class" — [PremAI](https://www.premai.io/blog/best-lightweight-language-models-worth-running/); [Turing Post](https://www.turingpost.com/p/slmslist)

### Inferences
- Architecture trend for 2025-2026 sub-1B models: hybrid linear-recurrent/convolution + sparse-attention stacks (Qwen3.5's DeltaNet, LFM2's conv blocks, Granite-4.0-H's Mamba-hybrid "h" variant, Falcon-H1), MTP heads, very large token budgets (19T for a 230M model) and distillation from a larger sibling. The pure-transformer SmolLM2/3 line is now the conservative baseline.
- For an owned commercial model, permissive comparators and teachers are Qwen3/Qwen3.5 (Apache), Granite 4.0 (Apache), Gemma 4 (Apache per Hub tag), SmolLM2/3 (Apache), OLMo-2 (Apache) and PleIAs (Apache). Gemma 3 outputs used as distillation data carry Gemma Terms of Use obligations. LFM2/2.5, MobileLLM and Llama 3.2 should not be teachers or bases if commercial freedom matters.
- LFM2.5-230M's IFEval 71.7 and BFCLv3 43.3 at 230M set the bar a 360M personal-assistant model must clear on instruction following and tool calling. SmolLM2-360M-Instruct's IFEval 41.0 shows how far the 2024 recipe now lags.

### Gaps
- Gemma 4 E2B and Granite 4.0 nano model cards were not fetched, so their benchmarks and design specifics are absent. The Gemma 4 Apache-2.0 licence comes only from the Hub API tag and secondary listings, not from Google's licence text.
- No independent, third-party benchmark comparison across all these models (same harness) was found. Every number is vendor-reported.
- Hub "trending" API results returned few relevant sub-2.5B text models on the research date, so "trending" status beyond Qwen3-0.6B is not established.

---

## 4. Hugging Face's own on-device work: SmolVLM, SmolLM demos, transformers.js, HuggingSnap

### Takeaway
HF's on-device evidence is mostly qualitative. HuggingSnap (iOS 18, fully offline, SmolVLM2-500M via MLX-Swift) is the flagship. SmolVLM-256M/500M are Apache-2.0 and built on SmolLM2-135M/360M, which demonstrates the 360M-class backbone in a shipping app. Transformers.js v4 (Mar 2026, WebGPU rewrite with ONNX Runtime) and WebGPU Spaces show SmolLM2 in-browser. HF itself publishes few tokens/s or RAM figures; LiquidAI publishes more.

### Cited Findings
- SmolVLM2 comes in 2.2B, 500M and 256M sizes; the latter two are called "the smallest video language models ever released". The 500M model's video ability is "very close to SmolVLM 2.2B". No RAM or speed figures are given — [HF blog: SmolVLM2](https://huggingface.co/blog/smolvlm2)
- The iPhone app runs SmolVLM2-500M fully locally. It was built with MLX-Swift (`mlx-swift-examples`, `llm-tool`). The model also runs on Mac with `mlx-vlm`, and a VLC media player integration is in progress — [HF blog: SmolVLM2](https://huggingface.co/blog/smolvlm2)
- The HuggingSnap App Store listing says it interprets the camera view in real time with no data sent to the cloud — [App Store](https://apps.apple.com/us/app/huggingsnap/id6742157364)
- It is free, needs iOS 18, and works offline; Apple's Visual Intelligence relies on ChatGPT by contrast — [Digital Trends](https://qa2.www.digitaltrends.com/mobile/huggingsnap-app-iphone-rival-visual-intelligence-works-offline)
- VoiceOver accessibility improved from "mostly unusable" in v1.0 to "fully accessible" in v1.2 — [AppleVis](https://www.applevis.com/apps/ios/productivity/huggingsnap-device-visual-assistant)
- SmolVLM-256M-Instruct is Apache-2.0. Its base_model metadata lists SmolLM2-135M-Instruct + SigLIP-base-patch16-512, and it has ~264k downloads. SmolVLM2-256M/500M-Video-Instruct are Apache-2.0; the 500M has ~1.4M downloads (Hub API) — [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct); [SmolVLM2-256M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)
- The SmolLM repo ships `tools/smollm_local_inference` and `tools/smolvlm_local_inference` guides — [github.com/huggingface/smollm](https://github.com/huggingface/smollm). SmolLM3 has an official ONNX export (Hub API) — [HuggingFaceTB/SmolLM3-3B-ONNX](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-ONNX)
- An HF Space runs SmolLM2-1.7B-Instruct in-browser on WebGPU via Transformers.js — [HF Space](https://huggingface.co/spaces/HuggingFaceTB/SmolLM2-1.7B-Instruct-WebGPU/blob/c9e5c6fcea399fb0ff125b2ab031982a84214316/README.md)
- Transformers.js v4 (listed as Mar 2026) rewrote the WebGPU backend in C++ with the ONNX Runtime team, and the same code runs in browsers and server runtimes — [releases.sh](https://releases.sh/hugging-face/transformers-js/highlights). Performance claims such as "up to 4x" are secondary and unverified — [howaiworks.ai](https://howaiworks.ai/blog/transformers-js-v4-release)
- A non-HF datapoint gives the best on-device numbers found: LFM2.5-230M decodes at 213 tok/s on a Galaxy S25 Ultra and 42 tok/s on a Raspberry Pi 5 — [LiquidAI/LFM2.5-230M](https://huggingface.co/LiquidAI/LFM2.5-230M)

### Inferences
- HF's on-device stack for a 360M text model has three reusable paths: MLX / MLX-Swift (Apple), Transformers.js / ONNX (web), and llama.cpp GGUF (general; not verified in this session). HuggingSnap shows that a SmolLM2-360M-class backbone is shippable in a privacy-first offline iOS app.
- HF does not publish rigorous latency or memory benchmarks. The team must measure its own on target devices, as Aptivra's equivalence-gate discipline would require anyway.

### Gaps
- No HF primary source gave tokens/s, RAM or battery figures for SmolLM2/SmolVLM2 on phones.
- The official Transformers.js v4 release notes and HF blog were not reached, so details rest on secondary sources.
- The HuggingSnap source code and repo were not located.

---

## 5. Hub mechanics for releasing our model correctly

### Takeaway
Release with a `README.md` model card whose YAML metadata covers `license` (or `other` + `license_name` + `license_link`), `library_name: transformers` (mandatory explicitly for repos created after Aug 2024), `datasets`, `language`, `pipeline_tag`, and eval results. A from-scratch base model must **omit `base_model`**. Its derived instruct, LoRA, merged and GGUF repos should point back to it with `base_model` (+ optional `base_model_relation`) so the Hub model tree links them.

### Cited Findings
- The model card is the `README.md` with a YAML header. Metadata can be edited through the UI, by editing the YAML, or programmatically with `huggingface_hub` — [Hub docs: model cards](https://huggingface.co/docs/hub/model-cards)
- `base_model` takes a single model ID or a list (for merges). The Hub infers the relation (`adapter`, `merge`, `quantized`, `finetune`), which can be set explicitly with `base_model_relation: quantized`. A `new_version` field links to a successor model — [Hub docs](https://huggingface.co/docs/hub/model-cards)
- Custom licences use `license: other`, `license_name`, `license_link`, or a LICENSE file in the repo — [Hub docs](https://huggingface.co/docs/hub/model-cards)
- "For model repos created after August 2024", a `config.json` no longer implies transformers, so `library_name: transformers` must be set explicitly — [Hub docs](https://huggingface.co/docs/hub/model-cards)
- `datasets:` entries render "Datasets used to train" and make the model filterable by dataset. A new, simpler eval-results metadata format exists alongside the model-index spec. Paper links auto-tag `arxiv:<id>`, and a `not-for-all-audiences` tag is available — [Hub docs](https://huggingface.co/docs/hub/model-cards)
- Precedents:
  - HF's own models set `base_model` from instruct to base. Example: SmolLM2-360M-Instruct → SmolLM2-360M; SmolLM2-135M (base) has none (Hub API) — [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
  - A VLM lists multiple bases: SmolVLM-256M → SmolLM2-135M-Instruct + SigLIP — [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- Gating examples: Gemma 3 and Llama 3.2 use `gated: manual`, xLAM-60k uses `gated: auto`, and Gemma 4 E2B is ungated (Hub API) — [gemma-3-270m](https://huggingface.co/google/gemma-3-270m); [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k); [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- Precedent for publishing alternate formats as separate sibling repos: LiquidAI publishes MLX 4/5/6/8-bit and bf16 variants of LFM2.5-230M, and HF publishes `SmolLM3-3B-ONNX` (Hub API) — [LiquidAI models](https://huggingface.co/LiquidAI); [SmolLM3-3B-ONNX](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-ONNX)

### Inferences
- A suggested release layout follows Hub conventions and HF and Liquid practice:
  - `org/aptivra-360m-base`: safetensors, Apache-2.0, no `base_model`, `datasets:` listing every pretraining set.
  - `org/aptivra-360m-instruct`: `base_model: org/aptivra-360m-base`, `datasets:` listing every post-training set, with attribution text for every CC-BY/ODC-BY source.
  - `org/aptivra-360m-instruct-GGUF` and `-ONNX` / `-MLX`, each with `base_model_relation: quantized`.
- If any ⚠ non-commercial dataset were used, the licence field would have to reflect it, or the dataset must be excluded. The cleanest path is exclusion.
- Gating (`manual`) suits a private early release, and the repo can later be flipped public.

### Gaps
- The Hub docs pages for gated models, GGUF uploads (`gguf` library or `llama.cpp convert_hf_to_gguf.py`), safetensors conversion, and the new eval-results format were not fetched. Their specifics are not cited here.
- A licence-compatibility rule set (e.g. whether ODC-BY training data imposes obligations on model weights) was not found in an authoritative HF source.
