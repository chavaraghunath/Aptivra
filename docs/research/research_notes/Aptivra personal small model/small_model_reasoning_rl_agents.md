# Small-Model Reasoning, RL and Agentic Tool Use (under 1.5B, with emphasis on under 1B)

Verification legend used throughout:
- **[V]** = number fetched and read from the primary source (paper abstract/HTML or official model card) during this session (2026-10-18).
- **[K]** = from the researcher's prior knowledge of the primary source; the URL is the primary source but the specific figure was NOT re-read this session. Treat as needing a check before it goes into an ADR.
- **[A]** = aggregator or third-party figure, not primary.
- Self-reported vendor/model-card numbers are claims by the authors, not independent replications, even when tagged [V].

## 1. RL with verifiable rewards (RLVR) on 0.5B-1.5B models

### Takeaway
RLVR on a *strong* 1.5B reasoning base yields large, cheap gains (DeepScaleR: AIME24 28.8 -> 43.1 for roughly 3.8k A100-hours; Tina: similar with LoRA for about $9), and even a single training example can double MATH500 on Qwen2.5-Math-1.5B. But the bulk of evidence says RLVR mostly *sharpens* behaviours already latent in the base (pass@1 rises, large-k pass@k often does not), Qwen-family results are partly confounded by pretraining contamination and spurious-reward effects, and RL from a weak base under about 1B is fragile. A from-scratch 360M model should expect RL to be a late-stage sharpener, not the source of reasoning.

### Cited Findings
**Headline small-model RLVR results**
- [V] DeepScaleR-1.5B-Preview (RL on DeepSeek-R1-Distill-Qwen-1.5B), pass@1 averaged over 16 samples: AIME24 28.8 -> 43.1; MATH500 82.8 -> 87.8; AMC23 62.9 -> 73.6; Minerva 26.5 -> 30.2; OlympiadBench 43.3 -> 50.0; average 48.9 -> 57.0. Beats o1-preview on AIME24 (40.0) and MATH500 (81.4). — [DeepScaleR model card](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
- [V] DeepScaleR recipe: about 40k problem-answer pairs (AIME 1984-2023, AMC pre-2023, Omni-MATH, Still); "iterative lengthening" of context 8K (steps 0-1040, AIME 22.9% -> 33%, batch 1024, 8xA100-80GB) -> 16K (steps 1040-1520, 33% -> 43%, 32xA100) -> 24K (32xA100). — [DeepScaleR model card](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
- [K] Total DeepScaleR compute reported in the Agentica blog as about 3,800 A100-hours (about $4,500). The blog page could not be fetched this session (Notion rendered empty), so this figure is unverified here. — [Agentica DeepScaleR blog](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
- [V] Tina: LoRA-based RL on a 1.5B base reaches AIME24 43.33% pass@1, "over 20%" reasoning gain, for about $9 USD total post-training plus evaluation cost (claimed 260x cheaper than full-parameter RL baselines). Base is a 1.5B model (per the paper body, DeepSeek-R1-Distill-Qwen-1.5B [K]). — [Tina, arXiv 2504.15777](https://arxiv.org/abs/2504.15777)
- [V] 1-shot RLVR: Qwen2.5-Math-1.5B trained with RLVR on a *single* example: MATH500 36.0 -> 73.6; six-benchmark average 17.6 -> 35.7. Matches the 1.2k-example DeepScaleR subset (73.6 / 35.9). Two examples: 74.8 / 36.6. Of the gain, only 8.6 MATH500 points (7.0 average) are beyond "format correction". Works with GRPO and PPO; also tested on Qwen2.5-Math-7B, Llama-3.2-3B-Instruct, R1-Distill-Qwen-1.5B. Observed "post-saturation generalisation" (test acc keeps rising after train acc saturates); gains come from the policy-gradient loss; entropy bonus helps exploration. — [Wang et al., arXiv 2504.20571](https://arxiv.org/abs/2504.20571)
- [V] SimpleRL-Zoo: zero-RL (RL directly on base, no SFT) across 10 base models including all Qwen2.5 sizes 0.5B-32B, Llama3-8B, Mistral-7B/24B; "substantial improvements in both reasoning accuracy and response length across most settings"; key levers were format-reward adjustment and controlling query difficulty; first "aha moment" observed in non-Qwen small models; longer responses do not always mean more verification behaviour. — [Zeng et al., arXiv 2503.18892](https://arxiv.org/abs/2503.18892)
- [V] ProRL (Nemotron-Research-Reasoning-Qwen-1.5B): prolonged RL training; "RL-trained models consistently outperform base models across a wide range of pass@k evaluations", including on tasks where the base fails regardless of sample count; boundary expansion correlates with base competence and training duration. — [Liu et al., arXiv 2505.24864](https://arxiv.org/abs/2505.24864). [K] Paper body reports over 2k RL steps on about 136k problems spanning math, code, STEM, logic puzzles and instruction following, with KL control and periodic reference-policy resets; compute on the order of 16k H100-hours. Not re-verified.
- [K] TinyZero reproduced R1-Zero "aha" behaviour on the Countdown/multiplication tasks with Qwen2.5-3B base for under $30; the author reported that the 0.5B base did *not* learn to reason under the same recipe, while 1.5B+ did. — [TinyZero GitHub](https://github.com/Jiayi-Pan/TinyZero)
- [K] Open-R1 (Hugging Face) is a fully open reproduction of the R1 pipeline (distillation datasets such as Mixture-of-Thoughts / OpenR1-Math-220k plus GRPO trainer in TRL); it is infrastructure rather than a sub-1B result. — [Open-R1 GitHub](https://github.com/huggingface/open-r1)

**Algorithm variants (all applicable at small scale)**
- [V] Dr. GRPO: identifies an optimisation bias in GRPO that "artificially increases response length (especially for incorrect outputs)"; the unbiased fix improves token efficiency without hurting accuracy; minimalist R1-Zero recipe gives 43.3% AIME24 on a 7B base. Also finds Qwen2.5 bases reason well even with no prompt template (suggesting pretraining bias) and DeepSeek-V3-Base already shows "aha". — [Liu et al., arXiv 2503.20783](https://arxiv.org/abs/2503.20783). [K] The specific biases are per-response length normalisation and per-group std normalisation of advantages.
- [K] DAPO (decoupled clip-higher, dynamic sampling that drops all-correct/all-wrong groups, token-level loss, overlong reward shaping) reached AIME24 50 on Qwen2.5-32B; results are 32B, not small-scale. — [DAPO, arXiv 2503.14476](https://arxiv.org/abs/2503.14476)
- [K] REINFORCE++ removes the critic and uses global (batch) advantage normalisation; claims better stability than GRPO and lower cost than PPO. — [REINFORCE++, arXiv 2501.03262](https://arxiv.org/abs/2501.03262)
- [V] The six RLVR algorithms tested in the "does RL incentivise reasoning" study "perform similarly" and all remain far below the base model's pass@k upper bound — algorithm choice is a second-order effect. — [Yue et al., arXiv 2504.13837](https://arxiv.org/abs/2504.13837)

**Does RL teach new skills or sharpen?**
- [V] Yue et al.: RLVR models beat the base at small k (pass@1) but the *base* has higher pass@k at large k; coverage and perplexity analysis says the reasoning is bounded by the base; distillation (unlike RLVR) "can introduce new reasoning patterns" and expand capability. — [Yue et al., arXiv 2504.13837](https://arxiv.org/abs/2504.13837)
- [V] Counter-evidence: ProRL reports pass@k gains even at large k and on tasks the base never solves, conditional on long training and moderate base competence. — [ProRL, arXiv 2505.24864](https://arxiv.org/abs/2505.24864)
- [V] Spurious rewards: on Qwen2.5-Math-7B, GRPO with ground-truth reward gives +29.1 MATH500 points, while *random* reward gives +21.4. Spurious rewards "often fail" on Llama3 and OLMo2. Mechanism attributed to GRPO's clipping bias amplifying pretrained behaviours (code-style reasoning rose from 65% to over 90% of responses). — [Shao et al., arXiv 2506.10947](https://arxiv.org/abs/2506.10947)
- [V] 1-shot RLVR: much of the 1.5B gain is format correction (only 8.6 of 37.6 MATH500 points are non-format). — [arXiv 2504.20571](https://arxiv.org/abs/2504.20571)

**Minimum base quality**
- [K] TinyZero: 0.5B base failed, 1.5B/3B succeeded on Countdown. — [TinyZero GitHub](https://github.com/Jiayi-Pan/TinyZero)
- [V] ProRL: boundary expansion "correlates with the base model's task competence". — [arXiv 2505.24864](https://arxiv.org/abs/2505.24864)
- [K] DeepSeek-R1 paper: RL directly on small dense models underperforms distilling R1 into them (shown for Qwen-32B: RL-only Qwen-32B-Zero is well below R1-Distill-Qwen-32B). — [DeepSeek-R1, arXiv 2501.12948](https://arxiv.org/abs/2501.12948); the model card restates that distilled reasoning beats "reasoning patterns discovered through RL on small models" [V] — [R1-Distill-Qwen-1.5B card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

**Meta MobileLLM-R1 (most relevant sub-1B reference point)**
- [V] Post-trained pass@1: MobileLLM-R1-950M: MATH500 74.0, GSM8K 67.5, AIME24 15.5, AIME25 16.3, LiveCodeBench-v6 19.9. Qwen3-0.6B: 73.0 / 79.2 / 11.3 / 17.0 / 14.9. Qwen2.5-0.5B-Instruct: 31.2 / 48.1 / 0.1 / 0.3 / 3.6. MobileLLM-R1-360M: MATH500 28.4, GSM8K 24.5, LCB 5.1 vs SmolLM2-360M-Instruct 3.4 / 8.1 / 0.7. MobileLLM-R1-140M: 6.2 / 4.1 / 1.7 vs SmolLM2-135M-Instruct 3.0 / 2.4 / 0.0. — [MobileLLM-R1-950M card](https://huggingface.co/facebook/MobileLLM-R1-950M)
- [V] Recipe: pretraining from scratch on about 2T high-quality tokens (two 2T phases, fewer than 5T total), two 100B-token mid-training phases with logit-KD from Llama-3.1-8B-Instruct, then SFT only (866K general + 6.2M reasoning samples). No RL stage is described. Licence: FAIR Non-Commercial research licence. — [MobileLLM-R1-950M card](https://huggingface.co/facebook/MobileLLM-R1-950M)
- [V] Base (pre-SFT) MobileLLM-R1-360M: MATH500 (4-shot) 13.4, GSM8K (8-shot) 39.4, HumanEval 32.9. — [same card](https://huggingface.co/facebook/MobileLLM-R1-950M)

### Inferences
- The strongest sub-1B reasoning results (MobileLLM-R1, Qwen3-0.6B) come from data quality + distillation + SFT, not RL. MobileLLM-R1-360M's 28.4 MATH500 is the realistic ceiling reference for a 360M from-scratch model with a very large, high-quality reasoning SFT corpus; it is roughly what Qwen2.5-0.5B-Instruct got with general post-training (31.2).
- RLVR value at 360M is unproven in the literature found; the TinyZero 0.5B failure and the "bounded by base" result suggest running RL only after SFT has given non-trivial pass@k (the model must sometimes sample a correct answer for a verifiable reward to have signal).
- Because spurious-reward and template effects are Qwen-specific, any Aptivra RL experiment needs a random-reward and format-only-reward control to separate real learning from format correction. This matches the project's "noise floor" memory rule.
- Cheap compute: LoRA-RL (Tina) and 1-shot/few-shot RLVR suggest RL experiments on a 360M model fit in single-GPU-hours, not thousands.

### Gaps
- DeepScaleR GPU-hours could not be read from the primary blog this session.
- No primary-source RLVR result found for models under 0.5B. SimpleRL-Zoo includes Qwen2.5-0.5B but the per-size numbers were not available on the abstract page.
- ProRL compute and per-benchmark numbers not re-verified.
- No independent replication of DeepScaleR/Tina numbers checked.

## 2. Reasoning distillation into small models, and licence implications

### Takeaway
Distillation is the dominant, most compute-efficient way to give small models reasoning: R1-Distill-Qwen-1.5B (800k R1 traces) hits AIME24 28.9 / MATH500 83.9, and curated sets of about 1k examples (s1, LIMO) suffice at 32B. But models at or under about 3B learn poorly from long chain-of-thought ("learnability gap"), so short/mixed traces work better for a 360M. Licence: R1 outputs are MIT and explicitly allow distillation; Qwen and Llama outputs carry licence terms that must be checked per model.

### Cited Findings
- [V] DeepSeek-R1-Distill-Qwen-1.5B: base Qwen2.5-Math-1.5B, SFT on 800k samples curated with DeepSeek-R1; AIME24 28.9 pass@1 / 52.7 cons@64; MATH500 83.9; GPQA-Diamond 33.8; LiveCodeBench 16.9; Codeforces rating 954. — [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [V] Licence: R1 code and weights are MIT, which "supports commercial use and derivative works, including distillation for training other LLMs"; the 1.5B distill derives from Qwen2.5 (Apache 2.0). — [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [K] The Llama-based R1 distills (8B, 70B) remain under the Llama 3.1/3.3 Community Licence, which requires "Built with Llama" attribution and that derived model names begin with "Llama"; the licence also restricts using Llama outputs to improve non-Llama models only in earlier (Llama 2) versions — the Llama 3.1 licence relaxed this. Verify current text before use. — [Llama 3.1 licence](https://www.llama.com/llama3_1/license/)
- [K] Some Qwen models (e.g. Qwen2.5-3B and -72B) use the "Qwen Research/Qwen License" rather than Apache 2.0; Qwen3 open-weight models are Apache 2.0. — [Qwen3 technical report, arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- [V] MobileLLM-R1 used Llama-3.1-8B-Instruct as KD teacher and is consequently released under a non-commercial licence. — [MobileLLM-R1 card](https://huggingface.co/facebook/MobileLLM-R1-950M)
- [V] Small Model Learnability Gap: models of 3B or fewer "do not consistently benefit from long CoT reasoning or distillation from larger models" and do better on shorter, simpler chains; Mix Distillation (mixing long and short CoT, or large- and small-teacher traces) significantly improves small-model reasoning over either alone. — [Li et al., arXiv 2502.12143](https://arxiv.org/abs/2502.12143)
- [V] Yue et al.: distillation, unlike RLVR, can introduce genuinely new reasoning patterns. — [arXiv 2504.13837](https://arxiv.org/abs/2504.13837)
- [V] LIMO: AIME24 63.3% and MATH500 95.6% (vs 6.5% and 59.2% for prior SFT models) using about 1% of prior approaches' training data; claims to beat models trained on 100x more data, with 45.8% absolute OOD gain. — [LIMO, arXiv 2502.03387](https://arxiv.org/abs/2502.03387). [K] 817 curated samples on Qwen2.5-32B-Instruct.
- [K] s1: 1,000 curated questions with Gemini-thinking traces, SFT of Qwen2.5-32B-Instruct in 26 minutes on 16 H100s, plus "budget forcing" (appending "Wait" to extend thinking) for test-time scaling; s1-32B exceeds o1-preview on competition math. — [s1, arXiv 2501.19393](https://arxiv.org/abs/2501.19393)
- [V] OpenThoughts: OpenThoughts2-1M and OpenThoughts3 (1.2M examples, QwQ-32B teacher, over 1,000 ablation experiments); OpenThoughts3-7B scores AIME25 53%, LiveCodeBench 51%, GPQA-D 54%, i.e. +15.3 / +17.2 / +20.5 points over R1-Distill-Qwen-7B. — [OpenThoughts, arXiv 2506.04178](https://arxiv.org/abs/2506.04178)
- [V] MobileLLM-R1 reasoning SFT used 6.2M samples, about 8x the R1 distillation set, for a 950M model. — [MobileLLM-R1 card](https://huggingface.co/facebook/MobileLLM-R1-950M)

### Inferences
- "Less is more" (s1, LIMO) has been demonstrated only on 32B models with strong latent knowledge; small models need far more data (800k for 1.5B, 6.2M for 950M). Do not plan a 360M reasoning stage around 1k examples.
- For a 360M on-device model, prefer short, verified, teacher traces (or mixed long/short) over 16k-token R1-style traces; long traces also cost on-device latency.
- For a licence-clean pipeline: teacher outputs from MIT (DeepSeek-R1) or Apache-2.0 (Qwen3) models are the safest; Llama-teacher distillation propagates Llama licence obligations and Meta's own sub-1B reasoning model is non-commercial.

### Gaps
- No primary benchmark numbers for Mix Distillation were available on the abstract page.
- Licence texts for Llama 3.x and each Qwen release were not re-read this session; legal review needed.

## 3. Test-time compute for small models

### Takeaway
With a strong external PRM and very large sample budgets, sub-1B to 3B policies can match much larger models on MATH-500 (1B Llama at N=512 edges 405B Llama, 72.2 vs 71.4), but the gain is narrow (fails on AIME for 1B), relies on 7B-72B reward models, and the "100x-1000x fewer FLOPS" claim does not translate to on-device latency, where serial sampling of hundreds of long traces dominates.

### Cited Findings
- [V] Liu et al. compute-optimal TTS: Llama-3.2-1B-Instruct with TTS (N=512) vs Llama-3.1-405B-Instruct CoT: MATH-500 72.2 vs 71.4, but AIME24 10.0 vs 23.3. Qwen2.5-0.5B-Instruct + TTS vs GPT-4o: MATH-500 76.4 vs 74.6; AIME24 10.0 vs 9.3. Llama-3.2-3B + TTS vs 405B: 75.6 vs 71.4 MATH-500; 30.0 vs 23.3 AIME24. R1-Distill-Qwen-1.5B + TTS vs o1-preview: MATH-500 91.6 vs 85.5; AIME24 63.3 vs 44.6. — [Liu et al., arXiv 2502.06703 (HTML)](https://arxiv.org/html/2502.06703)
- [V] Setup: budgets N in {4, 16, 64, 256} (N=512 for the 1B headline), beam width 4; PRMs used include Math-Shepherd-7B, RLHFlow-PRM-8B variants, Skywork-PRM-1.5B/7B, Qwen2.5-Math-PRM-7B/72B. Optimal strategy depends on policy, PRM and problem difficulty. Total-FLOPs claim: 100x-1000x less, e.g. Llama-3.2-3B 1.62e23 vs 405B 3.65e25. Math only. — [arXiv 2502.06703](https://arxiv.org/html/2502.06703)
- [V] DeepSeek-R1-Distill-Qwen-1.5B self-consistency: AIME24 28.9 pass@1 -> 52.7 cons@64. — [model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [K] Snell et al.: compute-optimal test-time scaling can be over 4x more efficient than best-of-N, and on easy/medium problems a smaller model with test-time compute can beat a 14x larger model in FLOPs-matched comparison; not on hard problems. — [Snell et al., arXiv 2408.03314](https://arxiv.org/abs/2408.03314)
- [K] Large Language Monkeys: coverage (pass@k) grows log-linearly with samples over orders of magnitude, but majority vote and reward-model selection plateau, so gains are only realisable where an automatic verifier exists (code tests, proof checkers). — [Brown et al., arXiv 2407.21787](https://arxiv.org/abs/2407.21787)

### Inferences
- On a phone, N=64-512 samples of thousand-token traces is seconds-to-minutes of wall time and substantial battery; best-of-N is viable only for short outputs or where a cheap verifier (tool result, schema check, unit test) exists. The FLOPs accounting in Liu et al. includes the large models' FLOPs per token but does not measure on-device latency or energy.
- A PRM larger than the policy defeats the on-device purpose; the realistic on-device options are self-consistency with small N and deterministic verifiers.

### Gaps
- No primary measurements of on-device latency/energy for small-model TTS were found in this session.
- No TTS results below 0.5B were found.

## 4. Tool use and agents at small scale

### Takeaway
Sub-2B models can be highly accurate at *single-turn* function calling (AST-style BFCL non-live 69-83%), but multi-turn/agentic reliability collapses below about 1.5B (BFCL multi-turn 3.6% for Qwen3-0.6B, 15.6% Hammer2.1-1.5B, 36% xLAM-2-1B per an aggregator). Specialised fine-tuning (xLAM, Hammer, Octopus) matters more than size for single calls.

### Cited Findings
- [A] BFCL (aggregator snapshots, May 2026; not the same BFCL snapshot for all models): xLAM-2-1b-fc-r overall 30.4%, non-live AST 69.0%, live 55.1%, multi-turn 36.0%. Hammer2.1-1.5b overall 27.9%, non-live AST 83.0%, live 69.5%, multi-turn 15.6%. Qwen3-0.6B overall 23.9%, non-live AST 71.8%, live 56.6%, multi-turn 3.6%. — [benchmarklist xLAM-2-1b](https://benchmarklist.com/models/xlam-2-1b-fc-r/); [benchmarklist Hammer2.1-1.5b](https://benchmarklist.com/models/hammer2.1-1.5b/); [benchmarklist Qwen3-0.6B](https://benchmarklist.com/models/qwen3-0.6b-6/). Verify against the official [Gorilla BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html).
- [A] A third-party Hugging Face repo claims xLAM-1b-fc-r reaches 78.94% on BFCL v3; this is an unverified, earlier-version claim. — [HF repo](https://huggingface.co/GatekeeperZA/xLAM-1b-fc-r-RKLLM-v1.2.3)
- [K] xLAM (Salesforce): family incl. xLAM-1B-fc (later "Tiny Giant"), trained on APIGen-generated verified function-calling data; the 1B model was reported to outperform much larger general models on BFCL v1/v2 at the time. — [xLAM, arXiv 2409.03215](https://arxiv.org/abs/2409.03215)
- [K] Hammer: function-masking (randomising function/parameter names) to make small models rely on descriptions, improving robustness across benchmarks. — [Hammer, arXiv 2410.04587](https://arxiv.org/abs/2410.04587)
- [K] Octopus v2: 2B Gemma model mapping Android API calls to "functional tokens", self-reported about 99.5% function-call accuracy on its own Android benchmark and latency reduced versus RAG-based GPT-4 calling. Narrow, closed-world benchmark. — [Octopus v2, arXiv 2404.01744](https://arxiv.org/abs/2404.01744)
- [K] tau-bench (multi-turn tool-agent-user conversations in retail/airline domains with pass^k reliability) shows even frontier models are below 50% success and pass^8 under 25% in retail at release; small models were not evaluated there. — [tau-bench, arXiv 2406.12045](https://arxiv.org/abs/2406.12045)
- [V] HyFunc proposes a hybrid cascade (small model + larger model) and dynamic templating to speed agentic function calls — evidence that practitioners route around small-model limits. — [HyFunc, arXiv 2602.13665](https://arxiv.org/pdf/2602.13665)

### Inferences
- For an on-device assistant, a 360M model can plausibly be trained to emit single, schema-valid tool calls for a closed set of personal-assistant tools (calendar, reminders, contacts) — Octopus/xLAM evidence — but should not be relied on for open multi-turn agent planning. A cloud fallback or a cascade is the realistic design for multi-step tasks.
- Constrained decoding (grammar/JSON-schema masks, e.g. llama.cpp GBNF, Outlines, XGrammar) guarantees syntactic validity regardless of size; it does not fix wrong tool selection or arguments. This is the researcher's knowledge, not verified here.

### Gaps
- Official BFCL v3/v4 numbers for these models were not fetched; aggregator figures are mixed snapshots.
- No ToolBench or tau-bench results for sub-1B models were found.
- No primary-source study of "minimum size for reliable tool use" was found; the inference above is from benchmark patterns.

## 5. Self-improvement and model collapse

### Takeaway
Self-improvement loops (STaR, ReST-EM, self-rewarding) work when outputs are filtered by a ground-truth verifier or an external signal, and usually plateau within 2-3 iterations; all landmark results are on 6B-70B models. Training recursively on unfiltered self-generated data causes collapse, which is avoided when real data is accumulated rather than replaced.

### Cited Findings
- [K] STaR: iteratively fine-tunes on self-generated rationales that reach the correct answer (with "rationalisation" hints for failures); GPT-J 6B on CommonsenseQA approached a 30x larger fine-tuned model. — [Zelikman et al., arXiv 2203.14465](https://arxiv.org/abs/2203.14465)
- [K] ReST-EM: EM-style self-training with binary verifier filtering on PaLM 2 for MATH and APPS; beat human-data SFT, but gains saturated after a few iterations and overfitting appeared with more iterations. — [Singh et al., arXiv 2312.06585](https://arxiv.org/abs/2312.06585)
- [K] Self-Rewarding LMs: Llama-2-70B as its own judge plus iterative DPO, three iterations, improved AlpacaEval 2.0; the authors note saturation is unexplored and results are at 70B. — [Yuan et al., arXiv 2401.10020](https://arxiv.org/abs/2401.10020)
- [K] Model collapse: recursive training on generated data causes tails of the distribution to disappear. — [Shumailov et al., Nature 2024](https://www.nature.com/articles/s41586-024-07566-y)
- [K] Accumulating synthetic data alongside the original real data (rather than replacing it) avoids collapse with a bounded test error. — [Gerstgrasser et al., arXiv 2404.01413](https://arxiv.org/abs/2404.01413)
- [V] RLVR is itself a filtered self-improvement loop and the evidence that it is bounded by the base (Yue et al.) is the closest small-scale analogue of the plateau. — [arXiv 2504.13837](https://arxiv.org/abs/2504.13837)

### Inferences
- A small personal model that learns from its own outputs must anchor to a verifier (tool success, user confirmation) and keep a replay buffer of real/teacher data; unverified self-training on the device is a collapse risk.

### Gaps
- No self-improvement study on models under 1B was found in this session.

## 6. Learning from user feedback on device (RLHF, DPO, KTO, online preference learning)

### Takeaway
DPO and KTO are the practical on-device options because they need no reward model or sampling-heavy RL loop, and KTO accepts unpaired thumbs-up/down signals that a personal assistant naturally collects. The literature on actually running preference optimisation on a phone is thin and mostly cloud-assisted or PEFT-based.

### Cited Findings
- [K] DPO: closed-form reformulation of the RLHF objective trained as classification on preference pairs, matching or beating PPO-RLHF on summarisation and dialogue without a reward model. — [Rafailov et al., arXiv 2305.18290](https://arxiv.org/abs/2305.18290)
- [K] KTO: optimises a prospect-theory utility from binary desirable/undesirable labels (no pairs), matching or exceeding DPO at 1B-30B scale. — [Ethayarajh et al., arXiv 2402.01306](https://arxiv.org/abs/2402.01306)
- [V] (search snippets only) CDCDA-PLM: a server-side LLM generates synthetic personal data, then parameter-efficient modules are fine-tuned on a small on-device model; evaluated on six personalisation tasks. — [arXiv 2508.21313](https://arxiv.org/html/2508.21313v2)
- [V] (search snippet) XPerT reports cutting on-device LLM personalisation compute by 83% and improving data efficiency by 51%. — [arXiv 2504.13938](https://arxiv.org/abs/2504.13938v1)
- [V] (search snippet) A NeurIPS 2026 paper, "Continual On-Device Personalization of Small Language Models from Hindsight", uses self-distillation from later user activity with a physical-device test and reports gains over reward-based learning and rejection fine-tuning. — [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/171331)
- [V] (search snippet) LSRP: privacy-preserving cloud-device collaboration with an SMFB-DPO comparison. — [arXiv 2505.05031](https://arxiv.org/pdf/2505.05031)

### Inferences
- For Aptivra: implicit signals (user edits, accepts, rejects) map naturally to KTO-style unpaired labels or to hindsight self-distillation, both cheaper than online RL. LoRA-sized updates (as in Tina) keep on-device compute and rollback tractable; the forgetting and collapse risks from Section 5 apply.

### Gaps
- The four on-device papers above were seen only as search snippets; their numbers and methods were not read in full.
- No primary measurements of DPO/KTO memory or energy on phone hardware were found.
