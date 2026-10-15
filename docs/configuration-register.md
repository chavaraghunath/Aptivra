# Configuration Register — Aptivra

Every runtime-tunable setting in this repository is listed here. **An
unregistered tunable does not exist**: if a value that a user, operator, or
environment should own is not in this table, that is a Configurability Mandate
violation and it stops the line (see `AGENTS.md`).

A row is added in the **same commit** that introduces the setting. Removing a
row requires the setting to be genuinely gone, not merely undocumented.

| Setting | Scope | Type | Default | Safe limits | RBAC role | Env-specific | Feature flag | API / config-as-code path | Requires restart |
|---------|-------|------|---------|-------------|-----------|--------------|--------------|---------------------------|------------------|
| `task_stream.num_train_per_task` | global | integer | `500` | 10–10000 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `task_stream.num_val_per_task` | global | integer | `100` | 5–2000 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `task_stream.num_test_per_task` | global | integer | `300` | 5–2000 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `task_stream.seed` | global | integer | `42` | 0–2^31-1 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `task_stream.max_generation_attempts` | global | integer | `100` | 1–1000 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `evaluation.max_new_tokens` | global | integer | `128` | 1–512 | researcher | yes | — | `configs/task_stream.yaml` | no |
| `model.d_model` | global | integer | `512` | 64–2048 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `model.n_layers` | global | integer | `8` | 1–32 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `model.n_heads` | global | integer | `8` | 1–32 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `model.d_ff` | global | integer | `2048` | 256–8192 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `model.max_seq_len` | global | integer | `512` | 1–512 | researcher | yes | — | `configs/backbone_200m.yaml` | no |
| `baseline.learning_rate` | service | float | `0.0001` | 1e-6–1e-2 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `baseline.replay_capacity` | service | integer | `500` | 10–50000 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `baseline.ewc_lambda` | service | float | `400.0` | 0.1–10000.0 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `execution.platform` | service | enum | `pytorch` | `pytorch`, `tiered_c` | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `multitier.enable_nvme_streaming` | service | boolean | `true` | `true`, `false` | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `multitier.prefetch_lookahead_layers` | service | integer | `1` | 0–4 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `multitier.hot_expert_pin_gb` | service | float | `4.0` | 0.0–128.0 | researcher | yes | — | `configs/backbone_50m.yaml` | no |
| `inference.model_repo_id` | global | string | `raghunath1/aptivra-200m-tiered-memory` | non-empty HF repo id | operator | yes | — | `configs/inference.yaml` | no |
| `inference.local_export_dir` | global | path | `scratch/hf_export` | directory with the Transformers layout or legacy multitier shards | operator | yes | — | `configs/inference.yaml` | no |
| `inference.device` | global | enum | `auto` | `auto`, `cuda`, `mps`, `cpu` | operator | yes | — | `configs/inference.yaml` | no |
| `inference.max_new_tokens` | global | integer | `96` | 1–512 | operator | yes | — | `configs/inference.yaml` | no |
| `inference.use_moe_at_inference` | global | boolean | `false` | `true`, `false` | operator | yes | — | `configs/inference.yaml` | no |
| `inference.server.host` | service | string | `127.0.0.1` | valid bind address | operator | yes | — | `configs/inference.yaml` | no |
| `inference.server.port` | service | integer | `8080` | 1024–65535 | operator | yes | — | `configs/inference.yaml` | no |
| `inference.server.model_name` | service | string | `aptivra-200m-tiered-memory` | non-empty | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.gradio_host` | service | string | `127.0.0.1` | valid bind address | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.gradio_port` | service | integer | `7860` | 1024–65535 | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.title` | service | string | `Aptivra Chat` | non-empty | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.single_turn` | service | boolean | `true` | `true`, `false` | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.sanitize_replies` | service | boolean | `true` | `true`, `false` | operator | yes | — | `configs/inference.yaml` | no |
| `inference.chat.max_reply_chars` | service | integer | `280` | 32–2048 | operator | yes | — | `configs/inference.yaml` | no |
| `model_card_release_gate.heldout_bpb_unigram_margin_bits` | global | float | `0.0` | ≥ 0.0 (validated) | researcher | yes | — | `configs/model_card_release_gate.yaml` | no |
| `model_card_release_gate.minimum_trained_prompt_samples` | global | integer | `7` | ≥ 1 (validated) | researcher | yes | — | `configs/model_card_release_gate.yaml` | no |
| `model_card_release_gate.minimum_heldout_prompt_samples` | global | integer | `24` | ≥ 1 (validated) | researcher | yes | — | `configs/model_card_release_gate.yaml` | no |
| `model_card_release_gate.minimum_heldout_text_bytes` | global | integer | `8192` | ≥ 1 (validated; bundled corpus is larger) | researcher | yes | — | `configs/model_card_release_gate.yaml` | no |
| `model_card_release_gate.supplied_record_metric_tolerance` | global | float | `0.000001` | 0.0–0.001 inclusive (validated) | researcher | yes | — | `configs/model_card_release_gate.yaml` | no |
| `verifier_sandbox.timeout_seconds` | service | float | `1.0` | 0.05–30.0 seconds (validated); bounds generated-code hangs in a child process, not a security boundary against hostile code | researcher | yes | — | `configs/verifier_sandbox.yaml` | no |
| `verifier_sandbox.memory_limit_mb` | service | integer | `512` | 64–2048 MiB (validated; applied where supported); contains accidental resource use, not hostile code | researcher | yes | — | `configs/verifier_sandbox.yaml` | no |
| `base_model_evaluation.candidates[]` | global | object list | eight pinned records (seven selectable, one reference); Amendment 4g fixes the five measured selection records | unique non-empty id/source; immutable revision preferred; licence/architecture optional fallback metadata; `reference_only` boolean; loader is `transformers` or `aptivra_legacy`; Amendment 4g rejects any candidate set other than its pre-registered five | researcher | yes | — | `configs/base_model_candidates.yaml`, `configs/base_model_candidates_amendment_4g.yaml` | no |
| `base_model_evaluation.measurement.device` | environment | enum | `auto` | `auto`, `cpu`, `mps`, `cuda` with availability validation | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.pristine_model_device` | environment | enum | `candidate_device` | `candidate_device`, `host`; placement does not reduce total unified-memory use on Apple silicon | researcher | yes | — | `configs/base_model_candidates_amendment_4d.yaml` | no |
| `base_model_evaluation.measurement.release_pristine_before_training` | environment | boolean | `false` | when `true`, pristine scoring and generation run first, weights are snapshotted to temporary disk, and one live model is restored before each isolated probe; Amendment 4e requires `true` | researcher | yes | set `false` to restore the legacy live-pristine order | `configs/base_model_candidates_amendment_4e.yaml` | no |
| `base_model_evaluation.measurement.local_files_only` | environment | boolean | `true` | `true`, `false`; `true` forbids Hub downloads | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.torch_dtype` | environment | enum | `float32` | `float32`, `float16`, `bfloat16` | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.deterministic_algorithms` | environment | boolean | `true` | `true`, `false` | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.deterministic_warn_only` | environment | boolean | `true` | `true`, `false`; controls unsupported deterministic-operator handling | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.seed` | global | integer | `20261014` | 0–2^31-1 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.variance_seeds` | global | integer list | `20261014`, `20261015`, `20261016` | at least three unique values in 0–2^31-1; first must equal `seed` | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.variance_top_candidates` | global | integer | `2` | 1–selectable candidate count; one is permitted only for smoke configurations | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.train_items_per_family` | global | integer | `256` | 1–10000 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.validation_items_per_family` | global | integer | `100` | 1–2000 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.test_items_per_family` | global | integer | `100` | 1–2000 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.max_generation_attempts` | global | integer | `1000` | 1–1000 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.max_new_tokens` | global | integer | `48` | 1–2048 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.probe_steps` | global | integer | `200` | 1–10000 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.probe_batch_size` | global | integer | `8` | 1–256 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.probe_micro_batch_sizes` | environment | integer list | `8`, `4`, `2`, `1` | unique descending divisors of `probe_batch_size`; must begin with the effective batch and end with 1; largest measured fitting value is selected | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.target_only_training_loss` | global | boolean | `true` | Amendment 4 requires `true`; projects decoder hidden states through the LM head only at shifted positions whose label is not `-100` | researcher | yes | `target_only_training_loss` | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.gradient_checkpointing_fallback` | environment | boolean | `true` | `true`, `false`; when enabled, checkpointing is attempted only after micro-batch 1 without checkpointing fails | researcher | yes | `gradient_checkpointing_fallback` | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.equivalence_candidate_ids` | global | string list | `smollm2-135m`, `gpt2` | non-empty unique selectable candidate IDs; every proof candidate must pass before sweep workers start | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.equivalence_steps` | global | integer | `3` | 1–100 optimizer steps | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.equivalence_loss_absolute_tolerance` | global | float | `0.00005` | 0.0–1.0 inclusive; immutable after proof observation | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.equivalence_update_max_absolute_tolerance` | global | float | `0.000001` | 0.0–1.0 inclusive; maximum absolute parameter-update delta | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.equivalence_update_relative_l2_tolerance` | global | float | `0.0001` | 0.0–1.0 inclusive; relative L2 parameter-update delta | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.equivalence_exact_loss_absolute_tolerance` | global | float | `0.000000000001` | 0.0–1.0 inclusive; Amendment 4a float64 CPU loss delta | researcher | yes | — | `configs/base_model_candidates_amendment_4a.yaml` | no |
| `base_model_evaluation.measurement.equivalence_exact_gradient_relative_l2_tolerance` | global | float | `0.0000000001` | 0.0–1.0 inclusive; Amendment 4a float64 CPU gradient delta | researcher | yes | — | `configs/base_model_candidates_amendment_4a.yaml` | no |
| `base_model_evaluation.measurement.equivalence_noise_floor_batches` | global | integer | `3` | 3–100 deterministic effective batches per proof model; reused as the Amendment 4b informational shape-control batch count | researcher | yes | — | `configs/base_model_candidates_amendment_4b.yaml` | no |
| `base_model_evaluation.measurement.equivalence_noise_floor_multiplier` | global | float | `2.0` | 1.0–10.0; applied batchwise and separately to gradient, update, and trajectory control floors | researcher | yes | — | `configs/base_model_candidates_amendment_4a.yaml` | no |
| `base_model_evaluation.measurement.equivalence_trajectory_steps` | global | integer | `50` | 1–1000 optimizer steps for the SmolLM2-135M outcome check | researcher | yes | — | `configs/base_model_candidates_amendment_4b.yaml` | no |
| `base_model_evaluation.measurement.equivalence_protocol` | global | enum | `amendment_4b` | `amendment_4a`, `amendment_4b`; selects the manifest-bound pre-sweep gate without changing frozen earlier evidence | researcher | yes | — | `configs/base_model_candidates_amendment_4b.yaml` | no |
| `base_model_evaluation.equivalence_gate.owner_override` | global | object | absent | when present, requires non-empty `by`, `date`, and `adr_section`; permits an owner-authorised failed equivalence gate to proceed and is copied verbatim into evidence | owner | yes | remove the object to restore fail-closed behavior | `configs/base_model_candidates_amendment_4b.yaml` | no |
| `base_model_evaluation.measurement.equivalence_shape_control_chunk_size` | environment | integer | `1` | 1–32768 sequence positions per LM-head call in the non-gating Amendment 4b shape-changing diagnostic | researcher | yes | — | `configs/base_model_candidates_amendment_4b.yaml` | no |
| `base_model_evaluation.measurement.equivalence_outcome_seeds` | global | integer list | `0`, `1` | exactly two distinct values in 0–2^31-1; first seed is shared by old/new and both seed old-recipe runs | researcher | yes | — | `configs/base_model_candidates_amendment_4b.yaml` | no |
| `base_model_evaluation.measurement.probe_learning_rates` | global | float list | `0.00001`, `0.00003`, `0.0001` | non-empty unique values, each 1e-8–1.0; candidate-wide selection uses validation-only probes, then retrains the selected rate before one test evaluation | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.probe_warmup_steps` | global | integer | `20` | 0–9999 and less than `probe_steps` | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.probe_weight_decay` | global | float | `0.0` | 0.0–1.0 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.evaluation_batch_size` | environment | integer | `16` | 1–512 prompts | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.confidence_level` | global | float | `0.95` | 0.8–0.999; two-sided pooled Wilson and family-clustered Student-t intervals | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.generation_benchmark_prompts` | global | integer | `2` | 1–1000 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.generation_benchmark_new_tokens` | global | integer | `32` | 1–2048 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.generation_benchmark_warmup_runs` | global | integer | `1` | 0–100 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.generation_benchmark_repetitions` | global | integer | `3` | 1–100 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.scoring_stride_tokens` | global | integer | `256` | 1–32768; capped at context length minus one | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.load_retries` | environment | integer | `4` | 1–20; transient load errors only | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.load_retry_backoff_seconds` | environment | float | `2.0` | 0.0–300.0 seconds with exponential backoff | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.memory_sample_interval_seconds` | environment | float | `0.02` | 0.001–10.0 seconds | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.swap_sample_interval_seconds` | environment | float | `0.5` | 0.05–60.0 seconds; used by both candidate and parent-process swap monitors | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.environment_minimum_free_memory_percent` | environment | float | `5.0` | 0.0–100.0%; critical runtime floor below which a result cannot be trusted and the attempt aborts `environment_invalid` | researcher | yes | — | `configs/base_model_candidates_amendment_4f.yaml` | no |
| `base_model_evaluation.measurement.environment_recovery_timeout_seconds` | environment | float | `900.0` | 0.0–86400.0 seconds to wait for normal host pressure before consuming an attempt | researcher | yes | — | `configs/base_model_candidates_amendment_4c.yaml` | no |
| `base_model_evaluation.measurement.environment_max_attempts` | environment | integer | `2` | 1–20 total attempts per candidate or repeat seed; exhaustion records `not_measured_environment` | researcher | yes | — | `configs/base_model_candidates_amendment_4c.yaml` | no |
| `base_model_evaluation.measurement.prestart_minimum_free_memory_percent` | environment | float | `70.0` | 0.0–100.0%; quiet-machine gate required before a candidate child is created | researcher | yes | — | `configs/base_model_candidates_amendment_4e.yaml` | no |
| `base_model_evaluation.measurement.prestart_max_other_process_rss_bytes` | environment | integer | `2147483648` | 1–2^63-1 bytes; no process outside the orchestrator tree may exceed this RSS before start | researcher | yes | — | `configs/base_model_candidates_amendment_4e.yaml` | no |
| `base_model_evaluation.measurement.prestart_timeout_seconds` | environment | float | `900.0` | 0.0–86400.0 seconds; expiry records `not_measured_environment` without starting the model process | researcher | yes | — | `configs/base_model_candidates_amendment_4e.yaml` | no |
| `base_model_evaluation.measurement.prestart_poll_interval_seconds` | environment | float | `5.0` | 0.05–300.0 seconds between quiet-machine samples | researcher | yes | — | `configs/base_model_candidates_amendment_4e.yaml` | no |
| `base_model_evaluation.measurement.candidate_timeout_seconds` | environment | float | `21600.0` | 1.0–86400.0 seconds per isolated candidate or repeat seed; expiry aborts `environment_invalid` | researcher | yes | — | `configs/base_model_candidates_amendment_4f.yaml` | no |
| `base_model_evaluation.measurement.expected_wall_time_seconds` | environment | float | `9000.0` for Amendment 4g | 1.0–86400.0 seconds; recorded beside actual orchestration time | researcher | yes | — | `configs/base_model_candidates_amendment_4g.yaml` | no |
| `base_model_evaluation.measurement.utf8_roundtrip_samples` | global | string list | whitespace, NUL, combining, CJK, RTL, emoji samples | non-empty valid UTF-8 strings; each must round-trip after NFC normalisation, while task-stream prompt/target text must round-trip exactly | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.heldout_text_path` | global | path | bundled held-out text v1 | readable UTF-8 file; hash frozen in manifest | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.measurement.heldout_prompts_path` | global | path | bundled held-out prompt/answer v1 | valid prompt/answer JSON; hash frozen in manifest | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.decision.permissive_licenses` | global | string list | `apache-2.0`, `mit` | non-empty canonical licence allowlist | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.decision.target_memory_bytes` | environment | integer | `38654705664` | 1–2^63-1 bytes | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.decision.minimum_memory_headroom_fraction` | environment | float | `0.25` | 0.0–0.95 | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.decision.minimum_mean_learned_accuracy` | global | float | `0.0` | 0.0–1.0; rejection requires the family-clustered interval upper bound to miss the threshold | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.decision.minimum_mean_accuracy_gain` | global | float | `0.0` | 0.0–1.0; rejection requires the family-clustered interval upper bound to miss the threshold | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.decision.learnability_tie_margin` | global | float | `0.02` | 0.0–1.0 absolute accuracy; differences at or below this value tie | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.decision.clustered_confidence_interval_intersects_tie_region` | global | boolean | `true` | `true`, `false`; when true, a paired task-family difference interval intersecting `[-learnability_tie_margin, +learnability_tie_margin]` ties | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.decision.maximum_swap_growth_bytes` | environment | integer | `1073741824` | 0–2^63-1 bytes; reporting threshold only: exceeding it marks `throughput_compromised` and excludes tokens/second from tie-breaking without invalidating accuracy or feasibility | researcher | yes | — | `configs/base_model_candidates_amendment_4f.yaml` | no |
| `base_model_evaluation.decision.require_complete_candidate_measurements` | global | boolean | `true` | `true`, `false`; Amendment 4c requires `true`, forcing no selection unless every non-reference candidate completed or was rejected by its measured footprint | researcher | yes | — | `configs/base_model_candidates_amendment_4c.yaml` | no |
| `base_model_evaluation.execution.output_dir` | environment | path | `experiments/base_model_candidates/amendment-4g` | writable directory inside the worktree; prior amendment evidence remains immutable | researcher | yes | — | `configs/base_model_candidates_amendment_4g.yaml` | no |
| `base_model_evaluation.execution.resume` | environment | boolean | `true` | `true`, `false`; only matching completed records are skipped | researcher | yes | — | `configs/base_model_candidates.yaml` | no |
| `base_model_evaluation.execution.allow_host_restart` | environment | boolean | `false` | `true`, `false`; when true, an ephemeral hostname is excluded from manifest identity so a preempted worker can resume identical records | researcher | yes | — | `configs/base_model_candidates_amendment_4d.yaml` | no |
| `vertex_base_sweep.project` | environment | string | `aptivra-training` | non-empty existing project ID; runner is render-only and never creates a project | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | no |
| `vertex_base_sweep.region` | environment | string | `us-central1` | non-empty Vertex region with the approved spot accelerator quota | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | no |
| `vertex_base_sweep.display_name_prefix` | environment | string | `aptivra-base-sweep-4d` | non-empty Vertex-compatible prefix | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | no |
| `vertex_base_sweep.package_uri` | environment | GCS URI | preregistered bucket package path | `gs://` only; immutable source tarball uploaded by reviewer | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | no |
| `vertex_base_sweep.artifact_uri` | environment | GCS URI | preregistered bucket evidence path | `gs://` only; distinct smoke/full suffixes are added by the runner | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | no |
| `vertex_base_sweep.executor_image_uri` | environment | digest-only image URI | `docker.io/pytorch/pytorch@sha256:1262abc856a428933c0477b4be87079173a0b087b7691c47fcb396b7b5775c56` | repository plus direct Linux/amd64 `sha256:` digest; tags and tag+digest forms are rejected | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.executor_image_tag` | environment | image tag | `2.14.1-cuda12.6-cudnn9-runtime` | provenance/documentation only; never emitted in `containerSpec.imageUri` | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | no |
| `vertex_base_sweep.python_executable` | environment | absolute path | `/usr/bin/python3` | absolute `sys.executable` path proven by the pinned container smoke; never resolves through a host launcher | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.expected_python_version` | environment | major.minor version | `3.12` | dotted numeric major/minor; checked before package/model download and again in the worker | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.expected_torch_version` | environment | semantic version | `2.14.1` | exact base version, ignoring only a local `+cu...` suffix; must match the pinned image | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.image_cuda_version` | environment | toolkit version | `12.6.3` | dotted numeric version recorded from immutable registry config history | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.expected_torch_cuda_version` | environment | CUDA ABI version | `12.6` | exact `torch.version.cuda` major/minor expected at runtime | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.expected_cuda_available` | environment | boolean | `true` | `true`, `false`; paid GPU jobs require `true` and fail before model download otherwise | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.pip_timeout_seconds` | environment | integer | `60` | 1–300 seconds per package-index request | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.pip_retries` | environment | integer | `4` | 0–10 bounded retries per package-index request | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.runtime_dependencies` | environment | exact package pins | Python 3.12 non-Torch closure from `uv.lock` | non-empty unique `name==version` list; Torch and NVIDIA packages forbidden so pip cannot replace the image CUDA build | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.replica_count` | environment | integer | `1` | exactly 1 | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.boot_disk_type` | environment | string | `pd-standard` | non-empty Vertex-supported disk type | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.boot_disk_size_gb` | environment | integer | `100` | 100–64000 GiB | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.scheduling_strategy` | environment | enum | `SPOT` | exactly `SPOT` for Amendment 4d | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.restart_job_on_worker_restart` | environment | boolean | `true` | must remain `true` for spot recovery | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.disable_retries` | environment | boolean | `false` | must remain `false` for spot recovery | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.full` | environment | machine object | `a2-highgpu-1g`, one `NVIDIA_TESLA_A100`, 10800 s | exact approved machine/accelerator/count; timeout 60–10800 s | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.smoke` | environment | machine object | `n1-standard-4`, one `NVIDIA_TESLA_T4`, 1200 s | one accelerator; timeout 60–1200 s | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |
| `vertex_base_sweep.storage.timeout_seconds` | environment | float | `30.0` | 1.0–300.0 seconds per GCS request | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | no |
| `vertex_base_sweep.storage.retries` | environment | integer | `4` | 1–10 attempts with bounded exponential backoff | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | no |
| `vertex_base_sweep.storage.retry_backoff_seconds` | environment | float | `1.0` | 0.0–60.0 seconds | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | no |
| `vertex_base_sweep.storage.artifact_retention_days` | environment | integer | `30` | 1–3650 days; bucket lifecycle file must match | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml`, `configs/gcs_lifecycle_30d.json` | no |
| `vertex_base_sweep.labels` | environment | string map | workload, protocol, cost center | non-empty string keys and values accepted by Vertex | cloud operator | yes | dry-run-only runner | `configs/vertex_base_sweep.yaml` | yes — new job |

## Column meanings

- **Scope** — global, per-environment, per-tenant, per-user, or per-service.
- **Safe limits** — the range/enum the UI and API both enforce. "None" is not
  an acceptable value; state the guardrail.
- **RBAC role** — the role permitted to write it. Read access is stated
  separately only when it differs.
- **Env-specific** — whether the value is scoped per environment and promoted
  explicitly between them.
- **Feature flag** — the flag gating the capability, if it is behind phased
  rollout.
- **Requires restart** — `no` is the expected answer. `yes` requires a note
  explaining why hot-apply is not technically feasible, surfaced in the UI
  before the change is applied.

## Excluded by the scope test

Values that alter the system's contract, data model, or security boundary are
**core platform changes**, not configuration. Record each such decision as an
ADR and link it below.

| Value | Why it is not configurable | ADR |
|---|---|---|
| Deep Learning Framework & Compute Abstraction | Framework engine choice (PyTorch 2.x) defines core runtime model and execution semantics | [ADR-0001](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0001-framework-and-device-selection.md) |
| Backbone Architecture & Attention Isolation Contract | Structural contract of decoder transformer and T0 attention interface boundary | [ADR-0002](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0002-model-size-and-backbone-architecture.md) |
| Expert Block Topology & Router Freezing Mechanics | Fundamental expert module parameter isolation and router parameter freezing rules | [ADR-0003](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0003-expert-block-design-and-router-architecture.md) |
| Episodic Store Write Gate Policy | Hard invariant requiring 100% verifier correctness ($v\_result = 1$) for episodic admission | [ADR-0004](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0004-episodic-store-retrieval-method.md) |
| Statistical Test for Promotion Gate | Mathematical decision boundary ($\alpha = 0.05$ paired statistical test) for model promotion | [ADR-0005](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0005-statistical-test-for-promotion-gate.md) |
| Tiered Memory Runtime & Pure C Engine | Architecture for NVMe expert streaming, usage-heat expert pinning, and pure C runtime export | [ADR-0006](file:///Users/raghunath/code/personal/Aptivra/docs/adr/0006-multitier-nvme-expert-tiering-and-c-engine.md) |
| Generated-code execution containment | Child-process execution, timeout, and process-group termination semantics; explicitly not a security boundary | [ADR-0007](adr/0007-generated-code-execution-containment.md) |
| Pretrained Tier 3 base and tokenizer contract | Changes the foundational backbone identity and replaces byte-token stop assumptions with decoded-string plus EOS semantics | [ADR-0008](adr/0008-adopt-pretrained-base-model.md) |
