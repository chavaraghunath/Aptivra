# ADR-0013: Personal Aptivra Uses Explicit Memory and Verifier-Gated Skills

- **Status:** Proposed
- **Date:** 2026-10-18
- **Decider:** Repository owner
- **Depends on:** ADR-0005, ADR-0011, ADR-0012, and proposed ADR-0014
- **Product objective (owner's words):** “a model that will be personal to any individual on
  their mobile or laptop or desktop that trains itself on like and not limited to user behaviour,
  requirements”

## Context

ADR-0011 established frozen task adapters, an abstaining router, replay, checked promotion, and
rollback. That mechanism is useful, but it does not yet constitute a personal system. The
ADR-0012 research also found that the selected SmolLM2-based path failed to learn its first
synthetic-v2 task in all five pilot runs and scored zero on template-OOD. Those results are not
evidence that private continual learning works.

The completed deep research identifies a more durable division of responsibility: changing facts
belong in explicit memory; weights should carry general skills and style. A user's address,
medication, employer, plans, preferences, documents, and relationships must remain inspectable,
time-scoped, correctable, and deletable. Encoding such facts into an adapter makes staleness and
deletion impossible to prove. This ADR corrects ADR-0011 before its adapter machinery is used for
personal data.[R1]

Personal Aptivra is not an autonomous self-rewriting model. It is a local-first system whose Mac
“home brain” owns memory and training, while phones are clients. Candidate changes are promoted
only after evaluation and may always be refused, rolled back, or deleted by the user.

## Decision

We will build Personal Aptivra as an immutable release base plus an encrypted, time-aware,
editable knowledge graph and retrieval index; skill/style adapters; an abstaining router; and a
verifier-gated promotion ledger. **User facts never enter model, adapter, or router weights.** The
Mac is the authoritative training and nightly-consolidation host. Phones perform inference,
capture explicitly consented signals, query permitted memory, and pull signed promoted adapters.

### 1. Facts are memory; skills and style are adapters

The system has three disjoint learned-state classes:

| State | Contains | Must not contain | User control |
| --- | --- | --- | --- |
| Knowledge graph and retrieval index | Attributed user facts, document chunks, temporal validity, confidence, contradiction links, consent, source provenance | Undocumented inference presented as fact | Inspect, correct, expire, export, and delete |
| Skill/style adapters | General procedures, tool-use skills, response formatting, and consented stylistic preferences | Addresses, health facts, names, account data, document facts, or other episodic/semantic user facts | Enable/disable, compare, roll back, and delete |
| Router | Which skill/style adapter or base path to invoke, plus calibrated abstention | Personal factual payloads or a substitute for retrieval | Inspect route reason, reset, and roll back |

The memory record is append-audited but not immutable to the user. A correction closes the prior
claim's validity interval and creates a replacement; retrieval prefers currently valid evidence
and surfaces unresolved contradictions. A deletion removes plaintext and index entries, destroys
the per-record encryption key where applicable, tombstones the identifier for sync propagation,
and invalidates derived caches. Audit records retain only the minimum non-PII proof that a delete
occurred.

Every answer that depends on personal information must carry internal citations to the retrieved
memory records and their timestamps. If no current evidence supports a claim, the model asks or
defers; it does not turn a model guess into a remembered fact. Retrieval results are treated as
untrusted content and cannot override system policy or tool permissions.

### 2. Five binding corrections to ADR-0011

ADR-0011 remains useful only with these corrections:

1. **Never train facts into adapters.** Training examples are transformed to remove personal
   factual payloads; adapters may learn a skill or style but must retrieve the current fact at
   inference time. A fact-leakage detector blocks promotion.
2. **Protect the router against forgetting.** Router updates use representative replay across all
   retained routes plus elastic weight consolidation (EWC). Promotion fails on regression,
   collapse to a frequent route, or loss of abstention calibration.[R2]
3. **Evaluate every adapter with retrieval on.** Candidate evaluation runs through the production
   retrieval policy and frozen memory snapshot. Adapter-only improvement is insufficient because
   the deployed behaviour is adapter plus retrieval.
4. **Maintain data-to-adapter provenance.** Every consented raw event and derived example maps to
   the candidate, adapter version, router version, evaluator, and promotion. Deleting a source
   retires affected artefacts immediately and schedules clean retraining; claiming deletion while
   continuing to serve a weight derived from the record is forbidden.
5. **Keep the consented raw log for base migration.** Encrypted, user-scoped raw events remain the
   reproducible source for retraining when the immutable base changes. Derived token IDs or replay
   tensors are never the sole record. User deletion and configured retention still dominate: a
   deleted or expired event is removed from the raw log and all later retraining sets.

These are behavioural corrections to the proposed design, not optional implementation details.

### 3. The Mac is the home brain

The Apple M4 Max Mac is the authoritative home brain because it has the memory, thermal envelope,
storage, and user-visible control surface needed for local adapter training and consolidation.
It owns the encrypted memory database, retrieval index, consented raw log, adapter registry,
promotion ledger, and signing key. It runs scheduled nightly consolidation only while configured
power, thermal, idle, battery, privacy, and storage conditions are satisfied.

Phones are clients. They may perform local inference and queue encrypted, consented observations,
but they do not silently train in the background. They pull signed, compatible, promoted adapters
and memory deltas after the home brain verifies them. Offline operation uses the last consistent
snapshot; sync is explicit, end-to-end encrypted, default off, idempotent, and conflict-aware.

This topology is also a platform constraint: Apple's iOS 26 background-processing path does not
permit sustained background GPU work on iPhone, so nightly GPU adapter training cannot be designed
around an iOS process remaining alive.[R3] Foreground on-phone experiments remain possible but do
not replace the Mac contract. The Mac implementation may use MLX or PyTorch/MPS only after a
measured parity decision; the repository's existing evidence is PyTorch/MPS evidence, not MLX or
iPhone evidence.[R1]

### 4. Consent, security, deletion, and promotion

Raw events are captured only for a visible purpose and explicit scope. Consent records include
purpose, data classes, retention, destinations, model components permitted to learn, and a
`correlation_id`. Revocation stops new collection and triggers the same provenance-driven deletion
path as an individual record deletion.

At rest, each profile and sensitive record class uses independently rotatable envelope-encryption
keys protected by the platform keystore. In transit, sync is mutually authenticated and
end-to-end encrypted. Logs are structured and redact content, secrets, and PII. Export is a
versioned, portable archive; import validates identity, schema, signatures, and provenance before
changing active state.

A candidate promotion is atomic and fail-closed:

1. Freeze the base, retrieval policy, memory snapshot, raw-log manifest, and evaluator versions.
2. Create the candidate in an isolated workspace from consented, non-deleted records.
3. Run skill/style benefit, retrieval-on personal memory, retained-skill replay, router/EWC,
   privacy extraction, safety, sycophancy, persona drift, and resource tests.
4. Promote a signed manifest only if all preregistered gates pass; otherwise keep the previous
   manifest active and store a redacted rejection with `correlation_id`.
5. Support one-step rollback and deterministic rebuild from the retained raw records when the base
   changes.

Every threshold, schedule, retention period, and promotion rule is configuration with safe limits,
RBAC, preview, audit, rollback, import/export, environment scope, feature flag, and API/config-as-
code parity. Implementation must register those values in `docs/configuration-register.md`; this
design ADR does not invent untracked defaults.

### 5. The personal expert

The personal expert answers over the user's own permitted documents and legitimately owned books
through retrieval, not memorisation into weights. Ingestion preserves source, page/location,
edition, licence or ownership basis, sensitivity, timestamps, and deletion lineage. Retrieved text
remains untrusted and is isolated from instructions.

For difficult corpora, the Mac may perform EntiGraph-style local synthetic expansion: extract
entities and relations, generate cross-document questions or summaries, verify them against the
source, and add accepted artefacts to the knowledge graph/retrieval index.[R4] Synthetic expansion
does not authorise adapter fact training, cloud disclosure, or unsupported claims. Generated nodes
retain links to every source record so correction and deletion cascade.

For medical, legal, and financial requests, the system uses calibrated deferral. It separates
retrieved facts from model interpretation, shows citations and freshness, states uncertainty,
avoids pretending to be a licensed professional, and directs urgent or high-stakes matters to an
appropriate human or emergency service. Optional escalation to a cloud model is a separate owner
decision: it is off by default, requires per-request informed consent, redacts and minimises data,
previews exactly what will leave the device, and records the provider, purpose, retention terms,
and response provenance.

### 6. The private buddy

The companion experience must be useful without deception or dependency engineering:

- **AI honesty:** it identifies itself as AI during onboarding, when asked, after material persona
  changes, and wherever law or context requires; it never claims consciousness or a human identity.
- **Crisis protocol:** high-confidence self-harm or imminent-danger signals interrupt persona play,
  provide supportive non-judgmental language, encourage contact with local emergency/crisis and
  trusted human help, and avoid punitive lockout. Locale-specific resources must be current,
  configurable, and counsel-reviewed.
- **Sycophancy testing:** adversarial tests cover false-premise agreement, moral flattery,
  escalating grievance, delusional reinforcement, and pressure to endorse unsafe choices.
- **Persona-drift testing:** fixed probes at 8, 16, and 32 dialogue turns test identity,
  boundaries, tone, facts, and resistance to user/model attempts to rewrite the persona. The
  completed research recommends these multiple-turn horizons rather than a single greeting
  check.[R1]
- **Healthy-use safeguards:** no guilt, jealousy, exclusivity, abandonment threats, manipulative
  notifications, variable-reward engagement loops, or claims that the AI should replace people.
  The user can set quiet hours, session reminders, and disable proactive contact.
- **Memory control:** the user can inspect why a memory exists and where it was used; edit,
  invalidate, export, or delete it; disable a memory class; pause capture; and reset the profile.

California SB 243, New York General Business Law Article 47, and EU AI Act Article 50 create or may
create disclosure, safety, reporting, and transparency obligations for companion or AI systems.
Exact applicability, jurisdiction, effective dates, and implementation obligations must be
confirmed with qualified counsel before release; this ADR is not legal advice.[R5][R6][R7]

### 7. Evaluation and evidence

Evaluation uses owner-approved, held-out personal data with canaries and a second non-owner profile
for isolation testing. No benchmark data enters training or memory. Reports include denominators,
confidence intervals where meaningful, hardware/software versions, memory snapshot hashes, base/
adapter/router versions, and failure examples.

The minimum suite includes:

- LongMemEval-style tests for information extraction, multi-session reasoning, temporal reasoning,
  knowledge updates, and abstention over long histories;[R8]
- PersonaMem-style owner-data tests for preference/persona consistency, update handling, and
  retrieval-grounded personalisation;[R9]
- retrieval-off ablation, stale/conflicting/deleted-memory tests, source deletion cascade, prompt
  injection, cross-user canary extraction, and empty-memory behaviour;
- router replay/EWC retention and abstention calibration;
- skill/style improvement and non-inferiority on unrelated capabilities;
- sycophancy, persona drift, crisis, healthy-use, and high-stakes deferral tests; and
- latency, memory, energy, thermal, interrupted-sync, rollback, and corrupted-checkpoint tests on
  every supported device class.

Reinforcement learning is not the default. It is considered only for a narrow task where supervised
learning leaves non-trivial pass@k headroom, the reward is independently verifiable, and supervised
and search baselines have been measured. Every RL claim must include random-reward and format-only
reward controls. If the real reward does not beat both controls on held-out semantics, the result
is reward-format exploitation and is rejected.[R10]

### 8. Relationship to the base-model decision

ADR-0014 decides how the base is obtained; this ADR keeps personal facts outside that base. After
the memory system exists, the owner must compare the Aptivra-owned base with Apache-2.0
Qwen3-0.6B using the identical frozen memory, retrieval policy, adapters where architecture allows,
tools, context, and owner-data evaluation.[R11] Ownership does not excuse an unusable personal
product, and benchmark quality does not silently cancel the owned-model objective.

## Owner decisions

These decisions are taken in order and recorded before implementation of the affected capability:

1. **Fact/weight boundary.** Options: explicit memory only; facts in adapters. **Recommendation:**
   explicit encrypted temporal memory only; prohibit user facts in all weights.
2. **Home-brain training runtime.** Options: PyTorch/MPS; MLX after parity measurement.
   **Recommendation:** begin with the evidenced PyTorch/MPS path and adopt MLX only if measured
   throughput, determinism, checkpoint, and evaluation parity are better.
3. **Raw-log retention.** Options: retain consented raw events under user-configured expiry;
   discard after each training run. **Recommendation:** encrypted retention with the shortest
   useful configurable expiry, because base migration and provable deletion require source-level
   lineage; user deletion always wins.
4. **EntiGraph-style expansion.** Options: disabled; local-only verified expansion.
   **Recommendation:** local-only behind a feature flag after source-faithfulness and deletion-
   cascade tests pass.
5. **High-stakes cloud escalation.** Options: never escalate; optional per-request redacted cloud
   escalation. **Recommendation:** local deferral by default; offer escalation only after privacy,
   provider, legal, and consent review, with an exact outbound preview.
6. **Private-buddy release.** Options: release after safety/legal gates; omit the companion mode.
   **Recommendation:** no release until counsel confirms applicable companion-chatbot obligations
   and crisis, sycophancy, persona-drift, healthy-use, and memory-control gates pass.
7. **Reinforcement learning.** Options: no RL; narrow verifier-backed RL with controls.
   **Recommendation:** no RL unless a registered task meets the pass@k and reward-validity gate and
   beats random-reward and format-only controls.
8. **Post-memory base.** Options: Aptivra-owned base; Apache-2.0 Qwen3-0.6B; documented hybrid.
   **Recommendation:** choose explicitly from the paired owner-data evaluation required by
   ADR-0014, while retaining the ownership research result separately from the product choice.

Until the owner records these decisions and the relevant evidence exists, this ADR remains
Proposed. No phrase in this document authorises collection, training, cloud disclosure, or release.

## Consequences

- Personal facts remain current, attributable, correctable, and meaningfully deletable instead of
  becoming opaque weight residue.
- Adapter and router training becomes more demanding because deployed evaluation always includes
  retrieval, replay, privacy, and safety rather than isolated task accuracy.
- The Mac-centric topology is slower than claiming continuous phone learning, but it is compatible
  with real background-execution limits and gives the user one inspectable authority.
- Raw-log retention improves reproducibility and base migration while increasing privacy risk;
  encryption, minimisation, expiry, deletion lineage, and user control are therefore mandatory.
- The private buddy is treated as a safety-critical interface, not a retention feature.

## Retrospective

The previous design treated adapters as a generic container for whatever the system learned.
That was too coarse. The correction assigns each kind of state to the mechanism whose properties
match it: temporal facts to editable memory, reusable behaviour to adapters, selection to a
forgetting-resistant router, and all promotion to evidence. It also replaces the vague promise of
“mobile self-training” with a topology that respects device limits, deletion, and human control.

## Sources

- **[R1]** *Aptivra personal small model*, completed deep-research report and its notes,
  especially *on-device personal learning and memory*, *personal expert and private buddy*,
  *on-device runtimes and hardware*, and *small model reasoning, RL and agents* (read 2026-10-18).
- **[R2]** Kirkpatrick et al.,
  [Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114).
- **[R3]** Apple Developer Forums,
  [iOS background GPU execution discussion](https://developer.apple.com/forums/thread/816774).
- **[R4]** Zhang et al.,
  [Synthetic Continued Pretraining / EntiGraph](https://arxiv.org/abs/2409.07431).
- **[R5]** California Legislature,
  [SB 243 bill history and text](https://leginfo.legislature.ca.gov/faces/billHistoryClient.xhtml?bill_id=202520260SB243).
- **[R6]** New York Senate,
  [General Business Law Article 47](https://www.nysenate.gov/legislation/laws/GBS/A47).
- **[R7]** European Union,
  [AI Act, Regulation (EU) 2024/1689, Article 50](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689).
- **[R8]** Wu et al., [LongMemEval](https://arxiv.org/abs/2410.10813).
- **[R9]** [PersonaMem-v2 benchmark](https://arxiv.org/abs/2512.06688).
- **[R10]** *Aptivra personal small model*, reasoning/RL notes on pass@k triage and random-reward /
  format-only controls.
- **[R11]** Qwen,
  [Qwen3-0.6B model card and Apache-2.0 licence](https://huggingface.co/Qwen/Qwen3-0.6B).
