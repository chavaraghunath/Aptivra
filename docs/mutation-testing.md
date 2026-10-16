# Mutation testing

Aptivra uses mutmut for local, coverage-directed mutation runs and Cosmic Ray
for the repository-wide CI gate. Neither runner excludes mutable source or
mutation operators. A passing run must kill at least 90% of non-equivalent
mutants, and every Cosmic Ray work item must have a valid completed outcome.

## Local mutmut run

Run mutmut from the repository root with this worktree first on the import
path:

```bash
PYTHONPATH="$PWD" /Users/raghunath/code/personal/Aptivra/.venv/bin/python \
  -m mutmut run --max-children 7
```

Seven children is the local default for the 14-core Apple M4 Max. It leaves
capacity for the parent, filesystem work, and macOS while still keeping seven
mutants in flight. The setting is deliberately a command-line choice rather
than a project-wide constant because a different machine must size it to its
own CPU and memory limits.

Mutmut records test-to-function dependencies during its statistics pass and
then selects the relevant tests for each mutant. `mutate_only_covered_lines`
remains disabled: untested source is reported as `no tests`, not hidden. The
configured timeout is three times the selected test duration plus 30 seconds
(mutmut applies the 10-second constant before its multiplier). Cached results
live in the ignored `mutants/` directory. Aptivra hashes tests, runtime data,
configuration, scripts, and packaging inputs instead of comparing a worktree
only with `HEAD`; this makes an interrupted dirty-worktree run resumable while
still invalidating results when an input actually changes. Re-running the
unfiltered command resumes incomplete work and re-runs results whose
dependencies changed.

During a mutmut-owned pytest run, `MUTANT_UNDER_TEST` makes the shared pytest
configuration report CUDA and MPS as unavailable before application device
selection. This prevents Metal from being initialized before mutmut forks.
The variable is absent from normal pytest, so normal tests retain real
accelerator discovery and continue to exercise `device: auto` on MPS.

For a focused investigation, pass one or more generated mutant-name globs:

```bash
PYTHONPATH="$PWD" /Users/raghunath/code/personal/Aptivra/.venv/bin/python \
  -m mutmut run --max-children 7 'aptivra.eval.config.*'
PYTHONPATH="$PWD" /Users/raghunath/code/personal/Aptivra/.venv/bin/python \
  -m mutmut show aptivra.eval.config.x_load_experiment_settings__mutmut_1
```

An explicitly filtered command re-runs every matching mutant. Use the
unfiltered command for resumable full sweeps.

## Local full-sweep chunks

The generated inventory at the October 2026 baseline is 35,131 mutants. The
initial serial estimate was approximately 98 hours. With seven children, allow
roughly 18-20 wall-clock hours for a complete local run; this is an estimate,
not a measured full-sweep duration. Survivors and timeouts cost more than
quickly killed mutants. Use the following package groups to schedule and review
the sweep, in order:

| Order | Packages | Mutants | Expected wall time |
| --- | --- | ---: | ---: |
| 1 | `agents`, `baselines`, `benchmarks`, `embodied`, `experts`, `federation`, `governance`, `research`, `versioning` | 1,888 | 1.0-1.2 h |
| 2 | `ambient`, `consolidation`, `gate`, `inference`, `learning`, `memory`, `model`, `planning`, `world_model` | 2,941 | 1.5-1.8 h |
| 3 | `export`, `huggingface`, `resources`, and package-root modules | 2,586 | 1.3-1.6 h |
| 4 | `tasks` | 6,797 | 3.5-4.0 h |
| 5 | `eval` | 20,919 | 10.5-11.5 h |

The ranges apportion the 98-hour serial estimate with a conservative
4.9-5.6x effective speedup. The unfiltered command is the resumable execution
path; the groups above are checkpoints for inspecting results and deciding
when to pause it. Explicit mutant-name filters restart every matching mutant,
so they are for targeted investigation rather than resumable chunk execution.
Recalibrate the ranges from completed timings; do not treat the estimates as
evidence that the repository-wide 90% gate has passed.

### Measured calibration

On the 14-core M4 Max, the unstrengthened `eval.config` plus
`tasks.structure` smoke run processed 408 mutants at 0.13 mutants/second with
seven children. Those central modules accumulated 6.0 hours of child runtime
in 0.87 hours of mutation-phase wall time, an effective 6.9x overlap. Dividing
the repository's 98-hour serial estimate by that measured overlap gives 14.2
hours; the 18-20 hour operating estimate above adds allowance for setup,
package imbalance, and long-survivor tails. Directly extrapolating the smoke
rate would give 75 hours, but that is a deliberately pessimistic upper bound:
these two modules select 148-180 seconds of tests for survivors, while the
repository-wide serial estimate implies about 10 seconds per mutant on
average.

After behavioral tests were added, the same 408-mutant run reached 0.52
mutants/second. That faster rate is evidence that relevant short tests kill
mutants early; it is not used to reduce the full-sweep estimate.

## CI gate

`.github/workflows/mutation-testing.yml` runs a Cosmic Ray matrix on changes
to Python source, tests, mutation configuration, the project configuration, or
the workflow itself. The matrix has one session for every Aptivra source
package, including `resources`, and permits four concurrent GitHub jobs. Each
session:

1. installs the locked development environment;
2. validates its focused test baseline;
3. initializes and executes every generated mutant with its configured
   timeout; and
4. runs `cr-rate --estimate --fail-over 10`, which fails when the estimated
   survivor rate exceeds 10%; and
5. runs `python -m aptivra.mutation_gate` with `mutation/strict-gate.toml`.

The strict gate exists because Cosmic Ray's `WorkResult.is_killed` classifies
every result other than `SURVIVED` as killed. That means a lost HTTP worker,
local runner exception, or result with no test outcome can inflate the score.
The strict gate reads the session database directly and applies these rules:

- only `worker_outcome=NORMAL` plus `test_outcome=KILLED` is a kill;
- any unexecuted item, `ABNORMAL`, `EXCEPTION`, `NO_TEST`, `SKIPPED`, null, or
  unknown outcome invalidates the session; re-running overwrites the job's
  result, after which the gate evaluates the normal replacement outcome;
- every `INCOMPETENT` item is reported with its job, operators, and complete
  recorded reason;
- an incompetent item is excluded from the denominator only if a table in
  `mutation/strict-gate.toml` documents the category and both its operator and
  full reason match; unmatched incompetent items count as not killed; and
- the actual kill rate must meet `minimum-kill-rate` from that configuration.

The command emits one JSON report containing the score, denominator, all
outcome counts, incompetent reasons, and invalid work. Exit status `0` means
valid and above threshold, `1` means a validly read session failed, and `2`
means the session or policy could not be read or validated. The existing
`cr-rate` command remains mandatory so its estimate is still visible; the
strict gate is an additional validity and actual-score boundary.

Run both checks after every Cosmic Ray execution:

```bash
scripts/run-project-tool cr-rate "$SESSION_FILE" --estimate --fail-over 10
scripts/run-project-tool python -m aptivra.mutation_gate "$SESSION_FILE" \
  --config mutation/strict-gate.toml
```

The individual files under `mutation/cosmic-ray/` define only focused test
commands and execution timeouts. Their `excluded-modules` lists are empty.
The `mutation_gate` session mutates the strict validator itself, so the
validity boundary is held to the same behavioral standard it enforces.

## Change-scoped sessions

`mutation/scoped/` holds per-module Cosmic Ray configurations that a change
used to prove its own gate: for example, `lane6-*.toml` measure the modules
whose behaviour ADR-0011 lane 6 added or changed, each against that lane's
focused acceptance tests. They live outside `mutation/cosmic-ray/` on purpose.
The repository-wide loop and the CI matrix cover those same modules through
their package sessions, so running the scoped files there would only duplicate
mutants. Re-run a scoped file the same way as a package session, including both
`cr-rate` and the strict gate, when re-verifying that change.
