# ADR-0007: Generated-Code Execution Containment

- **Status:** Accepted
- **Date:** 2026-10-14

## Context

Mechanical code-task verifiers and the research tool synthesizer execute generated Python.
Running that code in the evaluator process lets an infinite loop hang the entire evaluation and
lets accidental process exits or resource exhaustion damage the caller. A restricted builtin set
or AST filter does not make Python execution safe against deliberately hostile code.

## Decision

All generated Python execution uses the same stdlib-only child-process runner. The parent supplies
a validated timeout and memory limit, starts a new process session in a temporary working
directory with a minimal environment, and kills the child's process group when the wall-clock
timeout expires. The worker retains AST checks, a restricted builtin policy, and operating-system
resource limits where supported. Synthesized tools are stored as source and re-executed in a new
child for every invocation; generated code is never compiled or executed in the parent process.

This runner is operational containment for hangs and accidental failures during evaluation. It is
**not a security boundary against hostile code**. Deliberately untrusted or adversarial programs
require an external isolation boundary designed for that threat model, such as a locked-down
container or virtual machine with appropriate kernel, filesystem, network, and privilege controls.

## Alternatives considered

1. **Restricted globals in the parent process:** rejected because loops, exits, and resource abuse
   still affect the evaluator, and the `is_sandboxed` claim would be false.
2. **Thread-based timeouts:** rejected because Python cannot reliably terminate a runaway thread.
3. **Present the child runner as a hostile-code sandbox:** rejected because AST and builtin filters
   plus process resource limits are defense-in-depth controls, not a complete security boundary.

## Consequences

- Generated-code hangs fail within the configured timeout and trigger process-group termination.
- Crashes and validation failures become structured evaluation failures rather than parent-process
  failures.
- Inputs and return values for synthesized tools must be JSON-compatible so they can cross the
  process boundary.
- Each synthesized-tool invocation incurs child-process startup cost.
- The `is_sandboxed` field is true only after child-process validation succeeds and denotes this
  operational containment, not permission to execute hostile code.
