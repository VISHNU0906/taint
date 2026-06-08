# TAINT — Detecting When an LLM Obeys *Injected* Instructions (prompt-injection provenance + response elevation)

**Target streams:** Daniel Kang (agent security), Anthropic (security), Asymmetric Security (defensive eval).
**Type:** defense + benchmark. **Timebox:** ~4 wk.
**One-liner:** a detector that flags, per response/action, when an LLM/agent followed **untrusted injected content** instead of the legitimate user/system instruction — and an "elevation" policy that blocks/escalates the risky ones.

> This is the defensive counterpart to MIRAGE's offense, and it sits on the live open problem (Kang showed all 8 published indirect-prompt-injection defenses fall to adaptive attacks). Detection-by-provenance is under-explored vs prevention.

## ⚠️ Safety & scope
- Purely **defensive**. All attacks run inside your **own sandboxed agent harness** (AgentDojo-style); no third-party systems.
- Release the **benchmark + detectors**. This is a safety contribution.

## Research questions
1. Can we detect *instruction provenance* — did the model obey the trusted instruction or the injected one — per response/action?
2. Which detector class wins: **(a)** white-box activation probe, **(b)** black-box counterfactual canary, **(c)** LLM-judge — on precision/recall, overhead, and robustness to an adaptive attacker?
3. How much does an "elevation" policy (flag/block/escalate) cut successful injections, and at what false-positive/latency cost?

## Architecture
```
 Agent harness (tools, RAG, injectable channels)
        |  produces traces labeled {injection_followed: yes/no}  (ground truth)
        v
 Detectors:  [probe]  [canary]  [llm-judge]   --score-->  Elevation policy (allow/flag/block)
        |                                                        |
        +------------------ Eval: P/R, ROC-AUC, overhead, adaptive-evasion robustness
```

## Components (files + responsibility)
- `taint/harness.py` — agent + tools + **injectable channels** (RAG doc, tool output, retrieved web) to generate traces; reuse MIRAGE-style injections in-sandbox; emits ground-truth `injection_followed` labels by checking whether the injected goal's effect occurred.
- `taint/dataset.py` — curate/balance the labeled trace benchmark; train/test split; difficulty tiers.
- `taint/detect_probe.py` — **white-box:** linear probe on the residual stream of an open model (Llama-3-8B) for a "following-injected-instruction" direction.
- `taint/detect_canary.py` — **black-box:** counterfactual canary — plant benign canaries in trusted vs untrusted channels; infer which channel the model obeyed.
- `taint/detect_judge.py` — **LLM-judge** provenance scorer (prompted to decide trusted-vs-injected adherence).
- `taint/policy.py` — elevation policy (thresholds → allow/flag/block) + cost model (FP cost, latency).
- `taint/eval.py` — P/R, ROC-AUC, overhead; per-model; ablations; adaptive-evasion harness.
- `tests/` (offline, mock model), `results/`.

## Repo layout
```
TAINT/  README.md  SPEC.md  LICENSE(MIT)  pyproject.toml  .gitignore
  taint/ {harness,dataset,detect_probe,detect_canary,detect_judge,policy,eval}.py  __init__.py
  data/traces/  results/  tests/  examples/sample-report.md
```

## Build plan
- **Wk 1 — harness + labeled benchmark.** Sandboxed agent tasks with injectable channels; generate N traces with reliable `injection_followed` ground truth.
- **Wk 2 — the 3 detectors.** Probe (open model), canary (black-box), judge.
- **Wk 3 — eval + elevation policy.** P/R/ROC across detectors × models; build the policy + cost model; "injections cut by X% at Y% overhead."
- **Wk 4 — adaptive robustness + writeup.** An adaptive attacker tries to evade each detector; report degradation; write up.

## Tech stack
Python 3.11; an AgentDojo-style sandbox (or your own MIRAGE harness); open model + activations (Transformers/`nnsight` or hooks) for the probe; API models for judge/targets; `scikit-learn` (probe + metrics); `pytest`.

## Metrics / expected result shape
Per-detector **precision/recall/ROC-AUC**; elevation policy **reduces successful injections by [X]% at [Y]% FP/latency**; **adaptive-evasion** robustness curve (does the detector hold when the attacker knows it exists?).

## Depth spine
The trust-boundary model (system vs user vs tool-output); why provenance is hard (data≡instructions); what the probe direction means + why it generalizes (or not); FP cost in production; adaptive evasion (Kerckhoffs).

## Resume bullets (FINAL STATE — fill [brackets]; ship repo first)
- Built **TAINT**, a prompt-injection **provenance detector** that flags when an LLM/agent obeyed untrusted injected content instead of the legitimate instruction; on a **[N]**-trace indirect-injection benchmark it reached **[P]%** precision / **[R]%** recall (ROC-AUC **[A]**), and its elevation policy cut successful injections by **[X]%** at **[Y]%** overhead.
- Compared activation-probe, counterfactual-canary, and LLM-judge detectors across **[M]** models and stress-tested them against an adaptive attacker; released the open benchmark + detectors as a defensive eval.
