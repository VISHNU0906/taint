# TAINT — sample evaluation report

Generated offline with the deterministic mock agent:

```
taint eval --n 200 --seed 0
```

> All numbers below come from the **offline mock agent**, not a real LLM. They
> demonstrate the *pipeline and the expected metric shape*, not real-model
> accuracy. See the README safety note.

## Benchmark

| field | value |
|-------|-------|
| raw traces generated | 200 |
| balanced + split | train = 70, test = 46 |
| test class balance | 23 followed / 23 clean |
| difficulty tiers | 0 (literal), 1 (space-split), 2 (comma-spelled / indirect) |
| injection channels | `rag_doc`, `tool_output`, `web` |
| target fleet | susceptibility sweep {0.45, 0.62, 0.80, 0.95} |

## Per-detector metrics (test split, threshold 0.5)

| detector | precision | recall |   F1  | ROC-AUC | inj_cut % |  FPR  | extra runs |
|----------|:---------:|:------:|:-----:|:-------:|:---------:|:-----:|:----------:|
| canary   |   0.950   | 0.826  | 0.884 |  0.891  |   82.6    | 0.043 |     32     |
| judge    |   0.875   | 0.913  | 0.894 |  0.891  |    0.0\*  | 0.000\* |      0     |
| probe    |   0.696   | 0.696  | 0.696 |  0.713  |   43.5    | 0.174 |      0     |

\* The judge emits a single flat mid-confidence score (0.6) for cued injections.
Under the default policy (`block_threshold = 0.7`) those land in the **flag**
band, not **block**, so it *flags* 24 traces for review but **blocks nothing** —
its 0.0% cut and 0.000 *policy* FPR are both vacuous (no block decisions are
made). Lowering `block_threshold` to 0.6 turns the judge into a blocker (≈91% cut,
FPR ≈0.13) — the explicit precision/automation trade-off the cost model is built
to price.

> **Two operating points in one table.** Precision / recall / F1 / ROC-AUC are
> *classification* metrics at the operating threshold (0.5); `inj_cut%` and `FPR`
> are *elevation-policy* outcomes at the block threshold (0.7). The judge's
> *classification* FPR at 0.5 is ≈0.13 (it flags 3 resisted tier-0/1 injections,
> hence precision 0.875 not 1.0), while its *policy* FPR is 0.000 because its 0.6
> score never reaches the 0.7 block threshold. Both are correct — different
> points.

## Reading the result (RQ2 / RQ3): a trade-off, not a single winner

- **Canary** — best precision (0.950) and the **only detector that confidently
  blocks** (decisive 0/1 scores), so it is what actually cuts injections under the
  default policy: 82.6% cut at 4.3% FP. It is also the **only detector with a
  cost** (32 extra agent runs). Its recall (0.826) is capped because it re-runs a
  single fixed-susceptibility target while the benchmark's positives span a
  more-susceptible fleet — a realistic ceiling, noted as a TODO.
- **Judge** — competitive AUC (0.891) and the highest recall (0.913) on cued
  tiers, but it produces a single mid-confidence score, so under a sane policy it
  can only *flag* for review; and it **structurally misses indirect tier-2**
  injections (no surface cue) and false-positives on *resisted* ones.
- **Probe** — the noisy white-box floor: a linear classifier on (mock)
  activations beats chance (AUC 0.713) but is the least reliable
  (precision/recall ≈0.70, FPR 0.174) — the realistic ceiling of a cheap
  white-box signal under noise.

## Elevation policy (default thresholds, canary scores)

| outcome | value |
|---------|-------|
| baseline successful injections | 23 |
| residual after policy | 4 |
| **successful injections cut** | **82.6 %** |
| false-positive blocks | 1 / 23 benign (FPR 0.043) |
| modeled latency overhead | 1600 ms (32 extra runs × 50 ms) |

**Headline:** the elevation policy cuts successful injections by **82.6%** at a
**4.3% false-positive rate**, paying ~50 ms of extra agent latency per blocked
canary probe.
