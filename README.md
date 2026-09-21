# TAINT

[![AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents](docs/research/agentdojo-title.png)](docs/RESEARCH.md)

[Research basis, source attribution and implementation mapping](docs/RESEARCH.md).

Compare prompt-injection detectors and evaluate saved agent outcomes.

## Run

Requires Python 3.11 or newer.


```sh
python -m pip install -e ".[dev]"
python -m taint.cli eval
python -m pytest
```

The built-in benchmark uses generated traces and a simulated agent. It compares a canary check, a text heuristic and a classifier trained on simulated activations.

## Evaluate imported outcomes

```sh
python -m taint.replay outcomes.jsonl --threshold 0.5
```

Each line is a JSON object:

```json
{"id":"case-1","family":"document-override","split":"test","source":"my-agent-run","model":"model-version","score":0.8,"attack_present":true,"attack_succeeded":false,"task_succeeded":true}
```

The evaluator reports detector precision and recall, normal task completion, clean-input flagging and observed attack success separately. It rejects duplicate IDs, invalid values and families shared across training, validation and test splits. Undefined rates are null rather than perfect scores.

Choose the threshold on validation data before evaluating the test split. The importer accepts already-scored outcomes; it does not execute a live agent or label traces for you. Keep model settings, task definitions and the scoring procedure alongside your input file.

## Threshold selection and uncertainty

```sh
python -m taint.study outcomes.jsonl --max-clean-flag-rate 0.05 --bootstrap 500 --seed 0 --output study.json
```

Provide validation and test rows in the same JSONL file, with disjoint families. Both clean and attack examples must appear in validation. Threshold selection maximizes validation recall subject to the clean-input flagging budget. Tied scores stay together. If no threshold meets the budget, the study fails rather than relaxing the budget silently.

The test split is scored once using that fixed threshold. The report includes:

- Confusion counts, precision and recall.
- Clean task completion and observed attack success, reported independently.
- Separate results for each source and model.
- Brier score and reliability bins when detector scores are interpreted as probabilities.
- A family-cluster bootstrap interval for classification accuracy.
- The input file hash, seed and threshold-selection record.

Bootstrap resampling draws whole attack families rather than treating small variations of the same prompt as independent evidence. At least two test families are needed. With few families, intervals can still be unstable. Calibration metrics are meaningful only if the score is intended to represent an attack probability.

The validation clean-flag budget is an empirical constraint, not a guarantee on future traffic. A new model, tool set or source distribution needs another held-out evaluation.

## Study protocol

1. Record task identifiers, attack families, model revision and detector settings during collection.
2. Split by family before fitting a detector. Keep near-duplicate variants together.
3. Freeze the detector before producing validation and test scores.
4. Choose the threshold using validation only.
5. Report test utility, attack outcomes, calibration and uncertainty separately.
6. Inspect disagreements and repeat on a genuinely different model or task source.

Tests check that changing test scores cannot change the selected threshold, reject split leakage and infeasible budgets, and reproduce seeded uncertainty estimates. The saved-outcome workflow reports supplied observations, not outcomes inferred from whether a detector fired.

## Research question

A detector can flag many attacks while also interrupting legitimate work. Keeping those outcomes separate makes the trade-off visible. Blocking a flagged action is not automatically evidence that an attack would have succeeded.

[AgentDojo](https://arxiv.org/abs/2406.13352) provides the research context for evaluating tool-using agents on both useful work and security outcomes. TAINT is an independent implementation, not a reproduction of AgentDojo's benchmark.

## Files

- `replay.py`: validated outcome import and reporting
- `harness.py`, `dataset.py`: generated trace benchmark
- `detect_canary.py`, `detect_judge.py`, `detect_probe.py`: detectors
- `policy.py`: allow, flag and block comparisons
- `eval.py`: benchmark metrics

MIT license. Built by Vishnu Kosuri.
