# TAINT

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

## Design

A detector can flag many attacks while also interrupting legitimate work. Keeping those outcomes separate makes the trade-off visible. Blocking a flagged action is not automatically evidence that an attack would have succeeded.

[AgentDojo](https://arxiv.org/abs/2406.13352) provides the research context for evaluating tool-using agents on both useful work and security outcomes. TAINT is an independent implementation, not a reproduction of AgentDojo's benchmark.

## Files

- `replay.py`: validated outcome import and reporting
- `harness.py`, `dataset.py`: generated trace benchmark
- `detect_canary.py`, `detect_judge.py`, `detect_probe.py`: detectors
- `policy.py`: allow, flag and block comparisons
- `eval.py`: benchmark metrics

MIT license. Built by Vishnu Kosuri.
