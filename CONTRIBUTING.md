# Contributing

Contributions that improve reproducibility, add a candidate-producing grounding
adapter, or extend the registered analysis to a new dataset are welcome.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

## Contribution rules

1. Keep raw datasets, model weights, complete traces, and runtime logs out of Git.
2. Add tests for changes to candidate association, coverage, order events, or estimators.
3. Do not change a frozen configuration in place. Create a new version and record its hash.
4. Distinguish semantic correctness from perturbation-defined stability.
5. Report the output contract and probe distribution with every result.

Please open an issue before proposing a change to the primary estimand or the
registered output contract.
