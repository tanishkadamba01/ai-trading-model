# Release Notes - v0.2.0

## Highlights

- Unified backtest flow in `backtest.py` for non-leverage and leverage trade logs.
- Realistic leverage simulation with capital/notional sizing and fee impact.
- Added one-command end-to-end runner: `run_full_workflow.py`.
- Refactored parameter sweep runner with CLI options and cleaner execution flow.
- Professionalized README and release changelog.

## Improvements

- Added standard `requirements.txt` (kept `requirments.txt` compatibility copy).
- Added `CHANGELOG.md` for version tracking.
- Improved script robustness and CLI consistency across pipeline files.
- Fixed utility script pathing and import/runtime issues.

## Validation

- Full AST parse across project Python files passed.
- Full workflow smoke run passed.
- Parameter sweep smoke run passed.

## Upgrade Notes

- Preferred install command is now:

```bash
pip install -r requirements.txt
```

- Use the full workflow runner for reproducible runs:

```bash
python run_full_workflow.py --tp 0.0023 --prob 0.65 --leverage 3.0 --initial-capital 1000 --capital-fraction 1.0
```

## Version Tag

`v0.2.0`
