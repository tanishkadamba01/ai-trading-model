# Changelog

## v0.2.0 - 2026-02-28

### Added
- Unified backtest flow in `backtest.py` for both non-leverage and leverage trade logs.
- `run_full_workflow.py` to execute the complete pipeline in one command.
- `requirements.txt` as the standard dependency file (`requirments.txt` kept for compatibility).

### Changed
- `trade_simulation_leverage.py` now models leverage using capital, notional sizing, and fee impact in USD.
- `labeling.py` now accepts TP percentage from CLI to stay aligned with sweeps.
- `run_parameter_sweep.py` refactored with CLI options and cleaner execution logs.
- `README.md` rewritten for a publish-ready v0.2 release.

### Fixed
- `paper_trade.py` import/runtime issues and model path consistency.
- `open_data.py` hard-coded local path issue.
- Multiple script encoding artifacts and non-ASCII output noise.
