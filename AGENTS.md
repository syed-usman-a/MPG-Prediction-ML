# Repository Guidelines

## Project Structure & Module Organization

This is a small Python project exploring MPG prediction from vehicle weight using the UCI Auto MPG dataset.

- `load_data.py` defines dataset paths, loads weight and MPG columns, and rejects nonfinite values.
- `linreg.py` provides `predict` and `mse`, plus a runnable demonstration with a manually chosen slope and intercept.
- `hello.py` checks imports and prints library versions.
- `data/` contains the dataset, its original version, and documentation.
- `requirements.txt` pins dependencies. No dedicated test directory or build configuration currently exists.

## Build, Test, and Development Commands

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe hello.py
.\.venv\Scripts\python.exe load_data.py
.\.venv\Scripts\python.exe linreg.py
```

These create an isolated environment, install dependencies, check imports, inspect dataset ranges and shapes, and run predictions with mean squared error. There is no separate build step.

## Coding Style & Naming Conventions

Use four spaces for indentation, `snake_case` for functions and variables, and `UPPER_SNAKE_CASE` for constants such as `DATA_PATH`. Follow PEP 8 spacing and group standard-library imports before third-party and local imports. Add type hints and concise docstrings for reusable functions, documenting array shapes, units, return values, and exceptions. Keep runnable demonstrations inside `if __name__ == "__main__":`. No formatter or linter is configured.

## Testing Guidelines

No automated test framework or coverage threshold is configured. Run the three scripts above as smoke checks after relevant changes. Verify prediction arithmetic, MSE values, mismatched-shape rejection, and loader handling of missing files and nonfinite inputs when changing those behaviors. If adding automated tests, use `tests/test_*.py`, document the runner command, and declare any new test dependency.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as `Add predict function` and `Add mean square error`; no formal prefix convention is established. Keep commits focused. Pull requests should describe the behavior changed, explain the motivation, link relevant issues, and list validation commands and results. Include numerical before/after examples for prediction changes.

## Data & Environment Hygiene

Preserve dataset files unless a change explicitly requires updating them. Resolve data paths relative to the module, following `DATA_PATH`. Keep `.venv/`, Python caches, and local credentials out of Git.
