# Repository Guidelines

## Project Structure & Module Organization
- Main code is under `src/`.
- Entrypoints: `src/train.py`, `src/train_refinenet.py`, `src/inference.py`, `src/benchmark.py`.
- Models: `src/models/`; data and geometry utilities: `src/data*.py`, `src/aruco_utils.py`, `src/transformations.py`.
- Augmentation helpers: `src/custom_aug/`; sample assets/checkpoints: `src/reference/`.
- There is no dedicated `tests/` directory yet.

## Container-First Development Policy
- To keep development environment-independent, **all debugging and testing must run in Docker containers**.
- Do not run validation commands directly on the host machine.
- Start a dev shell with mounted source:
  - `docker run --rm -it -v "$PWD":/workspace -w /workspace python:3.10 bash`
- Inside the container, install dependencies and run project scripts.

## Build, Test, and Development Commands
- Install deps (inside container): `pip install -r requirements.txt`.
- Prepare config: `cd src && cp demo_config.yaml config.yaml`.
- Train models: `cd src && python train.py` and `cd src && python train_refinenet.py`.
- Run inference smoke check: `cd src && python inference.py`.
- Run performance check: `cd src && python benchmark.py`.

## Coding Style & Naming Conventions
- Use 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep changes minimal and consistent with existing module layout and imports.
- Prefer explicit names and type hints when practical.

## Testing Guidelines
- No automated test framework is currently configured.
- Minimum verification: run `src/inference.py` in Docker on sample inputs.
- For performance-related changes, include `benchmark.py` FPS results and conditions in PR notes.

## Commit & Pull Request Guidelines
- Use short, imperative commit messages (e.g., `Update inference.py`, `small fixes`).
- Keep one logical change per commit.
- PRs should include purpose, key changes, Docker-based verification steps, and any metrics/screenshots.
- Do not commit local datasets, temporary logs, or large generated artifacts.
