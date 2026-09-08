# AGENTS.md

## Project overview

This repository is a Python research app for soccer homography and tracking experiments. The interactive UI lives in [src/soccer_homography/__init__.py](src/soccer_homography/__init__.py), with the main app bootstrapped by `main()` and a Tkinter UI built around `AppState` and various canvas/tracking modules.

Most of the project is in the package under [src/soccer_homography](src/soccer_homography), while project metadata and dependency management live in [pyproject.toml](pyproject.toml) and [README.md](README.md).

## Working conventions

- Use `uv` for environment creation and dependency installs; the repo expects Python 3.12+.
- Keep changes scoped to the real app modules instead of broad refactors unless the task clearly requires it.
- This codebase heavily uses OpenCV, Tkinter, and model-based tracking; prefer small, explicit updates around existing data flow.
- When investigating UI or tracking issues, start with [src/soccer_homography/appState.py](src/soccer_homography/appState.py) and the relevant module in [src/soccer_homography](src/soccer_homography) rather than guessing at a cross-cutting rewrite.
- Use the project script defined in [pyproject.toml](pyproject.toml) for the main app launch rather than inventing a new entry point.

## Setup

```pwsh
uv venv create .venv
.venv\Scripts\activate.ps1
uv sync
```

## Run the app

```pwsh
uv run soccer-homography
```

## Useful starting points

- [src/soccer_homography/__init__.py](src/soccer_homography/__init__.py) — main UI and app lifecycle
- [src/soccer_homography/appState.py](src/soccer_homography/appState.py) — application state and configuration objects
- [src/soccer_homography/SportsTracker.py](src/soccer_homography/SportsTracker.py) — tracking pipeline and commands
- [src/soccer_homography/dataTypes.py](src/soccer_homography/dataTypes.py) — core data structures
- [README.md](README.md) — installation and project context

## Tracking and UI workflow

For work touching frame processing, homography, detection, or Tkinter UI behavior, follow the generated workflow at [.github/instructions/tracking-ui.instructions.md](.github/instructions/tracking-ui.instructions.md). The workflow is designed to keep agents focused on the main app lifecycle in [src/soccer_homography/__init__.py](src/soccer_homography/__init__.py), the live state in [src/soccer_homography/appState.py](src/soccer_homography/appState.py), and the tracking pipeline in [src/soccer_homography/SportsTracker.py](src/soccer_homography/SportsTracker.py).

## Notes for agents

- The project is experimental and not a production service; do not assume formal tests or a standard CI workflow are present.
- Validate changes with the smallest practical command or script that exercises the affected behavior.
- If a fix touches model/tracking behavior, inspect the surrounding state and data flow before changing logic, because this app passes frame, calibration, and track state through many UI and detection components.
