# AGENTS.md

# AGENTS.md

## Project overview

This repository is a Python research app for soccer homography and tracking experiments. The interactive UI lives in [src/soccer_homography/__init__.py](src/soccer_homography/__init__.py), with the main app bootstrapped by `main()` (imported from pyproject.toml) and a Tkinter UI built around `AppState` and various canvas/tracking modules.

Most of the project is in the package under [src/soccer_homography](src/soccer_homography). Key files include:
- `__init__.py` — main app lifecycle, widget creation, event handlers
- `AppState.py` — application state, calibration data, model configuration
- `SportsTracker.py` — tracking pipeline (YOLO detection + ByteTrack) with threading and command queues
- `dataTypes.py` — core data structures (Point2D, SelectionPoint, VideoData, Homography, Track, BoundingBox, etc.)

Project metadata lives in [pyproject.toml](pyproject.toml), while documentation is in [README.md](README.md). The project uses DuckDB for database persistence ([src/soccer_homography/db/](src/soccer_homography/db)).

## Working conventions

- Use `uv` for environment creation and dependency installs; the repo expects Python 3.12+.
- Keep changes scoped to the real app modules instead of broad refactors unless the task clearly requires it.
- This codebase heavily uses OpenCV, Tkinter, PyTorch (YOLO), ByteTrack, DuckDB, and prefer small, explicit updates around existing data flow.
- When investigating UI or tracking issues, start with [src/soccer_homography/appState.py](src/soccer_homography/appState.py) and the relevant module in [src/soccer_homography](src/soccer_homography), rather than guessing at a cross-cutting rewrite.
- Use the project script defined in [pyproject.toml](pyproject.toml) for the main app launch: `uv run soccer-homography`. Do not invoke `gui.py` or other non-standard entry points (README.md has an outdated reference to `uv run python .\\gui.py`).

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

- [src/soccer_homography/__init__.py](src/soccer_homography/__init__.py) — main UI and app lifecycle (`App` class, `main()` function)
- [src/soccer_homography/appState.py](src/soccer_homography/appState.py) — application state, calibration data, model configuration
- [src/soccer_homography/SportsTracker.py](src/soccer_homography/SportsTracker.py) — tracking pipeline (YOLO + ByteTrack) and command loop
- [src/soccer_homography/dataTypes.py](src/soccer_homography/dataTypes.py) — core data structures
- [src/soccer_homography/ui/](src/soccer_homography/ui/) — canvas components:
  - `MainCanvasController`: main image display with tracking boxes
  - `RadarCanvas`: bird's eye view radar map (point matcher)
  - `HomographyUI`: homography mapping interface
  - `LivePreview`: live preview canvas
  - `Configuration`: configuration widget
- [src/soccer_homography/db/](src/soccer_homography/db/) — database persistence (`duckdb`):
  - `persist.py`: DB initialization and access
  - `io.py`: DB IO operations

## Tracking and UI workflow

Use this workflow when changing detection, tracking, homography, or UI behavior in the soccer analysis app.

### Start from the real app data flow

1. [src/soccer_homography/appState.py](../../src/soccer_homography/appState.py) holds the long-lived application state, calibration data, and model configuration.
2. [src/soccer_homography/__init__.py](../../src/soccer_homography/__init__.py) is the main Tkinter app, widget lifecycle, and event handlers.
3. [src/soccer_homography/SportsTracker.py](../../src/soccer_homography/SportsTracker.py) contains the tracking pipeline and command loop (threaded).
4. [src/soccer_homography/ui/](../../src/soccer_homography/ui/) contains canvas components that render detection and mapping updates.

### Preferred investigation order

1. Trace the state object and relevant frame data before changing logic, especially if a fix affects homography, track persistence, or UI refresh.
2. Check the event handler or controller that triggers the behavior in [src/soccer_homography/__init__.py](../../src/soccer_homography/__init__.py).
3. Follow the downstream model/tracking flow in [src/soccer_homography/SportsTracker.py](../../src/soccer_homography/SportsTracker.py) and any canvas-specific logic in [src/soccer_homography/ui/](src/soccer_homography/ui/).
4. Keep edits small and explicit; do not rewrite the app around a new abstraction unless the task genuinely demands it.

### Safe editing patterns

- Prefer updating existing state fields and event callbacks instead of introducing a new app-wide pattern.
- Preserve the current threading and command flow around tracking jobs; UI and tracking state are tightly coupled via queues.
- If a change modifies homography or track calculations, also inspect any refresh/update calls that redraw the radar or live preview.
- Avoid broad refactors in the same patch as a bug fix; isolate the behavior change.

### Validation

- Prefer the smallest practical smoke test that imports the package and exercises the touched area (e.g., `uv run python -c "from soccer_homography import AppState"`).
- For Python-only changes, use a compile or import check such as `uv run python -m compileall src`.
- For UI or tracking behavior, validate the user-visible path by running the app with the project entry point and confirming the affected interaction still works.

### Entry point

Use the project script from [pyproject.toml](../../pyproject.toml):

```pwsh
uv run soccer-homography
```

> **Note**: The README.md contains an outdated reference to `uv run python .\\gui.py`. There is no standalone gui.py at the root level. The correct entry point is defined in pyproject.toml as `soccer-homography = "soccer_homography:main"`.

## Notes for agents

- The project is experimental and not a production service; do not assume formal tests or a standard CI workflow are present (pytest exists but is minimal).
- Validate changes with the smallest practical command or script that exercises the affected behavior.
- If a fix touches model/tracking behavior, inspect the surrounding state and data flow before changing logic, because this app passes frame, calibration, and track state through many UI and detection components.
- PowerShell scripts like [autoannotate.ps1](autoannotate.ps1) and [exportannotate.ps1](exportannotate.ps1) are used for video annotation workflows (YOLO inference → COCO dataset conversion). These are separate from the main tracking app.
