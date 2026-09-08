---
applyTo: "src/soccer_homography/**/*.py"
description: "Use when working on the tracking pipeline, Tkinter UI flow, homography mapping, or frame-processing state changes in the soccer homography app."
---

# Tracking / UI workflow

Use this workflow when changing detection, tracking, homography, or UI behavior in the soccer analysis app.

## Start from the real app data flow

- [src/soccer_homography/appState.py](../../src/soccer_homography/appState.py) holds the long-lived application state, calibration data, and model configuration.
- [src/soccer_homography/__init__.py](../../src/soccer_homography/__init__.py) is the main Tkinter app, widget lifecycle, and event handlers.
- [src/soccer_homography/SportsTracker.py](../../src/soccer_homography/SportsTracker.py) contains the tracking pipeline and command loop.
- [src/soccer_homography/ui](../../src/soccer_homography/ui) contains the canvas and preview components that render the detection and mapping updates.

## Preferred investigation order

1. Trace the state object and relevant frame data before changing logic, especially if a fix affects homography, track persistence, or UI refresh.
2. Check the event handler or controller that triggers the behavior in [src/soccer_homography/__init__.py](../../src/soccer_homography/__init__.py).
3. Follow the downstream model/tracking flow in [src/soccer_homography/SportsTracker.py](../../src/soccer_homography/SportsTracker.py) and any canvas-specific logic in [src/soccer_homography/ui](../../src/soccer_homography/ui).
4. Keep edits small and explicit; do not rewrite the app around a new abstraction unless the task genuinely demands it.

## Safe editing patterns

- Prefer updating existing state fields and event callbacks instead of introducing a new app-wide pattern.
- Preserve the current threading and command flow around tracking jobs; UI and tracking state are tightly coupled.
- If a change modifies homography or track calculations, also inspect any refresh/update calls that redraw the radar or live preview.
- Avoid broad refactors in the same patch as a bug fix; isolate the behavior change.

## Validation

- Prefer the smallest practical smoke test that imports the package and exercises the touched area.
- For Python-only changes, use a compile or import check such as `uv run python -m compileall src`.
- For UI or tracking behavior, validate the user-visible path by running the app with the project entry point and confirming the affected interaction still works.

## Entry point

Use the project script from [pyproject.toml](../../pyproject.toml):

```pwsh
uv run soccer-homography
```
