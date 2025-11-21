# Repository Guidelines

## Project Structure & Module Organization
- Core logic lives in `handTrack.py` and handles capture, gesture recognition (MediaPipe), and cursor control.
- Model asset `gesture_recognizer.task` sits alongside the script; keep it synced with the code version.
- Docs: `README.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`. Dependency manifests: `pyproject.toml`, `uv.lock`, `requirements.txt`.
- No tests yet; add new tests under `tests/` as `test_*.py` when introduced.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate` (Windows: `.\.venv\Scripts\activate`).
- Install deps: `pip install -r requirements.txt` (legacy pins) or `uv sync` to respect `pyproject.toml`/`uv.lock`.
- Run app: `python handTrack.py` launches webcam-driven cursor control.
- Quick syntax check (optional): `python -m compileall .`.

## Coding Style & Naming Conventions
- Python 3.10+ recommended; follow PEP 8 with 4-space indents.
- Use snake_case for functions/variables, PascalCase for classes/enums, descriptive gesture/state names (e.g., `POINT`, `detect_pinch`).
- Keep helpers small and single-purpose; briefly comment magic numbers (thresholds, cooldowns, margins).

## Testing Guidelines
- No automated suite yet; manual verification is expected: run `python handTrack.py` and exercise point, swipe, and pinch gestures in varied lighting/resolution setups.
- If adding tests, use `pytest`, place them under `tests/`, and name files `test_*.py`.

## Commit & Pull Request Guidelines
- Commit messages: short and imperative; observed prefixes include `feat:`, `build:`, `fix:`, or concise verbs (e.g., `disable scroll`).
- PRs should describe behavior changes, list manual test steps/results, and link issues when relevant.
- Include screenshots or short clips when UX/gesture behavior changes; note any new configuration knobs or defaults.
- Keep diffs focused—avoid mixing feature work with formatting-only changes.

## Security & Configuration Tips
- App uses webcam input and simulates mouse events; be mindful of privacy and safety impacts.
- Validate defaults for movement thresholds, inner-area percentage, and cooldowns to prevent runaway cursor motion.
- Avoid adding large binaries beyond the existing model; reference release assets if you must change models.
