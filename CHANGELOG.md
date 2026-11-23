# Changelog

## 0.1.4 - 2026-11-23

### Changed

- TODO: describe changes in this release.


## 0.1.3 - 2026-11-17

### Changed

- TODO: describe changes in this release.


## Unreleased

### Deprecated

- `WorkloadOptimizerError` is a deprecated alias of the new generic
  `ControlPlaneError` (`inference_control_plane.errors`); the old import
  paths keep working in 0.1.x.

### Removed

- `snapshot_from_pool_status` and `workload.event_utils.prepare_events`
  moved to the host application; they are no longer part of this package.
  Threshold literals were unified in `inference_control_plane.thresholds`.
- `workload.sample_data.build_sample_events` was removed. The host supplies
  its own events.


### Changed

- Internal refactor only (no public behavior or import-path changes): split `recommendations.py` and `vllm.py` into single-responsibility modules (`prometheus`, `rates`, `metrics_client`, `snapshot_recommendations`, `history_recommendations`, `comparison`, `actions`) behind the original facade modules, and removed redundant private helpers.

### Added

- In-memory `SnapshotHistory` ring buffer and asyncio `SnapshotPoller`, plus `rates_from_history` (host owns persistence; no threads, no storage).
- `recommend_from_history`: window-aggregated recommendations with sample-count evidence and confidence scaling.
- Fleet view: `summarize_fleet` / `FleetSummary` / `recommend_from_fleet` (ratios averaged, never summed; count sums name their coverage).
- `render_prometheus` exporter (None values skipped, never 0) and `docs/grafana-dashboard.json`.
- Read-only `SglangClient` and `snapshot_from_sglang_metrics` (provider `sglang`).
- Typed `WorkloadEvent` model; `analyze_events` accepts dicts or events and still returns dicts.
- `ActionSuggestion` model and `suggest_actions` (human-approved, review-only; no fabricated numeric suggestions).

## 0.1.2 - 2026-11-11

### Changed

- Publish workflow requires a TestPyPI smoke test before PyPI. A failed smoke test blocks the release.
- README local install command corrected to `pip install -e .` from the repository root.

## 0.1.1 - 2026-11-11

### Changed

- Publish pipeline checks that the `vX.Y.Z` tag matches `pyproject.toml`, runs tests before publish, and attaches the changelog section to the GitHub release.
- Added automatic release version planning (`scripts/plan_release_version.py`) and `scripts/set_version.py`.
- README intro names Freuron, and documentation links are absolute so they resolve on PyPI.

## 0.1.0
- Added workload-optimizer provider adapters (Helicone, Langfuse, OTLP), event preparation, sample data, and analysis.
- Added read-only vLLM monitoring (`VllmClient`), evidence-based recommendations, and measured-vs-estimated comparisons.
