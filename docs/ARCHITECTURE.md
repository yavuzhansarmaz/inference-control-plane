# Architecture

## Purpose

`inference-control-plane` is a stateless Python library for inference
observability and optimization. A host application embeds it to:

- pull and normalize inference events from external observability providers,
- analyze workloads deterministically (scoring, projections, candidates),
- monitor vLLM endpoints read-only, and
- compare a current endpoint against a managed serving pool.

The package owns no storage, no credentials, and no deployment. The host
application injects everything product-specific.

## Package layout

| Module | Responsibility |
|---|---|
| `inference_control_plane.models` | Public pydantic models (`EndpointConfig`, `MetricSnapshot`, `ComparisonRow`, `FleetSummary`, `ActionSuggestion`, ...). Missing metrics are `None`, never fabricated zeros. `ActionSuggestion` is review-only (`requires_approval=True`). |
| `inference_control_plane.vllm` | Read-only vLLM collector (`VllmClient`, capabilities probe) and `snapshot_from_metrics`. GET only. Re-exports `parse_prometheus`, `derive_rates`, `estimate_effective_cost` from their modules below. |
| `inference_control_plane.prometheus` | Prometheus exposition parsing, histogram p95, error-rate-from-totals (shared by vLLM/SGLang). |
| `inference_control_plane.rates` | Derived counter rates (`derive_rates`) and effective-cost estimates. |
| `inference_control_plane.metrics_client` | Private shared read-only metrics-client scaffold (SSRF validation, never-raise collect/test) for `VllmClient`/`SglangClient`. |
| `inference_control_plane.recommendations` | Facade re-exporting evidence-based recommendations and `compare_snapshots` (current vs managed) from the four modules below. |
| `inference_control_plane.snapshot_recommendations` / `history_recommendations` | `recommend_from_snapshot`; window-aggregated `recommend_from_history`. |
| `inference_control_plane.comparison` / `actions` | `compare_snapshots`; review-only `suggest_actions` (`requires_approval=True`). |
| `inference_control_plane.timeutil` | `now_iso` and `parse_iso` (naive timestamps treated as UTC). |
| `inference_control_plane.thresholds` | Shared recommendation cut-offs used by snapshot, history, and fleet recommendations. |
| `inference_control_plane.errors` / `workload.errors` | Canonical `ControlPlaneError`. `workload.errors` re-exports it. `WorkloadOptimizerError` is a deprecated alias for 0.1.x. |
| `inference_control_plane.workload.adapters` / `adapter_base` | Provider adapter registry and shared contracts, including URL validation (SSRF guard). |
| `inference_control_plane.workload.helicone` / `langfuse` / `otlp` | Provider-specific event pull and normalization. |
| `inference_control_plane.workload.analysis` | Deterministic workload scoring and managed-target projections (`analyze_events`). |
| `inference_control_plane.history` | In-memory bounded `SnapshotHistory` ring buffer and asyncio `SnapshotPoller`; caller-owned, no persistence, no threads. |
| `inference_control_plane.fleet` | Fleet aggregation (`FleetSummary`) and fleet-level recommendations over named snapshots. |
| `inference_control_plane.exporter` | Pure `render_prometheus` text rendering; no server. See `docs/grafana-dashboard.json`. |
| `inference_control_plane.sglang` | Read-only SGLang collector (`SglangClient`), same shape as `vllm`. |
| `inference_control_plane.workload.events` | Typed `WorkloadEvent` model. `analyze_events` accepts dicts or events and returns dicts. |

## Boundary

```
Host application -> (pinned) inference-control-plane -> providers / vLLM endpoints
```

- The package never imports host modules. It has no knowledge of host users,
  tenants, billing, settings, or databases.
- Callers inject base URLs, credentials, `allow_private_endpoints`, pricing
  targets (for example `managed_rate_usd_per_million`), and persistence.
- Tenant isolation, authorization, and credential encryption stay in the host
  application's repository filters and route guards.
- All external calls are read-only. The package never sends inference
  traffic, never mutates provider state, and never allocates compute.

## Data flow

1. The host creates a provider adapter (or a `VllmClient`) with an injected
   base URL and credentials.
2. The adapter pulls raw events (or a metrics snapshot) and normalizes them
   into plain dicts / `MetricSnapshot` values. Unknown values stay `None`.
3. The host persists or displays the result. `analyze_events` and
   `compare_snapshots` are pure functions over already-collected data.
4. Provenance is explicit: every comparison row and cost figure carries a
   `kind` of `measured`, `estimated`, or `user_assumption`.

## Extension points

### Adding a provider

Implement `ProviderAdapter.pull()` (or a read-only collector like
`VllmClient`), normalize missing metrics to `None`, register the adapter in
`workload/adapters.py` (or expose a client), and add mocked tests. No write
calls to external infrastructure.

### Adding a comparison metric

Extend `compare_snapshots` with a new `ComparisonRow`. Keep the managed side
`None` when the managed snapshot does not report the metric, and set `kind`
from the managed snapshot's status (`measured` only when it is `ok`).

### Custom pool payloads

The package does not parse serving-pool status payloads. The host maps its
pool into a `MetricSnapshot` (unset metrics stay `None`) and passes that
snapshot as `managed_snapshot` to `compare_snapshots`. `kind` is `measured`
only when that snapshot's status is `ok`.

## Design decisions

- **Library, not a service.** The reusable core is stateless, so it ships as
  a versioned package. A separate service would only be justified if the
  core needed independent scaling or data ownership.
- **Missing means `None`.** Absent metrics are never coerced to `0`; rates
  and projections derived from missing inputs are `None` as well.
- **Explicit private-endpoint opt-in.** Provider and endpoint URLs are
  validated before any request. Non-global addresses require
  `allow_private_endpoints=True` from the caller.
- **No pricing built in.** Target rates and GPU prices are caller-supplied
  assumptions and are labelled `user_assumption` in outputs.

## Known limitations

- A managed serving pool enters comparison only as a host-built
  `MetricSnapshot`. Fields the host leaves unset stay `None`.
- Snapshot history and polling are in-memory and caller-owned:
  `SnapshotHistory`/`SnapshotPoller` keep a bounded ring buffer in memory
  only; the host owns persistence, scheduling, and credentials.

Host-application telemetry, persistence, and billing stay with the host and are not part of this package.
