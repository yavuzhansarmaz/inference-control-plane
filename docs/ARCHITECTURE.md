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
| `inference_control_plane.models` | Public pydantic models (`EndpointConfig`, `MetricSnapshot`, `ComparisonRow`, ...). Missing metrics are `None`, never fabricated zeros. |
| `inference_control_plane.vllm` | Read-only vLLM collector (`VllmClient`), Prometheus parsing, derived rates, cost estimates. GET only. |
| `inference_control_plane.recommendations` | Evidence-based recommendations and `compare_snapshots` (current vs managed). |
| `inference_control_plane.pool_status` | Adapts a generic serving-pool status payload to a `MetricSnapshot`. |
| `inference_control_plane.workload.adapters` / `adapter_base` | Provider adapter registry and shared contracts, including URL validation (SSRF guard). |
| `inference_control_plane.workload.helicone` / `langfuse` / `otlp` | Provider-specific event pull and normalization. |
| `inference_control_plane.workload.analysis` | Deterministic workload scoring and managed-target projections (`analyze_events`). |
| `inference_control_plane.workload.event_utils` / `sample_data` / `errors` | Event preparation, synthetic sample data, shared error type. |

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

`snapshot_from_pool_status` accepts any payload with a `replicas` list whose
ready entries may carry `engine_unfinished_requests` and a `resources` dict
(`kv_cache_utilization`, `gpu_sm_utilization`). Hosts with a different pool
shape should adapt their payload to this shape (or to `MetricSnapshot`
directly) in host code, not in this package.

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

- Serving-pool payloads without measured per-replica `resources` read as
  `None` for those fields.
- No snapshot history or polling: snapshots are point-in-time collections;
  the package keeps no time series and does not poll endpoints.

Host-application telemetry, persistence, and billing stay with the host and are not part of this package.
