# Inference Control Plane

Reusable, open-source inference observability and optimization control plane. It lets you connect, inspect, and optimize existing inference pipelines (Helicone, Langfuse, OpenTelemetry, vLLM, SGLang) from a host application without moving that logic into the host. This package runs in production at [Freuron](https://freuron.com), the AI platform where teams train, fine-tune, and serve their own language models; it is the open-source edition of the inference control plane Freuron uses to inspect and optimize its workloads, released so anyone can get the same visibility into their own pipelines without adopting the rest of the platform.

## The audit step of an AI Model Factory

Freuron's [AI Model Factory](https://freuron.com/model-factory) loop starts by
auditing production traffic — which workloads are ready for a dedicated
fine-tuned model — and this package is that audit layer. To close the loop,
Freuron then runs the loop for you: four API calls (import, train, evaluate,
switch) take a dataset to a production-ready model in under three hours, with
more than two hours inside fine-tuning itself. See
[LLM fine-tuning platform](https://freuron.com/fine-tuning) and
[private LLM inference](https://freuron.com/inference).

## What it does

- Provider adapters for Helicone, Langfuse, and OpenTelemetry (OTLP) that pull and normalize inference events.
- Deterministic workload aggregation with cost/latency/quality projections against a caller-supplied target rate.
- Read-only vLLM and SGLang monitoring via Prometheus `/metrics` and `/health` (GET only).
- Evidence-based recommendations and measured-vs-estimated comparisons between a current endpoint and a managed serving pool.

The package is stateless. The host application owns authentication, persistence, pricing configuration, and credential storage, and injects them as arguments. See the [architecture documentation](https://github.com/yavuzhansarmaz/inference-control-plane/blob/main/docs/ARCHITECTURE.md).

## Install

```bash
pip install inference-control-plane
# local development against a checkout (run inside the repository root):
pip install -e .
```

## Quick start (vLLM, read-only)

```python
from inference_control_plane import EndpointConfig, VllmClient
from inference_control_plane.recommendations import recommend_from_snapshot

client = VllmClient(EndpointConfig(base_url="https://my-vllm.example.com"))
snapshot = await client.collect()          # GET /metrics only; failures -> status="unreachable"
print(snapshot)                            # missing metrics are None, never 0
print(recommend_from_snapshot(snapshot))   # evidence-based, review-only
```

Private/self-hosted endpoints require `allow_private_endpoints=True` (explicit opt-in; the SSRF guard otherwise rejects non-global addresses, credentials in URLs, and non-HTTPS).

## Snapshot history and polling

`SnapshotHistory(maxlen=...)` is a bounded in-memory ring buffer of snapshots (`append`, `window(seconds)`, `last(n)`); the host owns any persistence. `SnapshotPoller(client, history, interval_seconds=...)` polls any client with `async collect()` via asyncio (no threads), recording unreachable snapshots instead of raising; run it with `await poller.run(stop_event)`. `rates_from_history(history)` derives rates from the first/last snapshots of a window; unknown rates stay `None`.

## Window and fleet recommendations

`recommend_from_history(history_or_snapshots, window_seconds=...)` aggregates thresholds over a window, cites sample counts in its evidence, and scales confidence with sample count (empty input yields `insufficient_evidence`, never an error). `summarize_fleet({"name": snapshot, ...})` builds a `FleetSummary` (status counts, worst offenders by name, means over present values; count sums name their coverage), and `recommend_from_fleet(...)` flags cases such as one saturated endpoint beside idle ones (`fleet_load_imbalance`).

## Action suggestions

`suggest_actions(snapshot_or_history)` maps recommendations to `ActionSuggestion`s (`target_parameter`, `current_value`, `suggested_value`, rationale, expected effect) that always require human approval. No number is suggested unless it can be computed from the given values — otherwise `suggested_value` is `None` with the reasoning in the rationale.

## Prometheus exporter

`render_prometheus(snapshots, prefix="inference_control_plane")` renders gauges with an `endpoint` label, skipping `None` values entirely, plus a `snapshot_status{status=...} 1` series per endpoint. It is a pure function — serving it is the host's job. A matching importable dashboard is in [docs/grafana-dashboard.json](docs/grafana-dashboard.json).

## SGLang (read-only)

`SglangClient` mirrors `VllmClient` for SGLang `/metrics` (provider `"sglang"`, same SSRF validation and never-raise collection). `snapshot_from_sglang_metrics(text, source=...)` maps `sglang:num_running_reqs`, `sglang:num_queue_reqs`, `sglang:token_usage`, and counters; `tokens_per_second` comes only from the `sglang:gen_throughput` rate gauge.

## Typed workload events

`WorkloadEvent` (from `inference_control_plane.workload`) is a pydantic model of the normalized adapter event dict, keeping extra fields through `to_dict()`. `analyze_events` accepts `list[dict | WorkloadEvent]` and still returns dicts.

## Capabilities probe

`VllmClient.capabilities()` probes `/health`, `/v1/models`, and `/metrics` with GET only and never raises: failures become notes in `EndpointCapabilities.errors`. It reports `health_ok`, the served `models`, and `metrics_ok`. Use it to check an endpoint before trusting a snapshot.

## Derived rates

`derive_rates(prev, curr)` is a pure function over two `MetricSnapshot`s. It returns `DerivedRates` (`tokens_per_second`, `generation_tokens_per_second`, `prompt_tokens_per_second`, `requests_per_second`). A missing counter, a non-positive window, or a rewound counter yields `None` for that rate — rates are never fabricated.

## Cost estimate

`estimate_effective_cost(snapshot, gpu_hourly_usd, tokens_per_second=None)` returns `None` when no GPU hourly price is supplied. Otherwise it returns an `EffectiveCost` with `hourly_cost_usd`, `effective_cost_per_utilized_hour_usd` (hourly price spread over the utilized GPU fraction; `None` when utilization is unknown or zero), and `cost_per_million_generation_tokens_usd`. The result always carries `kind="user_assumption"`: the price is asserted by the user, not measured, so every figure is an estimate, not a bill.

## Comparison and `kind` labels

`compare_snapshots(current, managed_snapshot=..., managed_kind=...)` returns `ComparisonRow`s (`metric`, `current`, `managed`, `unit`, `kind`, `notes`). Every row is labelled by provenance via `kind`:

- `measured` — read from a live endpoint or a ready managed serving-pool replica.
- `estimated` — derived from measured inputs (for example a rate or a pool side whose status is not `ok`).
- `user_assumption` — computed from a user-supplied input such as `gpu_hourly_usd`.

Pass a host-built `MetricSnapshot` as `managed_snapshot` to compare a current endpoint with a managed serving pool. The host maps replica fields into that snapshot. Any metric left unset stays `None`.

## Workload analysis

```python
from inference_control_plane.workload.analysis import analyze_events

summary, opportunities, assumptions = analyze_events(
    events,
    window_days=30,
    managed_rate_usd_per_million=1.0,
    ttft_ms=200.0,
    output_tokens_per_second=50.0,
)
```

`managed_rate_usd_per_million` is the host-supplied target rate used for cost projections. The package ships no pricing of its own.

## Known limitations

- A managed serving pool is compared through a `MetricSnapshot` the host supplies. Fields the host leaves unset stay `None`.
- Snapshot history is in-memory only: the host owns persistence of anything the poller collects.
- Host-application telemetry, persistence, and billing are not part of this package; see the architecture documentation linked above.

## Development

```bash
pip install -e '.[dev]'
python -m pytest
python -m ruff check src tests
```

## Releasing / rollback

See the [releasing guide](https://github.com/yavuzhansarmaz/inference-control-plane/blob/main/docs/RELEASING.md). Consumers pin an exact version; rollback is pinning the previous version and redeploying — no data migration is involved.

## License

Apache License 2.0 — see the [license](https://github.com/yavuzhansarmaz/inference-control-plane/blob/main/LICENSE).
