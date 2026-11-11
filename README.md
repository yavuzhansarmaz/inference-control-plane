# Inference Control Plane

Reusable, open-source inference observability and optimization control plane. It lets you connect, inspect, and optimize existing inference pipelines (Helicone, Langfuse, OpenTelemetry, vLLM) from a host application without moving that logic into the host.

## About Freuron

[Freuron](https://freuron.com) is an AI platform where teams train, fine-tune, and serve their own language models. Inference observability — understanding what a model costs, how fast it answers, and where capacity is wasted — is a core part of that platform.

This repository is the open-source edition of an important part of Freuron: the inference control plane that runs in production inside Freuron to inspect and optimize inference workloads. We use this same package ourselves, and we are releasing it so anyone can connect, inspect, and optimize their existing inference pipelines — with vLLM, Langfuse, Helicone, or OpenTelemetry — without adopting the rest of Freuron.

## What it does

- Provider adapters for Helicone, Langfuse, and OpenTelemetry (OTLP) that pull and normalize inference events.
- Deterministic workload analysis: scoring, cost/latency/quality projections against a managed target rate, and migration-candidate detection.
- Read-only vLLM monitoring via Prometheus `/metrics` and `/health` (GET only).
- Evidence-based recommendations and measured-vs-estimated comparisons between a current endpoint and a managed serving pool.
- Sample-data generation for evaluating the analysis without real provider data.

The package is stateless. The host application owns authentication, persistence, pricing configuration, and credential storage, and injects them as arguments. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Install

```bash
pip install inference-control-plane
# local development against a checkout:
pip install -e ./inference-control-plane
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

`snapshot_from_pool_status(pool_payload)` converts a generic serving-pool payload (including per-replica `resources`) into a `MetricSnapshot` so it can be compared with an external vLLM snapshot. Missing metrics stay `None`, never `0`.

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

- Serving-pool payloads without measured per-replica `resources` read as `None` for those fields.
- No snapshot history or polling: `collect()` is a single point-in-time read. Callers that need rates call `derive_rates` on two snapshots they collected themselves.
- Host-application telemetry, persistence, and billing are not part of this package; see `docs/ARCHITECTURE.md`.

## Development

```bash
pip install -e '.[dev]'
python -m pytest
python -m ruff check src tests
```

## Releasing / rollback

See [docs/RELEASING.md](docs/RELEASING.md). Consumers pin an exact version; rollback is pinning the previous version and redeploying — no data migration is involved.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
