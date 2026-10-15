# Model Router

`model_router` is an internal AI routing service designed to sit behind a website or upstream application.

Its job is narrow:

- choose the model route for a request
- apply routing-profile policy
- do keyword / semantic rerouting
- do cascade fallback
- forward the request to the provider
- normalize the response

It is not intended to be the source of truth for:

- end-user authentication
- permission checks
- billing
- business audit logs

Those should live in the website or upstream system.

## Current Role

Recommended production shape:

- website authenticates the user
- website decides `user_id`
- website sends an internal request to the router
- router resolves `routing_profile`
- router applies routing policy and forwards the request

The router still supports a standalone compatibility mode for local testing, but that is no longer the primary deployment model.

## Core Capabilities

- OpenAI-compatible `POST /v1/chat/completions`
- OpenAI-compatible `GET /v1/models`
- multi-provider routing for OpenAI / Anthropic / DeepSeek / GLM
- provider key rotation and failover
- profile-driven routing policy
- profile-scoped keyword routing
- semantic routing
- profile-scoped cascade fallback
- ensemble routing
- provider health tracking
- route preview / inspect
- technical telemetry with `request_id`

## Deployment Docs

For rollout and staging instructions, see:

- [Production rollout](docs/deployment/PRODUCTION_ROLLOUT.md)
- [Deployment example](docs/deployment/DEPLOYMENT_EXAMPLE.md)

## Runtime Modes

### Internal Mode

Use this in production behind the website.

Recommended environment:

```bash
export ROUTER_RUNTIME_MODE=internal
export ROUTER_INTERNAL_API_KEY=replace-with-strong-secret
export ROUTER_ENABLE_STANDALONE_KEYS=false
export ROUTER_ENABLE_ADMIN_API=false
export ROUTER_ENABLE_USAGE_API=false
export ROUTER_DB=router.db
```

### Standalone Compatibility Mode

Use this only for local/dev testing when you want the router to accept direct API keys.

Example:

```bash
export ROUTER_RUNTIME_MODE=standalone
export ROUTER_ENABLE_STANDALONE_KEYS=true
export ROUTER_ENABLE_ADMIN_API=true
export ROUTER_ENABLE_USAGE_API=true
export ADMIN_API_KEY=dev-admin-secret
```

## Website To Router Contract

In internal mode, the website should call the router with:

- `Authorization: Bearer <ROUTER_INTERNAL_API_KEY>`
- `X-User-Id: <website user id>`
- `X-Routing-Profile-Id: <profile id>` when already known
- `X-Request-Id: <request id>`
- `X-Allowed-Models: <optional csv list>` when the website wants an extra hard restriction

## Routing Profiles

Routing behavior is driven by `routing_profile`.

The current `routing_profile.config_json` supports:

- `default_model`
- `preferred_providers`
- `allowed_models`
- `deny_models`
- `keyword_routing_enabled`
- `semantic_routing_enabled`
- `cascade_enabled`
- `cascade_override_chain`
- `cost_bias`
- `latency_bias`
- `quality_bias`

Example:

```json
{
  "default_model": "gpt-5.4",
  "preferred_providers": ["deepseek", "openai"],
  "allowed_models": ["gpt-5.4", "deepseek-v4-pro", "shared-model"],
  "deny_models": ["claude-sonnet-5"],
  "keyword_routing_enabled": true,
  "semantic_routing_enabled": true,
  "cascade_enabled": true,
  "cascade_override_chain": ["deepseek-v4-pro", "gpt-5.5"],
  "cost_bias": 1.0,
  "latency_bias": 0.5,
  "quality_bias": 0.2
}
```

## Cascade Local Judge

`cascade` can now do more than hard checks like minimum length or refusal detection.

You can optionally attach a local judge model to a cascade rule. The router will then:

1. run the normal hard validator
2. if that passes, call the configured judge model
3. accept the answer only if the judge says it is good enough
4. otherwise continue to the next model in the cascade chain

Useful cascade-rule fields:

- `local_judge_model`
- `local_judge_threshold`
- `local_judge_system_prompt`

The response `cascade` metadata may also include:

- `judge_model`
- `judge_score`
- `judge_reason`

Example admin payload:

```json
{
  "routing_profile_id": 1,
  "name": "careful-fallback",
  "source_routing_key": "deepseek-v4-pro",
  "escalation_chain": ["deepseek-v4-pro", "gpt-5.5"],
  "min_content_length": 20,
  "local_judge_model": "claude-sonnet-5",
  "local_judge_threshold": 0.75,
  "local_judge_system_prompt": "You are a strict production answer judge for cascade routing. Mark valid=true only when the candidate answer fully addresses the user's request, is specific, not evasive, and matches the requested output format. If the answer is partial, vague, padded, or misses important constraints, return valid=false. Return raw JSON only.",
  "max_escalations": 1,
  "priority": 10,
  "note": "use a local judge before escalating to gpt-5.5"
}
```

Recommended starting policy:

- use `local_judge_model` = `claude-sonnet-5` or your most reliable reasoning model
- use `local_judge_threshold` = `0.75`
- keep `min_content_length` as a cheap first-pass filter, not your main quality gate
- let the judge decide whether a long but weak answer should still escalate

## Request Flow

Internal request flow:

1. website authenticates the user
2. website sends internal request headers
3. router resolves `RequestContext`
4. router resolves `routing_profile`
5. router loads profile policy and effective routing settings
6. router resolves the initial model target
7. router optionally applies keyword / semantic rerouting
8. router optionally applies cascade fallback
9. router forwards the request upstream
10. router returns normalized output
11. router records technical telemetry

## API Surface

### Public Runtime Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

### Admin Endpoints

Admin endpoints are mounted under `/admin` when `ROUTER_ENABLE_ADMIN_API=true`.

Key areas:

- providers
- provider keys
- model routes
- routing profiles
- keyword rules
- cascade rules
- route preview
- status
- usage telemetry APIs

Note:

- standalone key endpoints are disabled when `ROUTER_ENABLE_STANDALONE_KEYS=false`
- usage telemetry endpoints are disabled when `ROUTER_ENABLE_USAGE_API=false`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Provider Access

This project seeds provider and model configuration from `config/providers.yaml`.

For a gateway-style deployment, the website can stay fixed while keys remain split by model family. A common setup is:

```bash
export GATEWAY_BASE_URL=https://your-gateway.example/v1
export OPENAI_KEY_1=sk-your-openai-family-key
export ANTHROPIC_KEY_1=sk-your-claude-family-key
export DEEPSEEK_KEY_1=sk-your-deepseek-family-key
export GLM_KEY_1=sk-your-glm-family-key
```

If needed, you can still override individual base URLs:

```bash
export OPENAI_BASE_URL=https://your-gateway.example/v1
export ANTHROPIC_BASE_URL=https://your-gateway.example/v1
export DEEPSEEK_BASE_URL=https://your-gateway.example/v1
export GLM_BASE_URL=https://your-gateway.example/v1
```

### 3. Start The Service

```bash
python -m app.main
```

Or:

```bash
./start.sh
```

### 4. Check Health

```bash
curl http://127.0.0.1:8000/health
```

## Configuration

Important runtime variables:

- `ROUTER_RUNTIME_MODE`
- `ROUTER_INTERNAL_API_KEY`
- `ROUTER_ENABLE_STANDALONE_KEYS`
- `ROUTER_ENABLE_ADMIN_API`
- `ROUTER_ENABLE_USAGE_API`
- `ROUTER_DB`
- `AUTO_SYNC_PROVIDER_MODELS`
- `AUTO_SYNC_PROVIDER_MODELS_TIMEOUT`
- `ADMIN_API_KEY`
- `ROUTER_MODEL_COOLDOWN_S` (default `60`; temporary model-level quarantine after an explicit unavailable-channel error)
- `ROUTER_MODEL_UNAVAILABLE_RETRIES` (default `3`; failure-only re-selection for `auto`, not quality cascade)

When an automatically recommended model returns an explicit "model not found",
"model unavailable", or "no available channel" error, the router temporarily
quarantines that real model and all of its routing-key aliases. The same
request can then select another ranked candidate, without judging the answer
or escalating a successful response. The quarantine is process-local and
expires automatically.

Use `GET /admin/status` to inspect active model quarantines, or
`POST /admin/route/preview` to see a quarantined model in the recommendation
`excluded` list.

## CLI Commands

Available management commands:

```bash
python -m app.manage init
python -m app.manage reset
python -m app.manage sync-models
python -m app.manage check-health
python -m app.manage preview-route gpt-5.4 "please translate this sentence"
python -m app.manage gen-key
python -m app.manage list-keys
python -m app.manage list-models
python -m app.manage stats
```

Other utilities:

```bash
python -m app.discover
python -m app.benchmark --admin-key "$ADMIN_API_KEY"
```

## Admin Setup Flow

In staging or dev, a common setup flow is:

1. create providers
2. add provider keys
3. sync models
4. create routing profiles
5. bind `user_id -> routing_profile`
6. add profile-scoped keyword or cascade rules
7. preview routes before real traffic

See [Deployment example](docs/deployment/DEPLOYMENT_EXAMPLE.md) for concrete commands.

## Architecture

```text
Website / Upstream App
  -> internal auth
  -> X-User-Id / X-Request-Id / X-Routing-Profile-Id
  -> model_router
     -> request context
     -> routing profile
     -> route policy
     -> keyword / semantic / cascade
     -> provider adapter
     -> upstream provider
```

## Project Structure

```text
model_router/
├── app/
│   ├── api/
│   │   ├── admin/
│   │   │   ├── providers.py
│   │   │   ├── models.py
│   │   │   ├── keys.py
│   │   │   ├── usage.py
│   │   │   ├── status.py
│   │   │   ├── keyword.py
│   │   │   ├── cascade.py
│   │   │   ├── inspect.py
│   │   │   └── routing_profiles.py
│   │   ├── chat.py
│   │   └── models.py
│   ├── core/
│   │   ├── auth.py
│   │   ├── request_context.py
│   │   ├── routing_policy.py
│   │   ├── routing_settings.py
│   │   ├── router.py
│   │   ├── cascade.py
│   │   ├── db.py
│   │   └── ...
│   ├── providers/
│   ├── schemas/
│   ├── main.py
│   ├── manage.py
│   ├── discover.py
│   └── benchmark.py
├── config/
│   └── providers.yaml
├── docs/
│   ├── deployment/
│   └── testing/
├── artifacts/
│   ├── reports/
│   └── usage_logs/
├── experiments/
│   └── chatbot-arena/
├── setup.py
├── start.sh
└── tests/
```

## Troubleshooting

### No Models Loaded

- check provider keys
- run `python -m app.manage sync-models`
- check `config/providers.yaml`

### Internal Requests Return 401

- verify `ROUTER_INTERNAL_API_KEY`
- verify website sends `Authorization: Bearer <token>`

### Standalone Keys Do Not Work

- verify `ROUTER_ENABLE_STANDALONE_KEYS=true`
- verify you are not running in locked-down internal mode

### Admin Endpoints Return 404

- verify `ROUTER_ENABLE_ADMIN_API=true`

### Usage Endpoints Return 404

- verify `ROUTER_ENABLE_USAGE_API=true`

### Provider Failures

- run `python -m app.manage check-health`
- inspect `/admin/status` in staging/dev
- verify upstream credentials and base URLs

## Testing

Focused local verification:

```bash
python -m unittest tests.test_router_flows tests.test_route_preview tests.test_provider_health tests.test_remote_provider_health -v
```

Real-provider integration tests exist in:

- [tests/test_integration_real.py](./tests/test_integration_real.py:1)

They require explicit environment setup and are not intended for every local run.

## Status

This repository is now aligned around an internal-service-first router:

- website-driven identity
- routing-profile-driven behavior
- provider routing and fallback
- optional standalone compatibility for local testing

## RSL Extension

This repository includes a bounded Recursive Self-Learning module aligned with the supplied recursive self-improvement handbook and the JEV/TypeSafe structured evaluator example.

The loop is:

`baseline -> generate candidates -> multi-signal evaluation -> retain top-k -> revise from concrete failures -> rollback-safe versioned memory`

Implemented features:

- measurable single-generation baseline;
- per-round candidate search with `candidates_per_round` and `top_k`;
- three controls: single generation, generate/evaluate/revise, and candidate search;
- deterministic external checks for JSON, Python syntax, regex, required keys, and format;
- optional held-out checks that are not shown to the generator;
- local LLM judge plus `typesafe_sdk` `Choice`/`Noul` structured evaluation when configured;
- output normalization with `raw_answer` and `postprocessed` audit fields;
- model-call, token, and duration budgets;
- score diversity, duplicate rate, calibration, and stop reasons;
- low-trust memory: candidate becomes active only after repeated verified passes;
- memory versioning and rollback endpoint;
- full per-candidate traces in SQLite.

Main endpoints:

- `POST /v1/rsl/run`
- `GET /v1/rsl/runs`
- `GET /v1/rsl/runs/{run_id}`
- `GET /v1/rsl/memory`
- `POST /v1/rsl/memory/{memory_id}/rollback`
- `GET /v1/rsl/health`

Detailed assignment notes and experiment results are in `docs/RSL_ASSIGNMENT.md`.

Focused tests:

```bash
python -m unittest tests.test_rsl_engine -v
```

Run the reproducible experiments while the router is active:

```bash
python scripts/run_rsl_experiment.py
```
