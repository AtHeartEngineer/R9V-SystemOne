# R9V System-One

R9V System-One is a local finite-candidate scoring service. It evaluates each
question against one shared state, assigns boundary-safe single-token labels to
the supplied semantic candidates, and normalizes only those candidates. The
HTTP endpoint is `POST /v1/systemone`. The production NixOS unit listens only
on `127.0.0.1:8101` and sends scoring requests to the existing R9V service at
`http://127.0.0.1:8000`.

## Configuration

Settings use the `SYSTEMONE_` prefix. The production unit fixes the public
listener, upstream model, and concurrency values below. Other settings retain
the application defaults unless an optional root-readable environment file
overrides them.

| Variable | Production value | Purpose |
| --- | --- | --- |
| `SYSTEMONE_BIND_HOST` | `127.0.0.1` | Loopback-only HTTP listener; non-loopback values are rejected |
| `SYSTEMONE_BIND_PORT` | `8101` | Local HTTP port |
| `SYSTEMONE_R9V_BASE_URL` | `http://127.0.0.1:8000` | Existing R9V OpenAI-compatible endpoint |
| `SYSTEMONE_R9V_MODEL` | `qwen3.8-flash-next` | Served model identifier |
| `SYSTEMONE_CONCURRENCY` | `1` | Maximum simultaneous upstream scoring requests |
| `SYSTEMONE_TIMEOUT_SECONDS` | `300` | Upstream request timeout |
| `SYSTEMONE_MAX_CHOICES` | `16` | Maximum candidate count for a choice request |
| `SYSTEMONE_LOG_LEVEL` | `INFO` | Structured-log level |
| `SYSTEMONE_CALIBRATION_PATH` | unset | Optional versioned calibration JSON file |
| `SYSTEMONE_API_KEY` | unset | Optional upstream API key |

On NixOS, place local secret overrides in `/etc/r9v-systemone.env`, owned by
root with mode `0600`. The file is optional. It is read at service start and is
never copied into the Nix store. Do not put keys in the Nix module, Git, command
arguments, or curl history.

## Question types

`choice` remains the foundational question type. Its criteria are an ordered
object of semantic key to string or null description. Its result contains the
winning key, the complete candidate distribution, and calibration metadata.

`noul` is the same scorer over two fixed semantic anchors in this exact order:
`false`, then `true`. The returned `noul` value is `P(true)` after the selected
temperature is applied. It is a yes/true candidate probability; `0.5` means the
two candidates received equal normalized weight, not a medium amount of a
property.

`score` accepts 2 through 10 ordered criteria. Position `i` is semantic anchor
`i`, and the result is:

```text
score = sum(i * probabilities[str(i)])
```

The range is therefore `0` through `K - 1`. The service returns every index
probability and a `legend` containing the original criteria. It does not
normalize the score to `0..1` or convert it into application units.

Scalar criteria descriptions may be strings, JSON objects, JSON arrays, or
null. Structured values are rendered into the prompt as compact JSON with
sorted object keys, so equivalent objects have deterministic prompt text.
Score's numeric response indices are not shown in the prompt: each description
is paired only with an opaque candidate label, and null is label-only. A Noul
criteria object is optional and may contain only `false` and/or `true`. This
service requires nonblank string `instructions` for every question type.

Example request:

```json
{
  "state": "sensor.condition: stable",
  "questions": {
    "usable": {
      "type": "noul",
      "instructions": "Is the sensor usable?",
      "criteria": {
        "false": "not usable",
        "true": {"meaning": "usable"}
      }
    },
    "condition": {
      "type": "score",
      "instructions": "Rate sensor condition.",
      "criteria": ["poor", {"label": "acceptable"}, ["excellent"]]
    }
  },
  "include_diagnostics": true
}
```

A Score result has this local response shape:

```json
{
  "type": "score",
  "score": 1.3,
  "legend": {"0": "poor", "1": {"label": "acceptable"}, "2": ["excellent"]},
  "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5},
  "probability_kind": "raw_renormalized",
  "calibration": {"applied": false, "temperature": 1.0},
  "raw_logprobs": {"0": -1.61, "1": -1.20, "2": -0.69}
}
```

`raw_logprobs` is present only when `include_diagnostics` is true. Noul returns
the corresponding `type`, `noul`, `probability_kind`, and `calibration` fields,
plus optional raw log probabilities keyed by `false` and `true`. Neither type
returns a `confidence` field.

A Choice request uses caller-defined semantic keys:

```json
{
  "state": "sensor.office_temperature: 25.4 C",
  "questions": {
    "comfort": {
      "type": "choice",
      "instructions": "Classify the room.",
      "criteria": {
        "cool": "below a comfortable range",
        "comfortable": "within a comfortable range",
        "warm": "above a comfortable range"
      }
    }
  }
}
```

Submit any request file with:

```bash
curl --fail-with-body --silent --show-error \
  -H 'content-type: application/json' \
  --data-binary @request.json \
  http://127.0.0.1:8101/v1/systemone | jq
```

## Calibration and compatibility boundary

`raw_renormalized` means candidate-only softmax output. It is not evidence of
empirical calibration and is not a probability that the answer is correct.
`temperature_calibrated` appears only when a configured, versioned family
temperature was actually applied. That metadata does not establish validity on
a new population.

This implementation is compatible with the published finite-anchor response
and arithmetic semantics. Its LitJev-style finite-choice prompt places all
level descriptions together, paired with opaque labels. Hosted Jev documents
that it evaluates a level without exposing the level number or neighboring
levels; this local prompt does not reproduce that internal procedure. It also
does not reproduce hosted Jev's proprietary model, learned calibration,
unpublished confidence statistic, performance, or numerical outputs. It does
not create continuous logits or inspect hidden states. LitJev itself describes
its implementation as an independent, hypothesis-based reproduction and says
its probabilities are not calibrated by default.

## Source snapshot

Semantics were checked on 2026-09-19 against these primary sources:

- LitJev current main at declared version 0.1.0, revision
  [`f21216c9fe5afe7fa52ff7064a402ee57fdbddd3`](https://github.com/zhengxuyu/litjev/tree/f21216c9fe5afe7fa52ff7064a402ee57fdbddd3).
  The repository had no tag or release at that revision. Its
  [`schema.py`](https://github.com/zhengxuyu/litjev/blob/f21216c9fe5afe7fa52ff7064a402ee57fdbddd3/src/litjev/schema.py#L31-L54)
  defines fixed `false`/`true` Noul anchors and 2–10 Score anchors; its
  [`decision.py`](https://github.com/zhengxuyu/litjev/blob/f21216c9fe5afe7fa52ff7064a402ee57fdbddd3/src/litjev/decision.py#L92-L110)
  returns `P(true)` for Noul and the full-distribution expected index for Score.
- Official TypeSafe documentation snapshots for
  [Noul](https://docs.typesafe.ai/primitives/noul),
  [Score](https://docs.typesafe.ai/primitives/score), and
  [confidence](https://docs.typesafe.ai/confidence), accessed 2026-09-19. The
  documentation site exposed no source commit; these are access-date snapshots.
- Official JavaScript SDK `v0.6.0`, revision
  [`66880ccded6cb642dc1809620c2b108c33730214`](https://github.com/typesafe-ai/typesafe-sdk-js/tree/66880ccded6cb642dc1809620c2b108c33730214),
  and Python SDK `v0.7.0`, revision
  [`2ce5c65f13646cab6e6f782328194c9d85f3300a`](https://github.com/typesafe-ai/typesafe-sdk-python/tree/2ce5c65f13646cab6e6f782328194c9d85f3300a).
  The JavaScript SDK enforces the two-level minimum but not the documented
  ten-level maximum. The Python SDK enforces neither bound fully and excludes
  top-level null descriptions. This service follows the explicit primitive
  documentation and LitJev for 2–10 nullable Score levels, while retaining its
  existing strict, required-instructions policy.

## Tests

Run the scalar compatibility suite:

```bash
.venv/bin/pytest tests/test_scalar.py -v
```

Run the complete local suite without the opt-in live checks:

```bash
.venv/bin/pytest -m 'not live'
```

Run the opt-in tests only while the accepted local R9V service is healthy:

```bash
SYSTEMONE_RUN_LIVE_API=1 .venv/bin/pytest tests/test_live_api.py -v
```

Run the benchmark contract suite against the committed corpora with:

```bash
.venv/bin/pytest tests/test_benchmark.py -v
```

The `r9v_systemone.benchmark.run_benchmark` application programming interface
is used by the production benchmark adapter; this project does not expose a
command-line benchmark entry point. Production runs record raw results and R9V
metric deltas. Review the output before drawing latency, cache, or correctness
conclusions; candidate-normalized scores are not automatically calibrated
probabilities.

## Health and systemd operation

`GET /health` checks both proxy liveness and bounded R9V/model readiness:

```bash
curl --fail-with-body http://127.0.0.1:8101/health
```

Healthy output is `{"status":"ok","r9v":"ready"}`. If R9V is starting or
unavailable, the proxy remains running and returns HTTP 503 with
`{"status":"degraded","r9v":"unavailable"}`. It does not start, stop, reload,
or restart R9V. Startup performs three bounded readiness checks before serving;
each check requires both R9V liveness and the configured model in `/v1/models`.
The proxy still starts in degraded mode if those checks fail. Failed proxy
processes use bounded `Restart=on-failure`; ordinary upstream degradation is
handled by the health response.

Useful local operations are:

```bash
systemctl status r9v-systemone.service
journalctl -u r9v-systemone.service --since today
sudo systemctl restart r9v-systemone.service
ss -ltn 'sport = :8101'
```

The NixOS unit is independent of `r9v.service`: it has no `Requires=`, `PartOf=`,
or restart propagation relationship. Restarting or removing the proxy leaves
the Qwen container, loaded model, and ordinary R9V endpoint unchanged.

## Privacy and filesystem boundaries

Structured logs contain request identifiers, state hashes, counts, timings,
token usage, and failure categories. They do not contain state text, prompts,
API keys, raw upstream responses, or private answer content. systemd runs the
service as a dedicated dynamic user with a private home, temporary directory,
and devices; a read-only system; no new privileges; and writable access only to
`/var/lib/r9v-systemone`.

## Rollback

To roll back an activation, select the preceding NixOS generation from the boot
menu or run `sudo nixos-rebuild switch --rollback`. To remove only this proxy,
revert the `../../modules/services/r9v-systemone.nix` import from
`hosts/aheo/default.nix` and switch the configuration. Neither action changes
the separate R9V checkout, image, container configuration, model files, or
cache. Verify rollback with `systemctl is-active r9v.service` and
`curl --fail http://127.0.0.1:8000/health`.
