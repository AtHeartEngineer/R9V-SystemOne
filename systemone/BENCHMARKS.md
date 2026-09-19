# R9V System-One benchmarks

Measured on 2026-09-19 against the retained production profile. These are
observations from one machine and one model/image combination, not portable
performance claims.

## Configuration and method

| Item | Measured value |
| --- | --- |
| R9V source | `c4da4860dcf6ddb60b3d4eaae06ad1a679056cc5` |
| Pinned vLLM source | `9ec060c402ed15fbeb1dbd2714615261b63dda12` |
| System-One benchmark source | `c30038bc528694fbb89db3a441f37376fcebb03f` |
| Image | `sha256:580bbcecccb3f03907471ef1407506f4c84efbcf6597a77e31311e5e726754b4` |
| Model/profile | `qwen3.8-flash-next`; accepted 256K/MTP2 profile |
| Runtime gates | `max_model_len=262144`, `max_num_seqs=1`, prefix caching enabled, GPUs `0,1` |
| Candidate scoring | One `/v1/completions` request per candidate, `max_tokens=1`, `logprobs=0`, singleton `allowed_token_ids` |
| Generation comparator | One ordinary completion per question, `max_tokens=12`, exact semantic-key parsing |
| Calibration | None; temperature 1.0; `raw_renormalized` candidate scores |

The scorer never sent `logprob_token_ids`. For each candidate token `i`, it
recorded the raw returned log probability `l_i`, then computed:

```text
p_i = exp(l_i / T - logsumexp(l_1 / T, ..., l_n / T))
```

Here `T=1.0`. The result is normalized only over the supplied candidates. It is
not a calibrated probability that the answer is correct. Each physical scoring
request still forced and decoded one token.

The committed known-answer corpus contains 36 cases (SHA-256
`357b3682e78d1a0cf89d63a142b4fcee3536301edbbfdf53f9722f567c2143da`).
The Home Assistant-like workload contains 24 questions over a 4,954-token
synthetic state and a 4,992-token complete shared prefix (workload SHA-256
`0f447e1965ec95508f6faee5928de3641d5f3cbcda7f23329f02ff606de1e4a7`).
No private household state was used.

## Concurrency-one comparison

| Workload and method | Correct | Parse / request failures | Logical / physical requests | Wall time | Median / p95 per question | Logical / physical requests per second | Prompt / decoded tokens | Cache-hit / local-compute tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 36-case corpus, System-One | 35/36 | 0 / 0 | 36 / 112 | 122.681 s | 3.690 / 5.654 s | 0.293 / 0.913 | 10,136 / 112 | 0 / 10,136 |
| 36-case corpus, generation | 1/36 | 35 / 0 | 36 / 36 | 46.591 s | 1.267 / 1.535 s | 0.773 / 0.773 | 3,578 / 412 | 0 / 3,578 |
| 24-question shared state, System-One | 14/24 | 0 / 0 | 24 / 63 | 229.446 s | 7.419 / 14.399 s | 0.105 / 0.275 | 318,618 / 63 | 201,600 / 117,018 |
| 24-question shared state, generation | 0/24 | 24 / 0 | 24 / 24 | 95.798 s | 4.012 / 4.129 s | 0.251 / 0.251 | 121,535 / 251 | 76,800 / 44,735 |

The generation comparator's low strict accuracy is mainly a response-contract
failure: it produced reasoning text instead of exactly one semantic key in 59
of 60 cases. There was no retry or answer repair. System-One's only independent
corpus miss was `binary-03`. Shared-state accuracy reflects this model and
prompt on the synthetic workload; it is not a general accuracy estimate.

Across both workloads, System-One made 175 physical requests for 60 logical
questions and decoded 175 tokens. Generation made 60 physical requests and
decoded 663 tokens. System-One decoded 488 fewer tokens (73.6%), but took
352.127 seconds versus 142.389 seconds (2.47 times) and submitted 328,754
prompt tokens versus 125,113 (2.63 times). This is the main cost of the
N-singleton compatibility protocol.

Engine counters agreed with response usage. System-One produced no speculative
draft activity. Generation recorded 576 draft tokens and 352 accepted draft
tokens across both workloads. Per-response cached-token usage was absent, so
cache findings below come from raw engine counter deltas.

## Shared-prefix cache conditions

The controlled cache run issued 44 logical questions and 88 physical singleton
requests. State A contained 4,954 tokens, changed state A contained 4,955, and
unrelated state B contained 5,008. Their SHA-256 hashes were respectively
`3c43c6b97cece95bed7f5b47a97a66cb06ba1a4d800a86b2698560d94696b774`,
`66971c646ec4bc247b6e4eb83e187d60168a5d32adce9960a8fb0293c865b7bb`,
and `c5993981c9ad80a2b65d8373e061a62996ab40ca4b1b9668e6c18f56bbcb39c3`.

| Condition | Logical / physical requests | Wall time | Query tokens | Cache-hit tokens | Local-compute tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold state A | 1 / 2 | 21.968 s | 10,088 | 0 | 10,088 |
| 20 suffixes over state A | 20 / 40 | 153.973 s | 201,760 | 128,000 | 73,760 |
| 20 suffixes over changed state A | 20 / 40 | 155.699 s | 201,800 | 128,000 | 73,800 |
| Identical question repeated twice | 2 / 4 | 15.404 s | 20,176 | 12,800 | 7,376 |
| Similar-length unrelated state B | 1 / 2 | 18.329 s | 10,196 | 0 | 10,196 |

The shared and repeated conditions reused exactly 3,200 tokens per physical
request. Changed state A altered a field late in canonical JSON, so the result
shows partial reuse of unchanged leading bytes, not reuse of changed content.
Cold state A and unrelated state B had no observed hit. Latency alone was not
used as cache evidence.

In the separate concurrency-one shared-state comparison, the first System-One
case took 7.076 seconds and later cases had a 7.481-second median. The first
generation case took 3.839 seconds and later cases had a 4.028-second median.
This run did not show a useful cold latency penalty beyond ordinary variation.

## Known-answer and concurrency gates

The five-case live sanity set selected all expected answers and used 15
physical requests in 23.087 seconds. It covered binary, three-way, and five-way
choices and returned finite distributions that summed to one.

Concurrency two was tested only on the 24-question shared-state workload. It
had zero request errors, engine error increments, preemptions, restarts, OOMs,
or malformed scores, but it failed the correctness gate:

| Method | Level 1 wall / median / p95 | Level 2 wall / median / p95 | Level 1 to 2 correctness |
| --- | ---: | ---: | ---: |
| System-One | 229.446 / 7.419 / 14.399 s | 228.276 / 14.841 / 28.641 s | 14/24 to 12/24 |
| Generation | 95.798 / 4.012 / 4.129 s | 95.725 / 7.985 / 8.126 s | 0/24 to 0/24 |

`doors-01` changed from the correct `closed` score 0.6773 to `open` score
0.6323. `doors-03` changed from the correct `unknown` score 0.8884 to `open`
score 0.6985. Method order was counterbalanced, so the run does not isolate
queue concurrency from execution-order or cache-state effects. The acceptance
rule rejects any correctness change. Concurrency two was rejected; its corpus
half and levels four and eight were not run. The production default remains
one.

## Raw evidence

The host-local raw JSON contains every per-case input hash, answer, score,
latency, usage record, counter snapshot, memory snapshot, and container
snapshot. The hashes below are the checked sources for this document.

| Artifact | SHA-256 |
| --- | --- |
| `level-1-corpus-20260919T064810.782129Z.json` | `ad02d619fc8e2cd57342c3010f72fa6d5dfdf7067b18b32673eb525f2b4be400` |
| `level-1-home-20260919T065118.870818Z.json` | `b25a20333ec968303730b0fc365c58bacd4a53bc703952fc7086977ffc30e8bb` |
| `level-2-home-20260919T065703.384140Z.json` | `f905ddfc9a80ca7e1f9f01c2455fec8320edd9eeedce0e43fdecc037687fb117` |
| `cache-summary-20260919T055715.615356Z.json` | `d2d7f583a713dedfb643c6d57618b8f5ecf6589415d57f99da5b5e9d0a4f9159` |
| `known-answer-results-20260919T055641.914654Z.json` | `ca118bf3d1871417746cb677122b354a5f32e49889fcdc7084de465c107a652b` |
| `post-check-20260919T070309.752403Z.json` | `e9e82a25bec59d17f4b2a2996962650bd61a2e8895d2aed401267be00261fdbd` |

The raw artifacts remain outside Git under
`/home/atheartengineer/r9v-systemone-artifacts/`. Benchmark inputs are the
committed `benchmarks/corpus.json` and `benchmarks/home_state.json` files.

## Limits and recommendation

- Scores are candidate-normalized model scores, not calibrated correctness
  probabilities. Temperature calibration is optional and family-specific; no
  calibration was applied in these runs.
- Noul returns the finite candidate probability of `true`. Score returns the
  expected index across 2 to 10 finite anchors. Neither is a continuous latent
  score, a hosted Jev confidence value, or hosted Jev numerical parity.
- The single machine, model, quantization, prompt family, and short corpus
  limit external validity. Generation was judged by strict exact-key parsing.
- The compatibility scorer saves decode work but repeats prompt prefill once
  per candidate. It does not eliminate autoregressive decoding.

The measurements justify designing, but not yet implementing, an engine-level
prompt-final-position `score_token_ids` endpoint. It should prefill once,
gather bounded candidate log probabilities from one final-position logit row,
and sample no token. It needs a separate before/after benchmark and must not
reuse the current AMD custom-token kernel until that kernel is repaired and
independently proved safe. The retained API-only implementation remains the
recommended production path at concurrency one.
