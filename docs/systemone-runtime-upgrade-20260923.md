# System-One runtime upgrade, 2026-09-23

Upstream source `9d5026c8c2ddc0a69433131176b7873d58edefad` is already
integrated. Its vLLM pin is `9ec060c402ed15fbeb1dbd2714615261b63dda12`.
This deployment changes the running image, not the upstream source pin.

- Candidate: `sha256:2dac17a215fb5b0e3461e4c3e36a2981eec8ac3d6021e73183d247e819740c03`.
- Release manifest: `release/image-bundle-wmma-prefill-20260915.json`.
- Rollback image: `sha256:580bbcecccb3f03907471ef1407506f4c84efbcf6597a77e31311e5e726754b4`.
- Profile: `profiles/qwen38-flash-next/dual-r9700-systemone/profile.env`.

Retain the existing weights, 256K context, MTP2, prefix caching, one sequence,
expert placement, KV allocation and 1024-token chunks. Select group 32 to use
the new WMMA prefill kernel. The published upstream qualification covers a
different 128K/MTP4 profile, so it does not qualify this configuration.

Deployment gates: hash-verified image import; profile and launch checks;
successful model startup; ordinary chat and System-One inference; comparison
with the retained chat-speed fixture. Do not probe `logprob_token_ids`: this
release does not establish a fix for the known AMD compiler failure.

## Status

Profile contract passed; release/image-loader tests: 27 passed. Deployment
profile committed to fork main. Nix service unit builds independently, but
the full system build failed in unrelated `open-bambu-networking`.

Image transfer is incomplete because of repeated release-host download stalls.
The current production image is unchanged; no restart or qualification of the
candidate has occurred. See `TODO-runtime-upgrade.md` for remaining gates.
