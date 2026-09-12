# Qwen3.8 Flash Next Q4_K_XL candidate

This package references the original Unsloth `UD-Q4_K_XL` GGUF release at revision `2c41bd2a0b3f51c503c11f1c7ed2e6bb34036beb`, with exact target shard lengths and SHA-256 hashes in `package.json`. Source: https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/2c41bd2a0b3f51c503c11f1c7ed2e6bb34036beb/UD-Q4_K_XL.

R9V does not generate or requantize these target weights. The original Q4 target head is Q8_0. The mixed expert tensors include Q4_K, Q5_K, Q5_1 and Q8_0; the placement catalog accounts for each layer and TP rank separately.

The Qwen model license, tokenizer/configuration, block-FP8 MTP auxiliary model and Q8 vision projector are pinned separately to the existing R9V auxiliary package revision `bf836f0c20b6c92fcad4226ad3115eb8a19f7582`. Each artifact identifies its own repository and revision. The Qwen Community License applies to model weights independently of R9V's Apache-2.0 code license. Read the included model LICENSE before download/use.

The PLE tensor is unchanged between these pinned packages: its complete 28,800,138,240-byte IQ4_NL payload hashes to `dd55c28902f38cd88134b2a569c51282c5ffce30080487e1a645740115c56cc3`. An existing verified extraction can be reused. Its location inside this Q4 shard is distinct from the IQ4 shard; resolve tensor offsets through GGUF metadata.

This is an experimental package profile. Kernel parity is necessary but does not qualify full-model startup, context, memory headroom or throughput. The bootstrap expert placement is explicitly unranked. See `docs/qwen-release-candidate.md` for integration status and the support flow.
