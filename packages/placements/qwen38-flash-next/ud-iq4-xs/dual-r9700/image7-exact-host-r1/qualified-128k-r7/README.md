# Image7 IQ4 reference placement

Experimental reference for the exact-sized cold-host allocator. The measured 71/450 static-expert placement with cache160/0 passed all seven workload checks at 131,072 context, including a 130,941-token prompt and at least 3 GiB free on each GPU. Reference median generation was 89.45 tokens/s; this is not a BetterBench result.

The public seed provides numeric measurements and commitments to independently preserved private evidence. Only provenance locations and the catalog link are projected; model tensors, runtime settings, expert ordering, counts, and memory arithmetic are unchanged. Every local placement must qualify through setup/start before its receipt is reusable.

Source implementation: `123f1ad03c51e2b630d3294292e80a567087ec81`. Image: `sha256:46ab688af195643e61322a72b4e7b7fa0999c12299bffb2a4515f8363c59393c`.

Private workload archive SHA256: `6de09f3662f08c810beb86c4e398eeb66afb002f2c931e3775968ed87c91499f`; complete compiled-cache archive: `9142b018fa3b604c9476644469e4d4bbcfd7a903fce9f897685bed5d0dd883a7`. Original supervisor failure flag is preserved: its equality-only cleanup check rejected 185 MiB more free VRAM. Independent review found no per-card shortfall throughout the 90-second aftermath, normal container exit, and no driver fault/reset/OOM.
