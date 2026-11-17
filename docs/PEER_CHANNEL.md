# The cross-vendor peer channel

This page describes the **channel** that lets a Strata engine (the *primary*) hand work to a peer
process that runs on **another vendor's card** (the *peer*): an NVIDIA engine driving an AMD card, or
the other way round.

**This is tranche 1: the transport only.** The channel exists, the protocol is real, the refusals are
real and tested end to end. What the peer computes is **not** in this tranche: the peer writes a
recognizable pattern in the rows it is given (derived from the row index), not expert results. The
expert kernels are the next tranche; the channel does not change when they arrive.

Everything here is described in the code that implements it:

| What | Where |
| --- | --- |
| the zone's layout, the tags, the refusal codes, the digest | `include/strata/core/peer_channel.hpp` |
| the layout, the status names, the state text, the POSIX shared memory helpers | `src/core/peer_channel_common.cpp` |
| the primary's side: the zone, the spawn, the plan publication, the answer | `src/core/peer_process.cpp` |
| the peer process: the mapping, the registration, the handshake, the checks | `src/core/peer_worker_main.cpp` |
| the end-to-end test | `tests/core/peer_channel_test.cpp` |

## Why a separate process, and why not `fork`

The two cards belong to two different vendors, so they cannot be in the same process: the CUDA
runtime and the HIP runtime cannot be linked into one binary. `CMakeLists.txt` refuses CUDA and HIP
in the same build for the same reason.

So the peer is a **separate binary** (`strata-peer`), built in the *other* backend's build tree. The
source is written against the CUDA API; the HIP build's shim
(`include/strata/hip_compat/cuda_runtime.h`) renames those calls to their HIP names, so the same
source is both the CUDA peer and the AMD peer.

The primary **spawns** the peer with `posix_spawnp`, never `fork`: by the time a plan is published
the engine has already initialised CUDA (`src/generate.cpp`, `src/core/peer_experts.cpp`) and runs
several threads, and `fork` in that state is undefined behaviour. `posix_spawnp` starts a fresh
process from an image that has nothing to do with the engine's runtime state.

### Where the peer binary is

In this order:

1. the `STRATA_PEER_BINARY` environment variable (an absolute path),
2. else the `strata-peer` **beside the running engine**, resolved through `/proc/self/exe` (the
   pattern `src/prefill/mmq_kq_loader.cpp` already uses).

So a CUDA engine finds the AMD peer if the AMD peer is installed next to it (the packaging ships it:
`packaging/package_engine.py` puts `strata-peer` in the package), and `STRATA_PEER_BINARY` overrides
that when the peers live in different folders.

## The zone

The zone is a **POSIX shared memory segment** (`shm_open` + `ftruncate` + `mmap(MAP_SHARED)`),
**never a file mapping**: `cudaHostRegister` refuses a file-backed mapping ("invalid argument") and
accepts the tmpfs pages of a shared memory segment. That was measured on the prototype
(`/home/hermes/e2/perf/peer_worker_v18.cpp`).

The primary owns the segment: it creates it, sizes it and unlinks it when it is done. The peer opens
it by name.

**Both processes register their own mapping with their own runtime**: the primary with
`cudaHostRegister` (or `hipHostRegister` on an AMD primary), the peer with its own. Each card then
gets its own device pointer on the same physical pages.

**The expert weights never travel through the zone.** The peer reopens the same files read-only; the
zone carries the window's activations, the routing and the results.

### The layout

The header is at offset 0 and carries every offset, so both sides compute the same layout from the
same numbers (`peer_zone_layout`):

```
header | x (n_tok * H floats) | ids (n_entries int32) | kind (n_entries int32)
       | plan (n_plan entries) | fmt (n_layers formats) | res (n_layers * n_expert int32)
       | gate (n_layers gates) | out (n_entries * H floats)
```

Every section starts on a 256-byte boundary.

## The protocol

One word (the header's `tag`) is the whole synchronisation:

| Tag | Meaning |
| --- | --- |
| `0` | idle: nothing published |
| `1` | the primary published a plan; the peer must verify it |
| `2` | the peer wrote its rows |
| `3` | the peer **refused** the plan; `header.status` says why |

The sequence:

1. The primary creates the zone, fills the header and registers its mapping.
2. The primary spawns the peer (`posix_spawnp`) with `--peer-zone <name>`, `--peer-device <n>`,
   `--mode <n>`, optionally `--pack <file>`. The peer's own state travels in the
   `STRATA_PEER_STATE` environment variable (in this tranche; the full engine reads it from the pack
   it opens).
3. The peer maps the zone, registers its mapping with its runtime and **publishes its own tables**
   (the residence table, the landing gate) and its residence epoch, then answers the handshake
   (`header.ready = 1`). The primary waits for it.
4. The primary publishes a plan: it fills `ids`, `kind`, the plan entries and the per-layer format
   table, stamps the plan's digest and raises the tag to `1`.
5. The peer **verifies the plan against its own state** and either writes its rows (tag `2`) or
   refuses (tag `3` + a status), writing **no row at all** when it refuses.

### The refusals

The peer refuses (and writes nothing) when the plan does not match its own state:

| Status | Name | When |
| --- | --- | --- |
| `2` | format | the per-layer format the primary published is not the peer's own (a different `n_embd`, `n_ff`, ggml type, APR flags or signs digest) |
| `3` | epoch | the residence epoch moved since the routing (a swap landed on the peer) |
| `4` | residency | an expert the plan routes to the peer is not resident on the peer |
| `5` | gate | the landing gate of that layer is closed (copies still in flight) |
| `6` | plan | the plan's digest does not match the plan (a torn plan) |

The status is read back by the primary (`await_result`). A refusal is a clean failure: the primary
never sees a silent zero where it expected rows.

## Running the peer by hand

```sh
# the peer alone (the primary creates the zone; this is only for a manual run)
strata-peer --peer-zone /strata_peer_1234_5678 --peer-device 0 --mode 0
```

It answers with a line on stderr:

```
strata-peer: ready (zone /strata_peer_1234_5678, 32 layers, 128 experts, device 0, registered)
```

and, for every plan:

```
strata-peer: 64 rows written (device mapping reachable)
strata-peer: refused the plan (the residence epoch moved): the residence epoch moved
```

Without `--peer-zone` it reads `STRATA_PEER_ZONE`. Without a usable device (no card, or no VRAM)
the peer says so and keeps the zone in host memory: the transport still works, the card does not
accelerate it.

## The test

```sh
ctest -R peer_channel --output-on-failure
# or directly:
./peer_channel_test /path/to/strata-peer
```

The test needs **no model and no card**. It:

- spawns the peer, waits for the handshake, checks the zone is established,
- publishes a plan, waits for the answer and reads the rows back: each row `r` holds
  `777 + r * 1000 + c` in column `c`, so the primary can tell the peer's rows from its own,
- triggers **five** refusals (format, epoch, residency, gate, digest) and checks each returned
  status, and that **no row was written** in any of them.

To run the test against the peer built with the *other* backend:

```sh
STRATA_PEER_BINARY=/path/to/the/other/build/strata-peer ./peer_channel_test
```

## What this tranche does not do

- The peer does not compute experts: it writes a pattern.
- The peer does not open the model: it receives its state through the environment for now.
- The primary does not route on the peer's tables yet: the tables are published, the checks are
  enforced by the peer, but the routing decision is still the engine's own (the plan's section 4 of
  the design is the next step).
- The channel is Linux only (POSIX shared memory and `posix_spawnp`). On the other platforms every
  entry point fails cleanly and says so.
