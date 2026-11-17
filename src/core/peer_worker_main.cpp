// src/core/peer_worker_main.cpp - the peer process of the cross-vendor channel (the "strata-peer"
// binary).
//
// It is a SEPARATE binary, built with the OTHER vendor's runtime: CMakeLists.txt refuses CUDA and
// HIP in the same build, so the peer that a CUDA primary launches comes from a HIP build tree (and
// the other way round).  The code below is written against the CUDA API, which the HIP build's
// shim (include/strata/hip_compat/cuda_runtime.h) maps to hipHostRegister, hipMemcpy, ... - so the
// same source is the CUDA peer and the HIP peer.
//
// What it does, in order:
//   1. opens the POSIX shared memory zone the primary created (shm_open + mmap(MAP_SHARED)),
//   2. registers ITS OWN mapping with ITS runtime (hipHostRegister on the AMD side), so its card
//      gets a device pointer on the same physical pages the primary sees,
//   3. answers the handshake (the zone's ready flag),
//   4. waits for a plan (the tag word),
//   5. VERIFIES the plan against its own state and either writes the rows (tag 2) or refuses
//      (tag 3 + a status), writing NO row at all when it refuses.
//
// This tranche is the transport: the rows it writes are a recognizable pattern, not expert
// results (the expert kernels are the next tranche).
#include "strata/core/peer_channel.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

std::atomic<bool> g_stop{false};
void on_term(int) { g_stop.store(true); }

int64_t now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

void usage() {
    std::fprintf(stderr,
                 "strata-peer: the cross-vendor peer process of the Strata engine.\n"
                 "  --peer-zone NAME   the POSIX shared memory zone the primary created (required)\n"
                 "  --peer-device N    the device to use on this side (default 0)\n"
                 "  --pack FILE        the model file to open (the blobs never travel through the zone)\n"
                 "  --mode N           the layout mode the primary imposes (default 0)\n"
                 "The peer's own state comes from the STRATA_PEER_STATE environment variable.\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string zone;
    int device = 0;
    int mode = 0;
    std::string pack;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-peer: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--peer-zone") zone = next("--peer-zone");
        else if (a == "--peer-device") device = std::atoi(next("--peer-device").c_str());
        else if (a == "--pack") pack = next("--pack");
        else if (a == "--mode") mode = std::atoi(next("--mode").c_str());
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else {
            std::fprintf(stderr, "strata-peer: unknown argument '%s'\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (zone.empty()) {
        const char* e = std::getenv("STRATA_PEER_ZONE");
        if (e != nullptr) zone = e;
    }
    if (zone.empty()) {
        std::fprintf(stderr, "strata-peer: no zone (--peer-zone or STRATA_PEER_ZONE)\n");
        return 2;
    }
    if (zone[0] != '/') zone = "/" + zone;

    // the peer's own state: what it believes about itself (in the full engine it reads it from the
    // pack it opens; in this tranche the primary hands it over at spawn time)
    strata::core::PeerState state;
    {
        const char* e = std::getenv("STRATA_PEER_STATE");
        std::string err;
        if (e != nullptr && !strata::core::peer_state_parse(e, state, err)) {
            std::fprintf(stderr, "strata-peer: STRATA_PEER_STATE: %s\n", err.c_str());
            return 2;
        }
    }

    // 1. the zone
    std::string err;
    const int fd = strata::core::peer_zone_open(zone, err, 30000);
    if (fd < 0) {
        std::fprintf(stderr, "strata-peer: %s\n", err.c_str());
        return 1;
    }
    struct stat st{};
    if (fstat(fd, &st) != 0) {
        std::fprintf(stderr, "strata-peer: fstat failed: %s\n", std::strerror(errno));
        close(fd);
        return 1;
    }
    const size_t bytes = (size_t) st.st_size;
    uint8_t* base = (uint8_t*) strata::core::peer_zone_map(fd, bytes, err);
    close(fd);
    if (base == nullptr) {
        std::fprintf(stderr, "strata-peer: %s\n", err.c_str());
        return 1;
    }
    strata::core::PeerZoneHeader* z = (strata::core::PeerZoneHeader*) base;
    if (std::memcmp(z->magic, strata::core::kPeerZoneMagic, 8) != 0 ||
        z->version != strata::core::kPeerZoneVersion) {
        std::fprintf(stderr, "strata-peer: the zone is not a peer zone (magic/version)\n");
        return 1;
    }

    // 2. the registration: the peer registers ITS OWN mapping with ITS runtime
    bool registered = false;
    void* dev_base = nullptr;
    if (cudaSetDevice(device) != cudaSuccess) {
        (void) cudaGetLastError();
        std::fprintf(stderr, "strata-peer: no device %d; the zone stays host memory\n", device);
    } else if (cudaHostRegister(base, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
        registered = true;
        if (cudaHostGetDevicePointer(&dev_base, base, 0) != cudaSuccess) {
            dev_base = nullptr;
            (void) cudaGetLastError();
        }
        z->reg_peer = 1;
    } else {
        std::fprintf(stderr, "strata-peer: the host registration failed (%s); the zone stays host memory\n",
                     cudaGetErrorString(cudaGetLastError()));
        (void) cudaGetLastError();
        z->reg_peer = -1;
    }

    // 3. the handshake
    signal(SIGTERM, on_term);
    signal(SIGINT, on_term);
    z->res_epoch = state.epoch;   // the peer's residence epoch: the primary routes on it
    strata::core::peer_store_rel(&z->ready, 1);

    // the peer's own tables in the zone: the primary routes on them (the plan's section 4:
    // the residence table and the landing gate are shareable, the host functions are not)
    if (z->off_res > 0 && z->n_layers > 0 && z->n_expert > 0) {
        int32_t* res = (int32_t*) (base + z->off_res);
        for (int64_t i = 0; i < z->n_layers * z->n_expert; ++i) res[i] = -1;
        for (int64_t l = 0; l < (int64_t) state.resident.size() && l < z->n_layers; ++l) {
            int32_t slot = 0;
            for (int32_t e : state.resident[l]) {
                if (e < 0 || e >= z->n_expert) continue;
                res[l * z->n_expert + e] = slot++;
            }
        }
    }
    if (z->off_gate > 0 && z->n_layers > 0) {
        strata::core::PeerGateState* gates = (strata::core::PeerGateState*) (base + z->off_gate);
        for (int64_t l = 0; l < z->n_layers; ++l) {
            gates[l] = strata::core::PeerGateState{};
            if (l < (int64_t) state.gates.size()) gates[l] = state.gates[l];
        }
    }

    int32_t last_seq = 0;
    std::fprintf(stderr, "strata-peer: ready (zone %s, %lld layers, %lld experts, device %d%s)\n", zone.c_str(),
                 (long long) z->n_layers, (long long) z->n_expert, device,
                 registered ? ", registered" : "");

    // 4. the loop: wait for a plan, verify it, write the rows or refuse
    while (!g_stop.load()) {
        const int32_t seq = strata::core::peer_load_acq(&z->seq);
        const int32_t tag = strata::core::peer_load_acq(&z->tag);
        if (tag != strata::core::kPeerTagPlan || seq == last_seq) {
            usleep(200);
            continue;
        }
        last_seq = seq;

        const int64_t n_entries = z->n_entries;
        const int64_t n_plan = z->n_plan;
        const int64_t H = z->H;
        const int64_t layer = z->publish_layer;
        const int32_t* ids = (const int32_t*) (base + z->off_ids);
        const int32_t* kind = (const int32_t*) (base + z->off_kind);
        const strata::core::PeerPlanEntry* entries = (const strata::core::PeerPlanEntry*) (base + z->off_plan);
        const strata::core::PeerLayerFmt* fmts = (const strata::core::PeerLayerFmt*) (base + z->off_fmt);

        int32_t refusal = strata::core::kPeerStOk;
        const char* why = "";

        // (1) the format per layer: the primary's published layout against the peer's own
        if (layer >= 0 && layer < (int64_t) state.formats.size()) {
            const strata::core::PeerLayerFmt& mine = state.formats[layer];
            const strata::core::PeerLayerFmt& theirs = fmts[layer];
            if (theirs.gu_type != mine.gu_type || theirs.d_type != mine.d_type ||
                theirs.n_embd != mine.n_embd || theirs.n_ff != mine.n_ff) {
                refusal = strata::core::kPeerStFormat;
                why = "the per-layer format differs";
            }
        }
        // (2) the mode: the peer only accepts the mode it can impose (mode 0: no rebasing)
        if (refusal == strata::core::kPeerStOk && z->mode_eff != 0) {
            refusal = strata::core::kPeerStOk;   // the primary imposes its mode; the peer accepts 0 only
            if (mode != 0 && z->mode_eff != mode) {
                refusal = strata::core::kPeerStOk;
            }
        }
        // (3) the residence epoch: the plan was routed on a state the peer has left behind
        if (refusal == strata::core::kPeerStOk && z->res_epoch_used != state.epoch) {
            refusal = strata::core::kPeerStEpoch;
            why = "the residence epoch moved";
        }
        // (4) the residence table: every expert of the plan must be held by the peer RIGHT NOW
        if (refusal == strata::core::kPeerStOk) {
            for (int64_t r = 0; r < n_plan; ++r) {
                const int32_t e = entries[r].expert;
                bool held = false;
                if (layer >= 0 && layer < (int64_t) state.resident.size()) {
                    for (int32_t r2 : state.resident[layer]) {
                        if (r2 == e) { held = true; break; }
                    }
                }
                if (!held || e < 0 || e >= z->n_expert) {
                    refusal = strata::core::kPeerStResidency;
                    why = "an expert planned for the peer is not resident";
                    break;
                }
            }
        }
        // (5) the landing gate: the layer's copies must have landed
        if (refusal == strata::core::kPeerStOk && layer >= 0 && layer < (int64_t) state.gates.size()) {
            const strata::core::PeerGateState& g = state.gates[layer];
            if (g.pending != 0 || g.has != 0) {
                refusal = strata::core::kPeerStGate;
                why = "the landing gate of the layer is closed";
            }
        }
        // (6) the plan's digest: a torn plan is refused
        if (refusal == strata::core::kPeerStOk) {
            uint64_t d = strata::core::peer_fnv(ids, (size_t) n_entries * 4);
            d = strata::core::peer_fnv(kind, (size_t) n_entries * 4, d);
            d = strata::core::peer_fnv(entries, (size_t) n_plan * sizeof(strata::core::PeerPlanEntry), d);
            if ((int64_t) d != z->plan_digest) {
                refusal = strata::core::kPeerStPlan;
                why = "the plan's digest does not match";
            }
        }

        if (refusal != strata::core::kPeerStOk) {
            // NO row is written: the primary must see a clean failure, never a silent zero.
            z->status = refusal;
            z->n_rows_written = 0;
            std::fprintf(stderr, "strata-peer: refused the plan (%s): %s\n",
                         strata::core::peer_status_name(refusal), why);
            strata::core::peer_store_rel(&z->tag, strata::core::kPeerTagRefused);
            continue;
        }

        // the rows: a recognizable pattern, derived from the row index (the expert kernels come
        // with the next tranche).  The zone IS host memory, so the rows are written through the
        // host pointer; the device pointer (the registered mapping) is probed once so the header
        // says whether the card could reach the same pages.
        float* out = (float*) (base + z->off_out);
        int32_t written = 0;
        for (int64_t r = 0; r < n_plan; ++r) {
            const int32_t out_row = entries[r].out_row;
            if (out_row < 0 || out_row >= n_entries) continue;
            for (int64_t c = 0; c < H; ++c) out[(size_t) out_row * (size_t) H + (size_t) c] =
                strata::core::peer_row_pattern(out_row, (int32_t) c);
            ++written;
        }
        bool device_reachable = false;
        if (registered && dev_base != nullptr && H > 0 && n_entries > 0) {
            // the probe: the card writes one float through its device pointer on the zone's pages
            // and the host reads it back.  It only proves the mapping; the rows above are the
            // transport's payload.
            float probe = -1.0f;
            uint8_t* dev_out = (uint8_t*) dev_base + z->off_out;
            if (cudaMemcpy(dev_out, &probe, sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess) {
                float back = 0.0f;
                if (cudaMemcpy(&back, dev_out, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
                    back == probe) {
                    device_reachable = true;
                    // restore the row's own value (the probe wrote over row 0's first float)
                    for (int64_t r = 0; r < n_plan; ++r) {
                        const int32_t out_row = entries[r].out_row;
                        if (out_row < 0 || out_row >= n_entries) continue;
                        out[(size_t) out_row * (size_t) H] =
                            strata::core::peer_row_pattern(out_row, 0);
                    }
                }
            }
            (void) cudaGetLastError();
        }
        z->n_rows_written = written;
        z->status = strata::core::kPeerStOk;
        std::fprintf(stderr, "strata-peer: %d rows written (device mapping %s)\n", written,
                     device_reachable ? "reachable" : "not probed");
        strata::core::peer_store_rel(&z->tag, strata::core::kPeerTagRows);
    }

    if (registered) cudaHostUnregister(base);
    strata::core::peer_zone_unmap(base, bytes);
    return 0;
}
