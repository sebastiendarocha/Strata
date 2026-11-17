// src/core/peer_channel_common.cpp - the pieces of the peer channel that BOTH processes need:
// the zone's layout, the digest, the refusal names, the state text and the POSIX shared memory
// helpers.
//
// The primary (src/core/peer_process.cpp) creates the zone and spawns the peer; the peer
// (src/core/peer_worker_main.cpp) maps it.  This file has no runtime dependency at all: the peer
// binary is built with the OTHER vendor's runtime (CUDA and HIP cannot live in the same build:
// CMakeLists.txt's guard), so the two processes share nothing but this layout.
#include "strata/core/peer_channel.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::core {

namespace {
constexpr int64_t kAlign = 256;
inline int64_t alignup(int64_t v) { return (v + kAlign - 1) & ~(kAlign - 1); }
}  // namespace

const char* peer_status_name(int32_t status) {
    switch (status) {
        case kPeerStOk: return "ok";
        case kPeerStFormat: return "the per-layer format does not match the peer's own layout";
        case kPeerStEpoch: return "the residence epoch moved (a swap landed)";
        case kPeerStResidency: return "an expert planned for the peer is not resident";
        case kPeerStGate: return "the landing gate of this layer is not open";
        case kPeerStPlan: return "the plan's digest does not match";
        default: return "unknown";
    }
}

PeerZoneLayout peer_zone_layout(int64_t H, int64_t n_tok, int64_t k, int64_t n_expert, int64_t n_layers) {
    const int64_t n_entries = n_tok * k;
    PeerZoneLayout L{};
    L.off_x = alignup((int64_t) sizeof(PeerZoneHeader));
    L.off_ids = alignup(L.off_x + n_tok * H * 4);
    L.off_kind = alignup(L.off_ids + n_entries * 4);
    L.off_plan = alignup(L.off_kind + n_entries * 4);
    L.off_fmt = alignup(L.off_plan + n_entries * (int64_t) sizeof(PeerPlanEntry));
    L.off_res = alignup(L.off_fmt + n_layers * (int64_t) sizeof(PeerLayerFmt));
    L.off_gate = alignup(L.off_res + n_layers * n_expert * 4);
    L.off_out = alignup(L.off_gate + n_layers * (int64_t) sizeof(PeerGateState));
    L.total = alignup(L.off_out + n_entries * H * 4);
    return L;
}

uint64_t peer_fnv(const void* data, size_t bytes, uint64_t seed) {
    const uint8_t* p = (const uint8_t*) data;
    uint64_t h = seed;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ------------------------------------------------------------------ the peer's own state (text)
// The peer reads its own state from the STRATA_PEER_STATE environment variable the primary sets
// at spawn time.  In the full engine the peer reads it from the pack it opens; in tranche 1 (the
// transport only, no model) the primary hands it over, so the two views can disagree on purpose
// and the refusal path is exercised.
std::string peer_state_serialize(const PeerState& state) {
    std::ostringstream os;
    os << "epoch " << state.epoch << "\n";
    for (size_t l = 0; l < state.formats.size(); ++l) {
        const PeerLayerFmt& f = state.formats[l];
        os << "fmt " << l << " " << f.gu_type << " " << f.d_type << " " << f.n_embd << " " << f.n_ff << " "
           << f.had_block << " " << f.had_gu << " " << f.had_d << " " << f.signs_digest << "\n";
    }
    for (size_t l = 0; l < state.resident.size(); ++l) {
        os << "resident " << l;
        for (int32_t e : state.resident[l]) os << " " << e;
        os << "\n";
    }
    for (size_t l = 0; l < state.gates.size(); ++l) {
        const PeerGateState& g = state.gates[l];
        os << "gate " << l << " " << g.landed << " " << g.has << " " << g.pending << " " << g.last << "\n";
    }
    return os.str();
}

bool peer_state_parse(const std::string& text, PeerState& state, std::string& err) {
    std::istringstream is(text);
    std::string line;
    int lineno = 0;
    while (std::getline(is, line)) {
        ++lineno;
        std::istringstream ls(line);
        std::string what;
        ls >> what;
        if (what.empty() || what[0] == '#') continue;
        if (what == "epoch") {
            ls >> state.epoch;
        } else if (what == "fmt") {
            size_t l = 0;
            PeerLayerFmt f{};
            ls >> l >> f.gu_type >> f.d_type >> f.n_embd >> f.n_ff >> f.had_block >> f.had_gu >> f.had_d >>
                f.signs_digest;
            if (state.formats.size() <= l) state.formats.resize(l + 1);
            state.formats[l] = f;
        } else if (what == "resident") {
            size_t l = 0;
            ls >> l;
            if (state.resident.size() <= l) state.resident.resize(l + 1);
            int32_t e = 0;
            while (ls >> e) state.resident[l].push_back(e);
        } else if (what == "gate") {
            size_t l = 0;
            PeerGateState g{};
            ls >> l >> g.landed >> g.has >> g.pending >> g.last;
            if (state.gates.size() <= l) state.gates.resize(l + 1);
            state.gates[l] = g;
        } else {
            err = "peer state: unknown directive '" + what + "' on line " + std::to_string(lineno);
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------ the zone (POSIX shared memory)
// A POSIX shared memory object (tmpfs), NEVER a file mapping: cudaHostRegister refuses a
// file-backed mapping ("invalid argument") and accepts the tmpfs pages of a shm segment.
#if !defined(_WIN32)

int peer_zone_create(const std::string& name, size_t bytes, std::string& err) {
    shm_unlink(name.c_str());
    const int fd = shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        err = std::string("shm_open(") + name + ") failed: " + std::strerror(errno);
        return -1;
    }
    if (ftruncate(fd, (off_t) bytes) != 0) {
        err = std::string("ftruncate(") + name + ") failed: " + std::strerror(errno);
        close(fd);
        shm_unlink(name.c_str());
        return -1;
    }
    return fd;
}

int peer_zone_open(const std::string& name, std::string& err, int wait_ms) {
    for (int waited = 0;; waited += 10) {
        const int fd = shm_open(name.c_str(), O_RDWR, 0600);
        if (fd >= 0) return fd;
        if (waited >= wait_ms) {
            err = std::string("shm_open(") + name + ") failed: " + std::strerror(errno);
            return -1;
        }
        usleep(10000);
    }
}

void* peer_zone_map(int fd, size_t bytes, std::string& err) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        err = std::string("mmap failed: ") + std::strerror(errno);
        return nullptr;
    }
    return p;
}

void peer_zone_unmap(void* p, size_t bytes) {
    if (p != nullptr && bytes != 0) munmap(p, bytes);
}

void peer_zone_destroy(const std::string& name) { shm_unlink(name.c_str()); }

#else   // _WIN32: the peer channel is Linux only (POSIX shared memory + posix_spawnp)

int peer_zone_create(const std::string&, size_t, std::string& err) {
    err = "the peer channel is Linux only (POSIX shared memory)";
    return -1;
}
int peer_zone_open(const std::string&, std::string& err, int) {
    err = "the peer channel is Linux only (POSIX shared memory)";
    return -1;
}
void* peer_zone_map(int, size_t, std::string& err) {
    err = "the peer channel is Linux only (POSIX shared memory)";
    return nullptr;
}
void peer_zone_unmap(void*, size_t) {}
void peer_zone_destroy(const std::string&) {}

#endif

}  // namespace strata::core
