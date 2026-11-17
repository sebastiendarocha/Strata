// src/core/peer_process.cpp - the primary's side of the cross-vendor peer channel.
//
// The primary (this file) creates the POSIX shared memory zone, spawns the peer process with
// posix_spawnp (NOT fork: CUDA is already initialised in the engine and the engine is
// multi-threaded), publishes the plans and reads the peer's answer back.
//
// The peer binary is a SEPARATE build (the other vendor's runtime: CUDA and HIP cannot live in the
// same build, CMakeLists.txt's guard).  Its path comes from:
//   1. the STRATA_PEER_BINARY environment variable,
//   2. else the "strata-peer" beside the running engine (/proc/self/exe), the pattern
//      src/prefill/mmq_kq_loader.cpp already uses.
//
// Who owns what in the zone:
//   the PRIMARY publishes  the per-layer format table, the ids/kind arrays, the plan entries, the
//                        epoch it routed on and the plan's digest,
//   the PEER publishes     its residence table, its landing gate and its residence epoch (the
//                        primary routes on them: the plan's section 4).
//
// This tranche is the TRANSPORT only: the peer writes a recognizable pattern in the rows it is
// given (the expert kernels are the next tranche).  The protocol, the zone and the refusals are
// the real thing.
#include "strata/core/peer_channel.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
extern char** environ;
#endif

namespace strata::core {

namespace {

int64_t now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

#if !defined(_WIN32)
/// The directory of the running binary (/proc/self/exe), or "" when it cannot be read.
std::string self_dir() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    buf[n] = '\0';
    std::string p(buf);
    const size_t slash = p.find_last_of('/');
    return slash == std::string::npos ? std::string() : p.substr(0, slash);
}

bool file_exists(const std::string& p) { return access(p.c_str(), X_OK) == 0; }
#endif

}  // namespace

PeerProcess::~PeerProcess() { close(); }

bool PeerProcess::open(int64_t H, int64_t n_tok, int64_t k, int64_t n_expert, int64_t n_layers,
                       const PeerView& view, const PeerState& state, const std::string& binary,
                       std::vector<std::string> extra_args, std::string& err) {
    close();
#if defined(_WIN32)
    (void) H; (void) n_tok; (void) k; (void) n_expert; (void) n_layers; (void) view; (void) state;
    (void) binary; (void) extra_args;
    err = "the peer channel is Linux only (POSIX shared memory + posix_spawnp)";
    return false;
#else
    H_ = H;
    n_tok_ = n_tok;
    k_ = k;
    n_expert_ = n_expert;
    n_layers_ = n_layers;
    n_entries_ = n_tok * k;
    layout_ = peer_zone_layout(H, n_tok, k, n_expert, n_layers);
    view_formats_ = view.formats;

    // 1. the binary
    binary_ = binary;
    if (binary_.empty()) {
        const char* e = std::getenv("STRATA_PEER_BINARY");
        if (e != nullptr && e[0] != '\0') binary_ = e;
    }
    if (binary_.empty()) {
        const std::string dir = self_dir();
        if (dir.empty()) {
            err = "the peer binary: /proc/self/exe is unreadable and STRATA_PEER_BINARY is not set";
            return false;
        }
        binary_ = dir + "/strata-peer";
    }
    if (!file_exists(binary_)) {
        err = "the peer binary " + binary_ + " does not exist or is not executable "
              "(STRATA_PEER_BINARY points at the peer built with the OTHER backend)";
        return false;
    }

    // 2. the zone: a POSIX shared memory segment, allocated by the primary, mapped by the peer
    zone_ = "/strata_peer_" + std::to_string((long) getpid()) + "_" + std::to_string((long) now_ms());
    const int fd = peer_zone_create(zone_, (size_t) layout_.total, err);
    if (fd < 0) return false;
    map_ = peer_zone_map(fd, (size_t) layout_.total, err);
    ::close(fd);
    if (map_ == nullptr) {
        peer_zone_destroy(zone_);
        zone_.clear();
        return false;
    }
    map_bytes_ = (size_t) layout_.total;
    header_ = (PeerZoneHeader*) map_;
    std::memset(header_, 0, sizeof(PeerZoneHeader));
    std::memcpy(header_->magic, kPeerZoneMagic, 8);
    header_->version = kPeerZoneVersion;
    header_->H = (int32_t) H;
    header_->n_tok = (int32_t) n_tok;
    header_->k = (int32_t) k;
    header_->n_entries = (int32_t) n_entries_;
    header_->n_layers = (int32_t) n_layers;
    header_->n_expert = (int32_t) n_expert;
    header_->mode_eff = view.mode_eff;
    header_->grid_groups = view.grid_groups;
    header_->off_fmt = layout_.off_fmt;
    header_->off_res = layout_.off_res;
    header_->off_gate = layout_.off_gate;
    header_->off_plan = layout_.off_plan;
    header_->off_ids = layout_.off_ids;
    header_->off_kind = layout_.off_kind;
    header_->off_out = layout_.off_out;
    header_->off_x = layout_.off_x;
    header_->total_bytes = layout_.total;
    header_->tag = kPeerTagIdle;
    header_->status = kPeerStOk;

    // 3. the primary registers ITS OWN mapping with its runtime (cudaHostRegister here, the peer
    //    does hipHostRegister on the other side).  A file-backed mapping would be refused; a POSIX
    //    shared memory segment is accepted.
    // the registration needs a device context: cudaSetDevice + cudaFree(0) creates it.  A machine
    // without a card (or a card whose VRAM is full) simply keeps the zone in host memory.
    if (cudaSetDevice(0) == cudaSuccess && cudaFree(0) == cudaSuccess &&
        cudaHostRegister(map_, map_bytes_, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess) {
        registered_ = true;
        header_->reg_primary = 1;
    } else {
        registered_ = false;
        header_->reg_primary = -1;
        std::fprintf(stderr, "strata peer: cudaHostRegister failed (%s); the zone stays host memory\n",
                     cudaGetErrorString(cudaGetLastError()));
        (void) cudaGetLastError();
    }

    // 4. spawn the peer: posix_spawnp, never fork (CUDA is initialised, the engine is threaded)
    std::vector<std::string> args;
    args.push_back(binary_);
    args.push_back("--peer-zone");
    args.push_back(zone_);
    args.push_back("--peer-device");
    args.push_back(std::to_string(view.device));
    // NOT --mode and NOT --grid-groups: the peer imposes its OWN mode (the operator configures the
    // peer side, STRATA_PEER_MODE / STRATA_PEER_GRID_GROUPS), and the primary publishes what it will
    // use in the header.  A difference is what the peer refuses (status 1).
    if (!view.model.empty()) {
        args.push_back("--pack");
        args.push_back(view.model);
    }
    for (const std::string& a : extra_args) args.push_back(a);

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    // the child's environment: the parent's, plus the zone and the peer's own state
    std::vector<std::string> envs;
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) envs.push_back(*e);
    envs.push_back("STRATA_PEER_ZONE=" + zone_);
    envs.push_back("STRATA_PEER_STATE=" + peer_state_serialize(state));
    std::vector<char*> envp;
    envp.reserve(envs.size() + 1);
    for (std::string& e : envs) envp.push_back(const_cast<char*>(e.c_str()));
    envp.push_back(nullptr);

    pid_t pid = -1;
    const int rc = posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), envp.data());
    if (rc != 0) {
        err = std::string("posix_spawnp(") + binary_ + ") failed: " + std::strerror(rc);
        close();
        return false;
    }
    pid_ = (int) pid;

    // 5. the handshake: the peer sets ready once it has mapped (and registered) the zone
    const int64_t deadline = now_ms() + 30000;
    while (now_ms() < deadline) {
        if (peer_load_acq(&header_->ready) == 1) return true;
        int wstatus = 0;
        if (waitpid(pid_, &wstatus, WNOHANG) == pid_) {
            err = "the peer exited before the handshake";
            pid_ = -1;
            close();
            return false;
        }
        usleep(2000);
    }
    err = "the peer did not answer the handshake within 30 s";
    close();
    return false;
#endif
}

bool PeerProcess::publish(const PeerPlan& plan, std::string& err, bool corrupt_digest) {
    if (header_ == nullptr) {
        err = "the peer channel is not open";
        return false;
    }
    if ((int64_t) plan.ids.size() != n_entries_ || (int64_t) plan.kind.size() != n_entries_) {
        err = "the plan's ids/kind do not match the window";
        return false;
    }
    if ((int64_t) plan.entries.size() > n_entries_) {
        err = "the plan has more entries than the window";
        return false;
    }

    uint8_t* base = (uint8_t*) map_;
    // the primary's belief: the per-layer format table (the peer compares it with its own)
    PeerLayerFmt* fmts = (PeerLayerFmt*) (base + layout_.off_fmt);
    for (int64_t l = 0; l < n_layers_; ++l) {
        fmts[l] = PeerLayerFmt{};
        if (l < (int64_t) view_formats_.size()) fmts[l] = view_formats_[l];
    }

    int32_t* ids = (int32_t*) (base + layout_.off_ids);
    int32_t* kind = (int32_t*) (base + layout_.off_kind);
    std::memcpy(ids, plan.ids.data(), (size_t) n_entries_ * 4);
    std::memcpy(kind, plan.kind.data(), (size_t) n_entries_ * 4);
    PeerPlanEntry* entries = (PeerPlanEntry*) (base + layout_.off_plan);
    std::memcpy(entries, plan.entries.data(), plan.entries.size() * sizeof(PeerPlanEntry));
    header_->n_plan = (int32_t) plan.entries.size();
    header_->publish_layer = plan.layer;
    header_->res_epoch_used = plan.res_epoch_used;

    uint64_t d = peer_fnv(ids, (size_t) n_entries_ * 4);
    d = peer_fnv(kind, (size_t) n_entries_ * 4, d);
    d = peer_fnv(entries, plan.entries.size() * sizeof(PeerPlanEntry), d);
    header_->plan_digest = corrupt_digest ? (int64_t) (d ^ 0x5a5a5a5a5a5a5a5aULL) : (int64_t) d;

    ++seq_;
    peer_store_rel(&header_->seq, seq_);
    header_->n_rows_written = 0;
    peer_store_rel(&header_->tag, kPeerTagPlan);
    return true;
}

bool PeerProcess::await_result(int timeout_ms, int32_t& status, std::string& err) {
    if (header_ == nullptr) {
        err = "the peer channel is not open";
        return false;
    }
    const int64_t deadline = now_ms() + timeout_ms;
    for (;;) {
        const int32_t tag = peer_load_acq(&header_->tag);
        if (tag == kPeerTagRows) {
            status = kPeerStOk;
            return true;
        }
        if (tag == kPeerTagRefused) {
            status = header_->status;
            return true;   // a refusal is an ANSWER: the caller reads the status
        }
        if (now_ms() >= deadline) {
            err = "the peer did not answer within " + std::to_string(timeout_ms) + " ms";
            return false;
        }
        usleep(200);
    }
}

const float* PeerProcess::rows() const {
    if (map_ == nullptr) return nullptr;
    return (const float*) ((const uint8_t*) map_ + layout_.off_out);
}

const float* PeerProcess::row(int32_t out_row) const {
    const float* r = rows();
    if (r == nullptr) return nullptr;
    return r + (size_t) out_row * (size_t) H_;
}

const PeerZoneHeader* PeerProcess::header() const { return header_; }

void PeerProcess::close() {
#if !defined(_WIN32)
    if (pid_ > 0) {
        // SIGTERM, then SIGKILL: the peer holds the zone and the device registration.
        kill(pid_, SIGTERM);
        for (int i = 0; i < 100; ++i) {
            int status = 0;
            if (waitpid(pid_, &status, WNOHANG) == pid_) {
                pid_ = -1;
                break;
            }
            usleep(10000);
        }
        if (pid_ > 0) {
            kill(pid_, SIGKILL);
            int status = 0;
            waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }
    if (map_ != nullptr) {
        if (registered_) {
            (void) cudaHostUnregister(map_);
            registered_ = false;
        }
        peer_zone_unmap(map_, map_bytes_);
        map_ = nullptr;
        map_bytes_ = 0;
        header_ = nullptr;
    }
    if (!zone_.empty()) {
        peer_zone_destroy(zone_);
        zone_.clear();
    }
#endif
}

}  // namespace strata::core
