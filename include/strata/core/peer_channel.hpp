// include/strata/core/peer_channel.hpp - the cross-vendor peer channel: the transport between the
// primary engine and a peer process that runs on ANOTHER vendor's card.
//
// This is the transport only (tranche 1): the zone, the handshake, the plan publication and the
// refusal codes.  The peer writes rows into the zone; WHAT it computes is the next tranche (the
// expert kernels).  The protocol follows the prototype that passed on raptor
// (/home/hermes/e2/perf/peer_worker_v18.cpp, report projects/strata-cross-vendor/proto_protocol_v18.md):
//
//   * the zone is a POSIX shared memory segment (shm_open + mmap(MAP_SHARED)), NEVER a file mapping:
//     cudaHostRegister refuses a file-backed mapping ("invalid argument"), and it accepts tmpfs;
//   * the primary owns the segment (it creates it) and the peer maps it;
//   * both processes register THEIR OWN mapping with their own runtime (cudaHostRegister on the
//     primary, hipHostRegister on the peer), so each card gets its own device pointer on the same
//     physical pages;
//   * the blobs (the expert weights) never travel through the zone: the peer reopens the same
//     files read-only.
//
// The tag word is the whole synchronisation:
//   kPeerTagIdle     0  nothing published
//   kPeerTagPlan     1  the primary published a plan (the peer must verify it)
//   kPeerTagRows     2  the peer wrote its rows
//   kPeerTagRefused  3  the peer refused the plan; header.status says why
//
// The peer verifies the plan against ITS OWN state before writing a single row and refuses with a
// status instead (see kPeerSt*).  A refusal is a clean failure: no row is written, and the
// primary reads the status back.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The zone's magic and version.  The layout below is version 1 of the channel (tranche 1).
inline constexpr char kPeerZoneMagic[8] = {'S', 'T', 'R', 'P', 'C', 'H', 'A', 'N'};
inline constexpr int32_t kPeerZoneVersion = 1;

/// The tag word (see the header comment).
enum : int32_t {
    kPeerTagIdle = 0,
    kPeerTagPlan = 1,
    kPeerTagRows = 2,
    kPeerTagRefused = 3,
};

/// The peer's refusal codes.  They mirror the upstream failure modes (the numbers are the
/// prototype's, so a log line reads the same on both sides).
///   kPeerStFormat     the primary's per-layer format is not the peer's own
///   kPeerStEpoch      the residence epoch moved since the routing
///   kPeerStResidency  an expert planned for the peer is not resident on the peer
///   kPeerStGate       the landing gate of the layer is closed
///   kPeerStPlan       the plan is torn (its digest does not match)
enum : int32_t {
    kPeerStOk = 0,
    kPeerStFormat = 2,
    kPeerStEpoch = 3,
    kPeerStResidency = 4,
    kPeerStGate = 5,
    kPeerStPlan = 6,
};

/// The human name of a refusal code (never null).
const char* peer_status_name(int32_t status);

#pragma pack(push, 1)
/// The format of ONE layer, as the primary reads it from the pack (expert_layout.cpp's lay.fmt[]).
struct PeerLayerFmt {
    int32_t gu_type, d_type;   ///< the ggml types of gate/up and of down
    int32_t n_embd, n_ff;      ///< the dimensions
    int32_t had_block;         ///< the APR rotation block (0 = none)
    int32_t had_gu, had_d;     ///< the APR folded projection
    int64_t signs_digest;      ///< the digest of the d_signs vector (0 = identity)
};

/// The landing gate of ONE layer (the upstream AdaptGate: peer_experts.cpp's set_overlap_gate).
/// The peer refuses a plan whose layer still has copies in flight.
struct PeerGateState {
    int32_t landed;   ///< the layers whose copies have landed
    int32_t has;      ///< 1 = this layer's copies are in flight
    int32_t pending;  ///< the swaps still landing on this layer
    int32_t last;     ///< the last layer issued
};

/// One entry of the plan: the expert the peer must compute, the token it reads and the row of the
/// output it writes (the row index in the primary's mapped rows).
struct PeerPlanEntry {
    int32_t expert;
    int32_t token;
    int32_t out_row;
};

/// The zone's header.  Everything after the header is at the offsets the header carries (both
/// processes compute the same layout; see peer_channel.cpp).
struct PeerZoneHeader {
    char magic[8];               ///< kPeerZoneMagic
    int32_t version;             ///< kPeerZoneVersion
    int32_t tag;                 ///< the tag word (kPeerTag*)
    int32_t status;              ///< the peer's status (kPeerSt*)
    int32_t seq;                 ///< the publication counter (monotone)
    int32_t ready;               ///< 1 once the peer has mapped and registered the zone
    int32_t reg_primary;         ///< the primary's registration: 0 not tried, 1 ok, -1 failed
    int32_t reg_peer;            ///< the peer's registration: 0 not tried, 1 ok, -1 failed
    int32_t n_rows_written;      ///< how many rows the peer wrote for this plan
    int32_t H;                   ///< the hidden size (the floats of one row)
    int32_t n_tok, k;            ///< the window
    int32_t n_entries;           ///< n_tok * k
    int32_t n_plan;              ///< the entries the peer must compute
    int32_t n_layers;            ///< the layers described by the format table
    int32_t n_expert;            ///< the model's experts per layer
    int32_t publish_layer;       ///< the layer this plan belongs to
    int32_t mode_eff;            ///< the effective layout mode both cards must use
    int64_t grid_groups;         ///< the grid_groups argument both sides must pass
    int64_t res_epoch;           ///< the peer's residence epoch (the peer publishes it)
    int64_t res_epoch_used;      ///< the epoch the primary routed on (the peer checks it)
    int64_t plan_digest;         ///< the digest of the plan (a torn plan is refused)
    int64_t off_fmt;             ///< n_layers * PeerLayerFmt
    int64_t off_res;             ///< n_layers * n_expert int32 (the primary's belief: slot or -1)
    int64_t off_gate;            ///< n_layers * PeerGateState
    int64_t off_plan;            ///< n_plan * PeerPlanEntry
    int64_t off_ids;             ///< n_entries int32
    int64_t off_kind;            ///< n_entries int32
    int64_t off_out;             ///< n_entries * H floats (the rows the peer writes)
    int64_t off_x;               ///< n_tok * H floats
    int64_t total_bytes;         ///< the zone's size
};
#pragma pack(pop)

/// The zone's layout, computed identically by the two sides.
struct PeerZoneLayout {
    int64_t off_fmt = 0, off_res = 0, off_gate = 0, off_plan = 0;
    int64_t off_ids = 0, off_kind = 0, off_out = 0, off_x = 0, total = 0;
};

/// Computes the layout for the given window.  Both processes call it with the same numbers.
PeerZoneLayout peer_zone_layout(int64_t H, int64_t n_tok, int64_t k, int64_t n_expert, int64_t n_layers);

/// FNV-1a 64 (the digest the plan is stamped with).
uint64_t peer_fnv(const void* data, size_t bytes, uint64_t seed = 1469598103934665603ULL);

/// The value the peer writes in a row, so the primary can tell the peer's rows from its own.
/// The pattern is derived from the row index: row r, column c -> kPeerRowMagic + r * 1000 + c.
inline constexpr float kPeerRowMagic = 777.0f;
inline float peer_row_pattern(int32_t row, int32_t col) { return kPeerRowMagic + (float) row * 1000.0f + (float) col; }

/// The atomic helpers of the tag word (a cross-process word: release/acquire, never lock-free
/// assumptions about the compiler's atomics on a mapped page).
inline int32_t peer_load_acq(const volatile int32_t* p) {
#if defined(__GNUC__) || defined(__clang__)
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#else
    return *p;
#endif
}
inline void peer_store_rel(volatile int32_t* p, int32_t v) {
#if defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#else
    *p = v;
#endif
}

/// What the peer knows about ITSELF: its per-layer format, the experts it holds, its landing gate
/// and its residence epoch.  In the full engine the peer reads this from the pack it opens; in
/// tranche 1 (the transport only, no model) the primary hands it over at spawn time, so the two
/// views can disagree on purpose and the refusal path is exercised.
struct PeerState {
    std::vector<PeerLayerFmt> formats;           ///< the peer's own per-layer format
    std::vector<std::vector<int32_t>> resident;  ///< per layer: the expert ids the peer holds
    std::vector<PeerGateState> gates;            ///< per layer: the peer's landing gate
    int64_t epoch = 1;                           ///< the peer's residence epoch
};

/// The state text (one directive per line).  The peer reads it from the STRATA_PEER_STATE
/// environment variable the primary sets at spawn time.
std::string peer_state_serialize(const PeerState& state);
bool peer_state_parse(const std::string& text, PeerState& state, std::string& err);

/// What the primary publishes for one window.
struct PeerPlan {
    int32_t layer = 0;                   ///< the layer this plan belongs to
    int64_t res_epoch_used = 1;          ///< the epoch the primary routed on
    std::vector<int32_t> ids;            ///< n_entries: the routed expert of every entry
    std::vector<int32_t> kind;           ///< n_entries: 0 primary, 1 share, 2 peer, -1 CPU
    std::vector<PeerPlanEntry> entries;  ///< the entries the peer must compute
};

/// The primary's side of the channel: it creates the zone, spawns the peer, publishes the plans
/// and reads the peer's answer back.  Linux only (posix_spawnp + POSIX shared memory); on the
/// other platforms every method fails cleanly and open() says so.
class PeerProcess {
public:
    PeerProcess() = default;
    ~PeerProcess();
    PeerProcess(const PeerProcess&) = delete;
    PeerProcess& operator=(const PeerProcess&) = delete;

    /// What the primary believes about the peer, and what it publishes in the zone.  The peer
    /// checks the published plan against its own PeerState, so a disagreement is what it refuses.
    struct PeerView {
        int device = 0;                  ///< the peer's device (its visible index)
        std::string model;               ///< the model file the peer opens ("" = none)
        std::vector<PeerLayerFmt> formats;  ///< the primary's per-layer format (published in the zone)
        int64_t epoch = 1;               ///< the residence epoch the primary routed on
        int32_t mode_eff = 0;            ///< the effective layout mode both cards must use
        int64_t grid_groups = 0;         ///< the grid_groups argument both sides pass
    };

    /// Creates the zone, spawns the peer binary and waits for its handshake.
    /// `binary` empty -> the STRATA_PEER_BINARY environment variable, else the "strata-peer" beside
    /// the running engine (/proc/self/exe).  `extra_args` are appended to the peer's command line.
    bool open(int64_t H, int64_t n_tok, int64_t k, int64_t n_expert, int64_t n_layers, const PeerView& view,
              const PeerState& state, const std::string& binary, std::vector<std::string> extra_args,
              std::string& err);

    /// Publishes a plan: fills the zone (ids, kind, the entries, the format table, the gate and the
    /// epoch), stamps the digest and raises the tag to kPeerTagPlan.  `corrupt_digest` stamps a
    /// digest that does not match the plan (the test's torn-plan case).
    bool publish(const PeerPlan& plan, std::string& err, bool corrupt_digest = false);

    /// Waits for the peer's answer (kPeerTagRows or kPeerTagRefused).  On success `status` is
    /// kPeerStOk; on a refusal it is the peer's code and nothing was written.
    bool await_result(int timeout_ms, int32_t& status, std::string& err);

    /// The rows the peer wrote (n_entries * H floats).  Valid after await_result() returned ok.
    const float* rows() const;
    /// One row of the peer's output.
    const float* row(int32_t out_row) const;
    /// The zone's header (the status and the counters live there).
    const PeerZoneHeader* header() const;

    /// Stops the peer (SIGTERM, then SIGKILL) and unmaps the zone.
    void close();

    /// The zone's name (the POSIX shared memory object).
    const std::string& zone_name() const { return zone_; }
    /// The peer's process id.
    int pid() const { return pid_; }
    /// The path of the peer binary that was spawned.
    const std::string& binary() const { return binary_; }

private:
    std::string zone_;
    std::string binary_;
    void* map_ = nullptr;
    size_t map_bytes_ = 0;
    PeerZoneHeader* header_ = nullptr;
    int pid_ = -1;
    int64_t H_ = 0, n_tok_ = 0, k_ = 0, n_entries_ = 0, n_expert_ = 0, n_layers_ = 0;
    std::vector<PeerLayerFmt> view_formats_;   ///< the primary's per-layer format (published)
    PeerZoneLayout layout_;
    int32_t seq_ = 0;
    bool registered_ = false;
};

// ------------------------------------------------------------------ the zone (POSIX shared memory)
// Shared by the primary and the peer (both are separate binaries): create / open / map / unmap.
int peer_zone_create(const std::string& name, size_t bytes, std::string& err);
int peer_zone_open(const std::string& name, std::string& err, int wait_ms);
void* peer_zone_map(int fd, size_t bytes, std::string& err);
void peer_zone_unmap(void* p, size_t bytes);
void peer_zone_destroy(const std::string& name);

}  // namespace strata::core
