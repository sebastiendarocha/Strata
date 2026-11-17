// tests/core/peer_channel_test.cpp - the end-to-end test of the cross-vendor peer channel.
//
// It proves the TRANSPORT of tranche 1, without any expert computation and without any model:
//   * the primary creates the POSIX shared memory zone and spawns the peer with posix_spawnp,
//   * the peer maps the zone, registers it with its own runtime and answers the handshake,
//   * the primary publishes a plan, the peer writes a recognizable pattern in the rows and the
//     primary reads it back,
//   * at least three refusal cases are triggered and the returned status is checked (and no row
//     is written when the peer refuses).
//
// The peer binary is the one built beside this test (the "strata-peer" of the same build tree);
// STRATA_PEER_BINARY points it elsewhere (the peer built with the OTHER vendor's runtime) for a
// true cross-vendor run.
#include "strata/core/peer_channel.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    if (!ok) ++g_failures;
}

PeerLayerFmt fmt(int32_t n_embd, int32_t n_ff) {
    PeerLayerFmt f{};
    f.gu_type = 0;
    f.d_type = 1;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.had_block = 0;
    f.had_gu = 0;
    f.had_d = 0;
    f.signs_digest = 0;
    return f;
}

/// The peer's own state: it holds every expert of the layer, its gate is open and its epoch is 1.
PeerState good_state(int64_t n_layers, int64_t n_expert, int32_t n_embd, int32_t n_ff) {
    PeerState s;
    s.epoch = 1;
    for (int64_t l = 0; l < n_layers; ++l) {
        s.formats.push_back(fmt(n_embd, n_ff));
        std::vector<int32_t> res;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) res.push_back(e);
        s.resident.push_back(res);
        PeerGateState g{};
        g.landed = (int32_t) l;
        g.has = 0;
        g.pending = 0;
        g.last = (int32_t) l;
        s.gates.push_back(g);
    }
    return s;
}

/// The primary's view: the same per-layer format, so the peer accepts the plan.
PeerProcess::PeerView good_view(int64_t n_layers, int32_t n_embd, int32_t n_ff) {
    PeerProcess::PeerView v;
    v.device = 0;
    v.epoch = 1;
    v.mode_eff = 0;
    v.grid_groups = 1;
    for (int64_t l = 0; l < n_layers; ++l) v.formats.push_back(fmt(n_embd, n_ff));
    return v;
}

/// A plan: every entry routed to the peer, one entry per (token, k) slot.
PeerPlan make_plan(int64_t n_tok, int64_t k, int32_t expert, int64_t layer, int64_t epoch_used) {
    PeerPlan p;
    p.layer = (int32_t) layer;
    p.res_epoch_used = epoch_used;
    const int64_t n = n_tok * k;
    p.ids.assign((size_t) n, expert);
    p.kind.assign((size_t) n, 2);   // 2 = the peer
    for (int64_t i = 0; i < n; ++i) {
        PeerPlanEntry e{};
        e.expert = expert;
        e.token = (int32_t) (i / k);
        e.out_row = (int32_t) i;
        p.entries.push_back(e);
    }
    return p;
}

bool rows_are_zero(const PeerProcess& peer, int64_t n_entries, int64_t H) {
    for (int64_t r = 0; r < n_entries; ++r) {
        const float* row = peer.row((int32_t) r);
        if (row == nullptr) return false;
        for (int64_t c = 0; c < H; ++c) {
            if (row[c] != 0.0f) return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const int64_t H = 8, n_tok = 4, k = 2, n_expert = 4, n_layers = 2;
    const int64_t n_entries = n_tok * k;

    std::string binary;
    if (argc > 1) binary = argv[1];
    if (binary.empty()) {
        const char* e = std::getenv("STRATA_PEER_BINARY");
        if (e != nullptr) binary = e;
    }

    std::printf("peer channel test (H=%lld n_tok=%lld k=%lld n_expert=%lld n_layers=%lld)\n",
                (long long) H, (long long) n_tok, (long long) k, (long long) n_expert, (long long) n_layers);

    // ---------------------------------------------------------------- 1. the happy path
    {
        PeerProcess peer;
        std::string err;
        const PeerState state = good_state(n_layers, n_expert, 64, 128);
        const PeerProcess::PeerView view = good_view(n_layers, 64, 128);
        if (!peer.open(H, n_tok, k, n_expert, n_layers, view, state, binary, {}, err)) {
            std::printf("  FAIL the channel did not open: %s\n", err.c_str());
            ++g_failures;
        } else {
            std::printf("  ok   the channel opened (zone %s, peer pid %d, binary %s)\n",
                        peer.zone_name().c_str(), peer.pid(), peer.binary().c_str());
            check(peer.header()->ready == 1, "the peer answered the handshake");
            std::printf("       registration: primary %d, peer %d\n", peer.header()->reg_primary,
                        peer.header()->reg_peer);

            PeerPlan plan = make_plan(n_tok, k, 2, 0, 1);
            check(peer.publish(plan, err), "the plan was published");
            int32_t status = -1;
            check(peer.await_result(30000, status, err), "the peer answered the plan");
            check(status == kPeerStOk, "the peer accepted the plan (status ok)");
            check(peer.header()->n_rows_written == (int32_t) n_entries,
                  "the peer wrote every planned row");

            // the rows the primary reads back are the ones the peer wrote
            bool pattern_ok = true;
            for (int64_t r = 0; r < n_entries; ++r) {
                const float* row = peer.row((int32_t) r);
                for (int64_t c = 0; c < H; ++c) {
                    if (row[c] != peer_row_pattern((int32_t) r, (int32_t) c)) pattern_ok = false;
                }
            }
            check(pattern_ok, "the primary read back the peer's rows (the row-index pattern)");
            peer.close();
        }
    }

    // ---------------------------------------------------------------- 2. the refusals
    // (a) the per-layer format differs (status 2)
    {
        PeerProcess peer;
        std::string err;
        PeerState state = good_state(n_layers, n_expert, 64, 128);
        state.formats[0].n_ff = 256;   // the peer believes another layout
        const PeerProcess::PeerView view = good_view(n_layers, 64, 128);
        if (!peer.open(H, n_tok, k, n_expert, n_layers, view, state, binary, {}, err)) {
            std::printf("  FAIL the channel did not open: %s\n", err.c_str());
            ++g_failures;
        } else {
            PeerPlan plan = make_plan(n_tok, k, 2, 0, 1);
            peer.publish(plan, err);
            int32_t status = -1;
            check(peer.await_result(30000, status, err), "the peer answered (format case)");
            check(status == kPeerStFormat, "refusal: the per-layer format differs (status 2)");
            check(peer.header()->n_rows_written == 0, "refusal: no row was written");
            check(rows_are_zero(peer, n_entries, H), "refusal: the output rows are untouched");
            peer.close();
        }
    }

    // (b) the residence epoch moved (status 3)
    {
        PeerProcess peer;
        std::string err;
        const PeerState state = good_state(n_layers, n_expert, 64, 128);
        const PeerProcess::PeerView view = good_view(n_layers, 64, 128);
        if (!peer.open(H, n_tok, k, n_expert, n_layers, view, state, binary, {}, err)) {
            std::printf("  FAIL the channel did not open: %s\n", err.c_str());
            ++g_failures;
        } else {
            PeerPlan plan = make_plan(n_tok, k, 2, 0, 7);   // the primary routed on an old epoch
            peer.publish(plan, err);
            int32_t status = -1;
            check(peer.await_result(30000, status, err), "the peer answered (epoch case)");
            check(status == kPeerStEpoch, "refusal: the residence epoch moved (status 3)");
            check(peer.header()->n_rows_written == 0, "refusal: no row was written");
            peer.close();
        }
    }

    // (c) a planned expert is not resident on the peer (status 4)
    {
        PeerProcess peer;
        std::string err;
        PeerState state = good_state(n_layers, n_expert, 64, 128);
        state.resident[0].clear();   // the peer holds nothing on that layer
        const PeerProcess::PeerView view = good_view(n_layers, 64, 128);
        if (!peer.open(H, n_tok, k, n_expert, n_layers, view, state, binary, {}, err)) {
            std::printf("  FAIL the channel did not open: %s\n", err.c_str());
            ++g_failures;
        } else {
            PeerPlan plan = make_plan(n_tok, k, 2, 0, 1);
            peer.publish(plan, err);
            int32_t status = -1;
            check(peer.await_result(30000, status, err), "the peer answered (residency case)");
            check(status == kPeerStResidency, "refusal: a planned expert is not resident (status 4)");
            check(peer.header()->n_rows_written == 0, "refusal: no row was written");
            peer.close();
        }
    }

    // (d) the landing gate of the layer is closed (status 5)
    {
        PeerProcess peer;
        std::string err;
        PeerState state = good_state(n_layers, n_expert, 64, 128);
        state.gates[0].pending = 2;   // copies still landing
        const PeerProcess::PeerView view = good_view(n_layers, 64, 128);
        if (!peer.open(H, n_tok, k, n_expert, n_layers, view, state, binary, {}, err)) {
            std::printf("  FAIL the channel did not open: %s\n", err.c_str());
            ++g_failures;
        } else {
            PeerPlan plan = make_plan(n_tok, k, 2, 0, 1);
            peer.publish(plan, err);
            int32_t status = -1;
            check(peer.await_result(30000, status, err), "the peer answered (gate case)");
            check(status == kPeerStGate, "refusal: the landing gate is closed (status 5)");
            check(peer.header()->n_rows_written == 0, "refusal: no row was written");
            peer.close();
        }
    }

    // (e) the plan's digest does not match (status 6)
    {
        PeerProcess peer;
        std::string err;
        const PeerState state = good_state(n_layers, n_expert, 64, 128);
        const PeerProcess::PeerView view = good_view(n_layers, 64, 128);
        if (!peer.open(H, n_tok, k, n_expert, n_layers, view, state, binary, {}, err)) {
            std::printf("  FAIL the channel did not open: %s\n", err.c_str());
            ++g_failures;
        } else {
            PeerPlan plan = make_plan(n_tok, k, 2, 0, 1);
            peer.publish(plan, err, /*corrupt_digest=*/true);
            int32_t status = -1;
            check(peer.await_result(30000, status, err), "the peer answered (digest case)");
            check(status == kPeerStPlan, "refusal: the plan's digest does not match (status 6)");
            check(peer.header()->n_rows_written == 0, "refusal: no row was written");
            peer.close();
        }
    }

    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
