//  Copyright (c) Lorenz Hübschle-Schneider
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

// during construction, hash only once. This reduces memory traffic for large
// objects (though not really relevant here) and reduces random accesses (more
// relevant), even though they are prefetchable (and prefetched)
#define RIBBON_PASS_HASH

#include "pcg-cpp/include/pcg_random.hpp"
#include "ribbon.hpp"
#include "rocksdb/stop_watch.h"

#include <tlx/cmdline_parser.hpp>
#include <tlx/logger.hpp>
#include <tlx/math/aggregate.hpp>
#include <tlx/thread_pool.hpp>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#define DO_EXPAND(VAL) VAL##1
#define EXPAND(VAL)    DO_EXPAND(VAL)

#if !defined(RIBBON_BITS) || (EXPAND(RIBBON_BITS) == 1)
#undef RIBBON_BITS
#define RIBBON_BITS 8
#endif

using namespace ribbon;

template <uint8_t depth, typename Config>
void run(unsigned num_filters, size_t num_slots, size_t num_queries, double eps,
         size_t seed, unsigned num_threads) {
    IMPORT_RIBBON_CONFIG(Config);
    constexpr bool debug = false;

    const double slots_per_item = eps + 1.0;
    const size_t num_items = num_slots / slots_per_item;
    LOG1 << "Running parallel test with " << num_slots << " slots, eps=" << eps
         << " -> " << num_items << " items, seed=" << seed
         << " config: L=" << kCoeffBits << " B=" << kBucketSize
         << " r=" << kResultBits << " FCA1? " << kFirstCoeffAlwaysOne << " SC? "
         << kSparseCoeffs << " mode " << (int)kThreshMode << " interleaved? "
         << kUseInterleavedSol << " CLS? " << kUseCacheLineStorage;

    rocksdb::StopWatchNano timer(true);
    using ribbon_t = ribbon_filter<depth, Config>;
    std::vector<ribbon_t> ribbons(num_filters);

    // Generate seeds
    std::vector<size_t> seeds(num_filters + 2 * num_threads);
    {
        std::seed_seq seq({seed >> 32, seed});
        std::vector<uint32_t> seeds32(2 * num_filters + 4 * num_threads);
        seq.generate(seeds32.begin(), seeds32.end());
        for (size_t i = 0; i < seeds.size(); i++) {
            seeds[i] = (static_cast<uint64_t>(seeds32[2 * i]) << 32) +
                       static_cast<uint64_t>(seeds32[2 * i + 1]);
        }
    }

    // all threads use the same input keys to make queries easier
    auto input = std::make_unique<Key[]>(num_items);
    std::iota(input.get(), input.get() + num_items, 0);

    std::mutex stats_lock;
    tlx::Aggregate<double> agg_init, agg_add, agg_backsub, agg_total, agg_size,
        agg_tl_bumped, agg_tl_empty, agg_tl_frac_empty, agg_tl_thresh_bytes;
    auto construct = [&](unsigned id) {
        [[maybe_unused]] rocksdb::StopWatchNano timer(true), total_timer(true);

        ribbons[id].Init(num_slots, slots_per_item, seeds[id], 0);
        auto t_init = timer.ElapsedNanos(true);
        LOG << "Ribbon " << id << " allocation took " << t_init / 1e6 << "ms";

        ribbons[id].AddRange(input.get(), input.get() + num_items);
        auto t_add = timer.ElapsedNanos(true);
        LOG << "Ribbon " << id << " insertion took " << t_add / 1e6
            << "ms in total";

        ribbons[id].BackSubst();
        auto t_backsub = timer.ElapsedNanos(true);
        LOG << "Ribbon " << id << " backsubstitution took " << t_backsub / 1e6
            << "ms in total";

        auto t_total = total_timer.ElapsedNanos();

        auto [tl_bumped, tl_empty_slots, tl_frac_empty, tl_thresh_bytes] =
            ribbons[id].GetStats();
        const size_t bytes = ribbons[id].Size();
        const double relsize =
            (bytes * 8 * 100.0) / (num_items * Config::kResultBits);

        // statistics data structures are protected by a mutex
        stats_lock.lock();
        agg_size.add(bytes);
        agg_tl_bumped.add(tl_bumped);
        agg_tl_empty.add(tl_empty_slots);
        agg_tl_frac_empty.add(tl_frac_empty);
        agg_tl_thresh_bytes.add(tl_thresh_bytes);
        agg_init.add(t_init / 1e6);
        agg_add.add(t_add / 1e6);
        agg_backsub.add(t_backsub / 1e6);
        agg_total.add(t_total / 1e6);
        stats_lock.unlock();

        LOG1 << "Ribbon " << std::setw(std::ceil(std::log10(num_filters))) << id
             << " total construction time: " << std::fixed
             << std::setprecision(2) << t_total / 1e6
             << "ms = " << std::setprecision(2) << t_total * 1.0 / num_items
             << "ns per item, size: " << bytes << " Bytes = " << relsize << "%";
        LOG1 << "RESULT type=cons id=" << id << " n=" << num_items
             << " m=" << num_slots << " eps=" << eps << " d=" << (int)depth
             << dump_config<Config>() << " tcons=" << t_total << " tinit=" << t_init
             << " tadd=" << t_add << " tbacksub=" << t_backsub
             << " bytes=" << bytes << " tlempty=" << tl_empty_slots
             << " tlbumped=" << tl_bumped << " tlemptyfrac=" << tl_frac_empty
             << " tlthreshbytes=" << tl_thresh_bytes
             << " overhead=" << relsize - 100 << " threads=" << num_threads;
    };

    tlx::ThreadPool pool(num_threads);
    for (unsigned i = 0; i < num_filters; i++) {
        pool.enqueue([&, i]() { construct(i); });
    }
    pool.loop_until_empty();

    auto cons_nanos = timer.ElapsedNanos();
    const size_t bytes_per_filter = agg_size.avg();
    LOG1 << "Parallel construction of " << num_filters << " filters with "
         << num_threads << " threads took " << cons_nanos / 1e6
         << "ms = " << cons_nanos * 1.0 / (num_items * num_filters)
         << "ns per item; total size " << static_cast<size_t>(agg_size.sum())
         << "B = " << std::setprecision(3) << (agg_size.sum() / 1e9) << "GB or "
         << bytes_per_filter << "B = " << std::setprecision(4)
         << (bytes_per_filter / 1e6) << "MB per filter = "
         << (bytes_per_filter * 8 * 100.0) / (num_items * Config::kResultBits)
         << "%";

    /**************************************************************************/

    std::vector<uint64_t> t_posquery(num_threads), t_posgen(num_threads),
        t_negquery(num_threads), t_neggen(num_threads), num_fps(num_threads);
    std::vector<double> fprates(num_threads);
    std::vector<char> ok(num_threads);
    std::atomic<uint64_t> check_nanos = 0;
    auto query = [&](unsigned id, bool positive, unsigned offset = 0) {
        constexpr size_t block_size = (size_t{1} << 20);
        std::vector<std::pair<Key, unsigned>> queries(block_size);
        pcg64 rng(seeds[id + num_threads + offset]);

        size_t count = 0, my_found = 0;

        rocksdb::StopWatchNano timer;
        uint64_t t_gen = 0, t_query = 0;

        while (count < num_queries) {
            // first, generate a block of queries
            timer.Start();
            size_t block_items = std::min(block_size, num_queries - count);
            for (size_t item = 0; item < block_items; item++, count++) {
                uint64_t rand = rng();
                Key key = rocksdb::FastRangeGeneric(rand, num_items);
                if (!positive)
                    key += num_items;
                unsigned filter = rocksdb::FastRangeGeneric(
                    rand ^ 0x876f170be4f1fcb9UL, num_filters);
                queries[item] = std::make_pair(key, filter);
            }
            queries.resize(block_items);
            t_gen += timer.ElapsedNanos(true);
            sLOG << "Generated" << block_items << "items";

            for (const auto [key, filter] : queries) {
                bool found = ribbons[filter].QueryFilter(key);
                assert(!positive || found);
                my_found += found;
            }
            t_query += timer.ElapsedNanos();
            sLOG << "Queried" << queries.size() << "items";
        }
        check_nanos.fetch_add(t_query);

        std::stringstream out;
        out << "Thread " << std::setw(std::ceil(std::log10(num_threads))) << id
            << " spent " << std::fixed << std::setprecision(2) << t_gen / 1e6
            << "ms generating " << num_queries << " queries, " << t_query / 1e6
            << "ms on queries = " << t_query * 1.0 / num_queries
            << "ns per query. ";
        if (positive) {
            ok[id] = (my_found == num_queries);
            t_posgen[id] = t_gen;
            t_posquery[id] = t_query;
            out << "Positive check "
                << (my_found == num_queries ? "OK" : "FAILED") << " found "
                << my_found << " of " << num_queries;
        } else {
            double fprate = my_found * 1.0 / num_queries;
            t_neggen[id] = t_gen;
            t_negquery[id] = t_query;
            num_fps[id] = my_found;
            fprates[id] = fprate;
            out << "Negative queries with " << my_found
                << " FPs = " << std::setprecision(5) << fprate * 100 << "%, i.e. "
                << fprate * (1ul << Config::kResultBits) << "x expected";
        }
        LOG1 << out.str();
    };
    std::vector<std::thread> threads;
    timer.Start();
    for (unsigned i = 0; i < num_threads; i++) {
        threads.emplace_back(query, i, true);
    }
    for (auto& t : threads)
        t.join();

    const uint64_t t_pos_aggregate = check_nanos,
                   t_pos_total = timer.ElapsedNanos();
    const bool all_ok =
        std::all_of(ok.begin(), ok.end(), [](const char o) { return o == 1; });
    LOG1 << "Parallel check with " << num_threads << " threads "
         << (all_ok ? "successful" : "FAILED") << " and took "
         << t_pos_total / 1e6 << "ms total; queries took "
         << t_pos_aggregate * 1.0 / (num_queries * num_threads) << "ns per key";

    /**************************************************************************/
    threads.clear();
    check_nanos = 0;
    timer.Start();
    for (unsigned i = 0; i < num_threads; i++) {
        threads.emplace_back(query, i, false, /* seed offset */ num_threads);
    }
    for (auto& t : threads)
        t.join();

    const uint64_t t_neg_aggregate = check_nanos,
                   t_neg_total = timer.ElapsedNanos();
    for (unsigned i = 0; i < num_threads; i++) {
        LOG1 << "RESULT type=query id=" << i << " n=" << num_items
             << " m=" << num_slots << " eps=" << eps << " d=" << (int)depth
             << dump_config<Config>() << " ok=" << (int)ok[i]
             << " tpos=" << t_posquery[i]
             << " tpospq=" << (t_posquery[i] * 1.0 / num_items)
             << " tneg=" << t_negquery[i]
             << " tneqpq=" << (t_negquery[i] * 1.0 / num_items)
             << " fps=" << num_fps[i] << " fpr=" << fprates[i]
             << " ratio=" << fprates[i] * (1ul << Config::kResultBits)
             << " threads=" << num_threads;
    }

    const uint64_t found = std::accumulate(num_fps.begin(), num_fps.end(), 0);
    const double fprate = found * 1.0 / (num_threads * num_queries),
                 ratio = fprate * (1ul << Config::kResultBits);
    const double relsize =
        (agg_size.avg() * 8 * 100.0) / (num_items * Config::kResultBits);

    LOG1 << "Negative check took " << t_neg_total / 1e6 << "ms total; queries took "
         << t_neg_aggregate * 1.0 / (num_queries * num_threads) << "ns per key, "
         << found << " FPs = " << fprate * 100 << "%, expecting "
         << 100.0 / (1ul << Config::kResultBits) << "% -> ratio = " << ratio;
    // Write aggregate results
    LOG1 << "RESULT type=agg n=" << num_items << " m=" << num_slots
         << " eps=" << eps << " d=" << (int)depth << dump_config<Config>()
         << " tcons=" << cons_nanos / 1e6 << " tconsavg=" << agg_total.avg()
         << " tconsdev=" << agg_total.stdev() << " tinitavg=" << agg_init.avg()
         << " tinitdev=" << agg_init.stdev() << " taddavg=" << agg_add.avg()
         << " tadddev=" << agg_add.stdev() << " tbacksubavg=" << agg_backsub.avg()
         << " tbacksubdev=" << agg_backsub.stdev()
         << " totalbytes=" << static_cast<size_t>(agg_size.sum())
         << " bytesavg=" << agg_size.avg() << " bytesdev=" << agg_size.stdev()
         << " tlemptyavg=" << agg_tl_empty.avg()
         << " tlemptydev=" << agg_tl_empty.stdev()
         << " tlbumpedavg=" << agg_tl_bumped.avg()
         << " tlbumpeddev=" << agg_tl_bumped.stdev()
         << " tlemptyfracavg=" << agg_tl_frac_empty.avg()
         << " tlepmtyfracdev=" << agg_tl_frac_empty.stdev()
         << " tlthreshbytesavg=" << agg_tl_thresh_bytes.avg()
         << " tlthreshbytesdev=" << agg_tl_thresh_bytes.stdev()
         << " overhead=" << relsize - 100 << " ok=" << all_ok
         << " tposagg=" << t_pos_aggregate << " tpostot=" << t_pos_total
         << " tpospq=" << t_pos_aggregate * 1.0 / (num_threads * num_queries)
         << " tnegagg=" << t_neg_aggregate << " tnegtot=" << t_neg_total
         << " tneqpq=" << t_neg_aggregate * 1.0 / (num_threads * num_queries)
         << " fps=" << found << " fpr=" << fprate << " ratio=" << ratio
         << " threads=" << num_threads;
}

// shut up and use 64-bit keys
template <size_t coeff_bits, size_t result_bits, ThreshMode mode, bool sparse,
          bool interleaved, bool cls, int shift = 0>
struct QRConfig
    : public RConfig<coeff_bits, result_bits, mode, sparse, interleaved, cls, shift, int64_t> {
    static constexpr bool log = false; // quiet
    using Key = int64_t;
    // using Index = uint64_t;
};

template <ThreshMode mode, uint8_t depth, size_t L, size_t r, bool interleaved,
          bool cls, typename... Args>
void dispatch_sparse(bool sparse, Args&... args) {
    if (sparse) {
        if constexpr (interleaved) {
            LOG1 << "Sparse coefficients + interleaved sol doesn't make sense";
        } else {
            // run<depth, QRConfig<L, r, mode, true, interleaved, cls>>(args...);
        }
    } else {
        run<depth, QRConfig<L, r, mode, false, interleaved, cls>>(args...);
    }
}

template <ThreshMode mode, uint8_t depth, size_t L, size_t r, typename... Args>
void dispatch_storage(bool cls, bool interleaved, Args&... args) {
    assert(!cls || !interleaved);
    if (cls) {
        dispatch_sparse<mode, depth, L, r, false, true>(args...);
    } else if (interleaved) {
        dispatch_sparse<mode, depth, L, r, true, false>(args...);
    } else {
        dispatch_sparse<mode, depth, L, r, false, false>(args...);
    }
}

template <ThreshMode mode, uint8_t depth, typename... Args>
void dispatch_width(size_t band_width, Args&... args) {
    static constexpr size_t r = RIBBON_BITS;
    switch (band_width) {
        case 16: dispatch_storage<mode, depth, 16, r>(args...); break;
        case 32: dispatch_storage<mode, depth, 32, r>(args...); break;
        case 64: dispatch_storage<mode, depth, 64, r>(args...); break;
        // case 128: dispatch_storage<mode, depth, 128, r>(args...); break;
        default: LOG1 << "Unsupported band width: " << band_width;
    }
}

template <ThreshMode mode, typename... Args>
void dispatch_depth(unsigned depth, Args&... args) {
    switch (depth) {
        case 0: dispatch_width<mode, 0>(args...); break;
        case 1: dispatch_width<mode, 1>(args...); break;
        case 2: dispatch_width<mode, 2>(args...); break;
        case 3: dispatch_width<mode, 3>(args...); break;
        // case 4: dispatch_width<mode, 4>(args...); break;
        default: LOG1 << "Unsupported recursion depth: " << depth;
    }
}

template <typename... Args>
void dispatch(ThreshMode mode, Args&... args) {
    switch (mode) {
        case ThreshMode::onebit:
            dispatch_depth<ThreshMode::onebit>(args...);
            break;
        case ThreshMode::twobit:
            dispatch_depth<ThreshMode::twobit>(args...);
            break;
        case ThreshMode::normal:
            dispatch_depth<ThreshMode::normal>(args...);
            break;
        default:
            LOG1 << "Unsupported threshold compression mode: " << (int)mode;
    }
}

int main(int argc, char** argv) {
    tlx::CmdlineParser cmd;
    size_t seed = 42, num_slots = 1024 * 1024, num_queries = 0;
    unsigned ribbon_width = 32, depth = 3;
    double eps = -1;
    unsigned num_filters = 100, num_threads = std::thread::hardware_concurrency();
    bool onebit = false, twobit = false, sparsecoeffs = false, cls = false,
         interleaved = false;
    cmd.add_size_t('s', "seed", seed, "random seed");
    cmd.add_size_t('m', "slots", num_slots, "number of slots per filter");
    cmd.add_size_t('q', "queries", num_queries, "number of queries per thread");
    cmd.add_unsigned('k', "filters", num_filters, "number of filters (shards)");
    cmd.add_unsigned('L', "ribbon_width", ribbon_width, "ribbon width (16/32/64)");
    cmd.add_unsigned('d', "depth", depth, "ribbon recursion depth");
    cmd.add_double('e', "epsilon", eps, "epsilon, #items = filtersize/(1+epsilon)");
    cmd.add_unsigned('t', "threads", num_threads, "number of query threads");
    cmd.add_bool('1', "onebit", onebit,
                 "use one-plus-a-little-bit threshold compression");
    cmd.add_bool('2', "twobit", twobit, "use two-bit threshold compression");
    cmd.add_bool('S', "sparsecoeffs", sparsecoeffs,
                 "use sparse coefficient vectors");
    cmd.add_bool('C', "cls", cls, "use cache-line solution storage");
    cmd.add_bool('I', "interleaved", interleaved,
                 "use interleaved solution storage");

    if (!cmd.process(argc, argv) || (onebit && twobit)) {
        cmd.print_usage();
        return 1;
    }
    if (eps == -1) {
        if (onebit) {
            size_t bucket_size = 1ul << tlx::integer_log2_floor(
                                     ribbon_width * ribbon_width /
                                     (4 * tlx::integer_log2_ceil(ribbon_width)));
            eps = -0.666 * ribbon_width / (4 * bucket_size + ribbon_width);
        } else {
            eps = -3.0 / ribbon_width;
        }
    }
    if (num_queries == 0)
        num_queries = num_slots * num_filters;
    if (seed == 0)
        seed = std::random_device{}();
    cmd.print_result();

    ThreshMode mode = onebit ? ThreshMode::onebit
                             : (twobit ? ThreshMode::twobit : ThreshMode::normal);

    dispatch(mode, depth, ribbon_width, cls, interleaved, sparsecoeffs,
             num_filters, num_slots, num_queries, eps, seed, num_threads);
}
