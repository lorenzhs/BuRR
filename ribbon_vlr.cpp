//  Copyright (c) Lorenz Hübschle-Schneider
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

//#define RIBBON_PASS_HASH
#include "ribbon.hpp"
#include "serialization.hpp"
#include "rocksdb/stop_watch.h"

#include <tlx/cmdline_parser.hpp>
#include <tlx/logger.hpp>

#include <atomic>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>

using namespace ribbon;

bool no_queries = false;

// NOTE: When an input file is given, the tests here still require num_items to
// be given because otherwise, the number of inserted elements can't be reconstructed.

template <size_t coeff_bits, size_t result_bits, ThreshMode mode = ThreshMode::twobit,
          bool sparse = false, bool interleaved = false, bool cls = false,
          int bucket_sh = 0, typename Key = int>
struct TestConfig : public RConfig<coeff_bits, result_bits, mode, sparse, interleaved, cls, bucket_sh, Key> {
    static constexpr bool kUseVLR = true;
    static constexpr bool kVLRFlipInputBits = false;
    static constexpr bool kVLRFlipOutputBits = false;
};

template <uint8_t depth, typename Config>
void run(size_t num_items, double eps, size_t seed, unsigned num_threads, std::string ifile, std::string ofile) {
    IMPORT_RIBBON_CONFIG(Config);

    const double slots_per_item = eps + 1.0;

    static_assert(kUseVLR);
    ribbon_filter<depth, Config> r;
    rocksdb::StopWatchNano timer(true);

    if (ifile.length() == 0) {
        // we need an extra one at the left
        assert(static_cast<ResultRowVLR>(~(ResultRowVLR(1) << (sizeof(ResultRowVLR) * 8U - 1))) >= num_items);
        auto input = std::make_unique<std::pair<Key, ResultRowVLR>[]>(num_items);
        for (ResultRowVLR i = 1; i <= num_items; ++i) {
            input.get()[i-1].first = i;
            input.get()[i-1].second = i | (ResultRowVLR(1) << (sizeof(ResultRowVLR) * 8U - rocksdb::CountLeadingZeroBits(i)));
        }
        LOG1 << "Input generation took " << timer.ElapsedNanos(true) / 1e6 << "ms";
        //const Index num_ribbons = sizeof(ResultRowVLR) * 8U;
        const Index num_ribbons = 22;
        r = ribbon_filter<depth, Config>(slots_per_item, seed, num_ribbons);
        LOG1 << "Allocation took " << timer.ElapsedNanos(true) / 1e6 << "ms\n";
        LOG1 << "Adding rows to filter....";
        r.AddRange(input.get(), input.get() + num_items);
        LOG1 << "Insertion took " << timer.ElapsedNanos(true) / 1e6 << "ms in total\n";

        input.reset();

        r.BackSubst();
        LOG1 << "Backsubstitution took " << timer.ElapsedNanos(true) / 1e6
             << "ms in total\n";
    } else {
        r.Deserialize(ifile);
        LOG1 << "Deserialization took " << timer.ElapsedNanos(true) / 1e6 << "ms\n";
    }
    if (ofile.length() != 0) {
        r.Serialize(ofile);
        LOG1 << "Serialization took " << timer.ElapsedNanos(true) / 1e6 << "ms\n";
    }

    const size_t bytes = r.Size();
    const double relsize = (bytes * 8 * 100.0) / (num_items * Config::kResultBits);
    LOG1 << "Ribbon size: " << bytes << " Bytes = " << (bytes * 1.0) / num_items
         << " Bytes per item = " << relsize << "%\n";

    std::atomic<bool> ok = true;
    auto pos_query = [&r, &ok, &num_items, &num_threads](unsigned id) {
        bool my_ok = true;
        size_t start = num_items / num_threads * id;
        // don't do the same queries on all threads
        for (ResultRowVLR v = start + 1; v <= num_items; v++) {
            ResultRowVLR val = r.QueryRetrieval(v);
            // for these simple tests to work, the values cannot be zero
            bool correct =  static_cast<ResultRowVLR>(val >> rocksdb::CountLeadingZeroBits(v)) == v;
            my_ok &= correct;
        }
        for (ResultRowVLR v = 1; v <= start; v++) {
            ResultRowVLR val = r.QueryRetrieval(v);
            bool correct =  static_cast<ResultRowVLR>(val >> rocksdb::CountLeadingZeroBits(v)) == v;
            my_ok &= correct;
        }
        if (!my_ok)
            ok = false;
    };
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < num_threads && !no_queries; i++) {
        threads.emplace_back(pos_query, i);
    }
    for (auto& t : threads)
        t.join();

    const auto check_nanos = timer.ElapsedNanos(true);
    LOG1 << "Parallel check with " << num_threads << " threads "
         << (ok ? "successful" : "FAILED") << " and took " << check_nanos / 1e6
         << "ms = " << check_nanos * 1.0 / num_items << "ns per key";

    auto [tl_bumped, tl_empty_slots, tl_frac_empty, tl_thresh_bytes] =
        r.GetStats();
    LOG1 << "RESULT n=" << num_items << " m=" << static_cast<size_t>(num_items * slots_per_item)
         << " eps=" << eps
         << " d=" << (int)depth << dump_config<Config>() << " bytes=" << bytes
         << " tlempty=" << tl_empty_slots << " tlbumped=" << tl_bumped
         << " tlemptyfrac=" << tl_frac_empty
         << " tlthreshbytes=" << tl_thresh_bytes << " overhead=" << relsize - 100
         << " ok=" << ok << " tpos=" << check_nanos
         << " tpospq=" << (check_nanos * 1.0 / num_items) << " threads=" << num_threads;
}


template <ThreshMode mode, uint8_t depth, size_t L, size_t r, bool interleaved,
          bool cls, bool sparse, typename... Args>
void dispatch_shift(int shift, Args&... args) {
    switch (shift) {
        case 0:
            run<depth, TestConfig<L, r, mode, sparse, interleaved, cls, 0>>(args...);
            break;
        /*case -1:
            run<depth, TestConfig<L, r, mode, sparse, interleaved, cls, -1>>(args...);
            break;
        case 1:
            run<depth, TestConfig<L, r, mode, sparse, interleaved, cls, 1>>(args...);
            break;*/
        default: LOG1 << "Unsupported bucket size shift: " << shift;
    }
}


template <ThreshMode mode, uint8_t depth, size_t L, size_t r, bool interleaved,
          bool cls, typename... Args>
void dispatch_sparse(bool sparse, Args&... args) {
    if (false && sparse) {
        if constexpr (interleaved) {
            LOG1 << "Sparse coefficients + interleaved sol doesn't make sense";
        } else {
            dispatch_shift<mode, depth, L, r, interleaved, cls, true>(args...);
        }
    } else {
        dispatch_shift<mode, depth, L, r, interleaved, cls, false>(args...);
    }
}

template <ThreshMode mode, uint8_t depth, size_t L, size_t r, typename... Args>
void dispatch_storage(bool cls, bool interleaved, Args&... args) {
    assert(!cls || !interleaved);
    if (cls) {
        // dispatch_sparse<mode, depth, L, r, false, true>(args...);
        LOG1 << "Cache-Line Storage is currently disabled";
    } else if (interleaved) {
        dispatch_sparse<mode, depth, L, r, true, false>(args...);
    } else {
        dispatch_sparse<mode, depth, L, r, false, false>(args...);
    }
}

template <ThreshMode mode, uint8_t depth, typename... Args>
void dispatch_width(size_t band_width, Args&... args) {
    static constexpr size_t r = 1;
    switch (band_width) {
        // case 16: dispatch_storage<mode, depth, 16, r>(args...); break;
        //case 32: dispatch_storage<mode, depth, 32, r>(args...); break;
        case 64: dispatch_storage<mode, depth, 64, r>(args...); break;
        /*case 128: dispatch_storage<mode, depth, 128, r>(args...); break;*/
        default: LOG1 << "Unsupported band width: " << band_width;
    }
}

template <ThreshMode mode, typename... Args>
void dispatch_depth(unsigned depth, Args&... args) {
    switch (depth) {
        /*case 0: dispatch_width<mode, 0>(args...); break;
        case 1: dispatch_width<mode, 1>(args...); break;
        case 2: dispatch_width<mode, 2>(args...); break;*/
        case 3: dispatch_width<mode, 3>(args...); break;
        //case 4: dispatch_width<mode, 4>(args...); break;
        default: LOG1 << "Unsupported recursion depth: " << depth;
    }
}

template <typename... Args>
void dispatch(ThreshMode mode, Args&... args) {
    switch (mode) {
        /*case ThreshMode::onebit:
            dispatch_depth<ThreshMode::onebit>(args...);
            break;*/
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
    size_t seed = 42, num_items = 1024 * 1024;
    unsigned ribbon_width = 32, depth = 3;
    double eps = -1;
    unsigned num_threads = std::thread::hardware_concurrency();
    bool onebit = false, twobit = false, sparsecoeffs = false, cls = false,
         interleaved = false;
    int shift = 0;
    std::string ifile, ofile;
    cmd.add_size_t('s', "seed", seed, "random seed");
    cmd.add_size_t('n', "items", num_items, "number of items in the retrieval data structure");
    cmd.add_unsigned('L', "ribbon_width", ribbon_width, "ribbon width (16/32/64)");
    cmd.add_unsigned('d', "depth", depth, "number of recursive filters");
    cmd.add_double('e', "epsilon", eps, "epsilon, #items = filtersize/(1+epsilon)");
    cmd.add_unsigned('t', "threads", num_threads, "number of query threads");
    cmd.add_int('b', "bsshift", shift,
                "whether to shift bucket size one way or another");
    cmd.add_bool('1', "onebit", onebit,
                 "use one-plus-a-little-bit threshold compression");
    cmd.add_bool('2', "twobit", twobit, "use two-bit threshold compression");
    cmd.add_bool('S', "sparsecoeffs", sparsecoeffs,
                 "use sparse coefficient vectors");
    cmd.add_bool('C', "cls", cls, "use cache-line solution storage");
    cmd.add_bool('I', "interleaved", interleaved,
                 "use interleaved solution storage");
    cmd.add_bool('Q', "noqueries", no_queries,
                 "don't run any queries (for scripting)");
    cmd.add_string('r', "read", ifile, "file to read serialized data");
    cmd.add_string('w', "write", ofile, "file to write serialized data");

    if (!cmd.process(argc, argv) || (onebit && twobit)) {
        cmd.print_usage();
        return 1;
    }
    if (eps == -1) {
        if (onebit) {
            size_t bucket_size = 1ul << tlx::integer_log2_floor(
                                     ribbon_width * ribbon_width /
                                     (4 * tlx::integer_log2_ceil(ribbon_width)));
            if (shift < 0)
                bucket_size >>= -shift;
            else
                bucket_size <<= shift;
            eps = -0.666 * ribbon_width / (4 * bucket_size + ribbon_width);
        } else {
            // for small ribbon widths, don't go too far from 0
            const double fct = ribbon_width <= 32 ? 3.0 : 4.0;
            eps = -fct / ribbon_width;
        }
    }
    if (seed == 0)
        seed = std::random_device{}();
    cmd.print_result();

    ThreshMode mode = onebit ? ThreshMode::onebit
                             : (twobit ? ThreshMode::twobit : ThreshMode::normal);

    dispatch(mode, depth, ribbon_width, cls, interleaved, sparsecoeffs, shift,
             num_items, eps, seed, num_threads, ifile, ofile);
}
