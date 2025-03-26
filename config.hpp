//  Copyright (c) Lorenz Hübschle-Schneider
//  Copyright (c) Facebook, Inc. and its affiliates.
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include <tlx/math/integer_log2.hpp>

#include <cstdint>
#include <iomanip>
#include <utility>

// requires xxhash3, i.e. libxxhash v0.8.0 or later
// available from https://github.com/Cyan4973/xxHash/releases/tag/v0.8.0
#define XXH_INLINE_ALL
#include <xxhash.h>

namespace std {
ostream &operator<<(ostream &os, const __uint128_t &v) {
    return os << "0x" << ::std::hex << static_cast<uint64_t>(v >> 64)
              << ::std::setw(16) << static_cast<uint64_t>(v) << ::std::dec;
}
} // namespace std

namespace ribbon {

// clang-format off
template <size_t bits>
using at_least_t = std::conditional_t<bits <= 8, uint8_t,
                   std::conditional_t<bits <= 16, uint16_t,
                   std::conditional_t<bits <= 32, uint32_t,
                   std::conditional_t<bits <= 64, uint64_t,
                   std::conditional_t<bits <= 128, __uint128_t,
                                      /* error */ void>>>>>;

template <size_t bits>
using at_least_fast_t = std::conditional_t<bits <= 8, uint_fast8_t,
                        // use uint32_t here, it's much faster
                        std::conditional_t<bits <= 16, uint32_t,
                        std::conditional_t<bits <= 32, uint_fast32_t,
                        std::conditional_t<bits <= 64, uint_fast64_t,
                        std::conditional_t<bits <= 128, __uint128_t,
                                           /* error */ void>>>>>;

template <typename uint_type>
using make_fast_t = at_least_fast_t<8u * sizeof(uint_type)>;
// clang-format on

// threshold compression modes
enum class ThreshMode : int { normal = 0, onebit = 1, twobit = 2 };

// mode for searching for the bucket in which to bump elements between threads
// (in the parallel version)
enum class BucketSearchMode : int { minbump = 0, maxprev = 1, diff = 2 };

template <typename Config>
constexpr unsigned thresh_meta_bits =
    Config::kThreshMode == ThreshMode::normal
        ? (Config::kSparseCoeffs
               ? tlx::integer_log2_floor(
                     Config::kBucketSize /
                     // sparse start pos alignment: L = 16,32: 4,  L = 64,128: 8
                     (sizeof(typename Config::CoeffRow) >= 8 ? 8 : 4))
               : tlx::integer_log2_floor(Config::kBucketSize))
        : (Config::kThreshMode == ThreshMode::onebit
               ? 1
               : (Config::kThreshMode == ThreshMode::twobit ? 2 :
                                                            /* fallthrough */ 9999));

// To avoid writing 'typename' everywhere that we use types like 'Index'
#define IMPORT_RIBBON_CONFIG(RibbonConfig)                                      \
    using CoeffRow = typename RibbonConfig::CoeffRow;                           \
    using ResultRow = typename RibbonConfig::ResultRow;                         \
    using Index = typename RibbonConfig::Index;                                 \
    using Hash = typename RibbonConfig::Hash;                                   \
    using Key = typename RibbonConfig::Key;                                     \
                                                                                \
    /* Some more additions */                                                   \
    [[maybe_unused]] static constexpr auto kCoeffBits =                         \
        static_cast<Index>(sizeof(CoeffRow) * 8U);                              \
    [[maybe_unused]] static constexpr auto kResultBits =                        \
        RibbonConfig::kResultBits;                                              \
    [[maybe_unused]] static constexpr Index kBucketSize =                       \
        RibbonConfig::kBucketSize;                                              \
    [[maybe_unused]] static constexpr Index kMinBucketsPerThread =              \
        RibbonConfig::kMinBucketsPerThread;                                     \
                                                                                \
    /* Export to algorithm */                                                   \
    [[maybe_unused]] static constexpr bool kIsFilter = RibbonConfig::kIsFilter; \
    [[maybe_unused]] static constexpr bool kFirstCoeffAlwaysOne =               \
        RibbonConfig::kFirstCoeffAlwaysOne;                                     \
    [[maybe_unused]] static constexpr ThreshMode kThreshMode =                  \
        RibbonConfig::kThreshMode;                                              \
    [[maybe_unused]] static constexpr bool kSparseCoeffs =                      \
        RibbonConfig::kSparseCoeffs;                                            \
    [[maybe_unused]] static constexpr bool kUseInterleavedSol =                 \
        RibbonConfig::kUseInterleavedSol;                                       \
    [[maybe_unused]] static constexpr bool kUseCacheLineStorage =               \
        RibbonConfig::kUseCacheLineStorage;                                     \
    [[maybe_unused]] static constexpr bool kUseMHC = RibbonConfig::kUseMHC;     \
    [[maybe_unused]] static constexpr bool kUseMultiplyShiftHash =              \
        RibbonConfig::kUseMultiplyShiftHash;                                    \
    [[maybe_unused]] static constexpr Index kBucketSearchRange =                \
        RibbonConfig::kBucketSearchRange;                                       \
    [[maybe_unused]] static constexpr BucketSearchMode kBucketSearchMode =      \
        RibbonConfig::kBucketSearchMode;                                        \
                                                                                \
    static_assert(!kUseInterleavedSol || !kUseCacheLineStorage,                 \
                  "can't have both");                                           \
    static_assert(1 << tlx::integer_log2_floor(kBucketSize) == kBucketSize,     \
                  "bucket size must be a power of two");                        \
    static_assert(                                                              \
        sizeof(CoeffRow) + sizeof(ResultRow) + sizeof(Index) + sizeof(Hash) +   \
                sizeof(Key) + kBucketSize + kResultBits + kCoeffBits +          \
                kFirstCoeffAlwaysOne + (int)kThreshMode + kSparseCoeffs +       \
                kUseInterleavedSol + kUseCacheLineStorage + kUseMHC +           \
                kUseMultiplyShiftHash >                                         \
            0,                                                                  \
        "avoid unused warnings, semicolon expected after macro call")


// for printing sqlplottools RESULT lines
template <typename Config>
std::string dump_config() {
    IMPORT_RIBBON_CONFIG(Config);
    std::stringstream s;
    s << " L=" << kCoeffBits << " B=" << kBucketSize << " r=" << kResultBits
      << " mode=" << (int)kThreshMode << " sparse=" << kSparseCoeffs << " sol="
      << (kUseInterleavedSol ? "int" : (kUseCacheLineStorage ? "cls" : "basic"))
      << " fcao=" << kFirstCoeffAlwaysOne << " idxbits=" << 8u * sizeof(Index)
      << " keybits=" << 8u * sizeof(Key) << " filter=" << kIsFilter
      << " minbpt=" << kMinBucketsPerThread << " srange=" << kBucketSearchRange
      << " smode=";
    if (kBucketSearchRange == 0) {
        s << "nosearch";
    } else {
        switch (kBucketSearchMode) {
        case BucketSearchMode::minbump:
            s << "minbump";
            break;
        case BucketSearchMode::maxprev:
            s << "maxprev";
            break;
        case BucketSearchMode::diff:
            s << "diff";
            break;
        default:
            s << "unknown";
            break;
        }
    }
    return s.str();
}

// This is a very basic config that has all required attributes but doesn't
// necessarily make the best choices! Do not use this as is, instead have a look
// at RConfig and its derivatives below.  This class is mainly useful as it
// documents the effect of the various settings.
template <typename CoeffRow_, typename ResultRow_, typename Key_>
class DefaultConfig {
public:
    // An unsigned integer type.  The size of this type determines L, the width
    // of the ribbon.  A good choice is uint64_t.  Fewer than 16 bits yield high
    // overheads without improving performance, but correctness is maintained
    // down to uint8_t.  You can also use __uint128_t, which can be a bit slow
    // but yields excellent overhead (sub-0.1% possible).
    using CoeffRow = CoeffRow_;
    // A type that is big enough to hold the data that is stored in the data
    // structure (retrieval) / the fingerprints (filter)
    using ResultRow = ResultRow_;
    // The key type.  For retrieval, input consists of pairs of Key and
    // ResultRow, while for filters, it's just Key.
    using Key = Key_;

    // An unsigned type that is large enough to hold the maximum index in the
    // input (i.e., input size muts fit into Index).  Use uint64_t if you have
    // ribbons holding billions of items.
    using Index = uint32_t;
    // An unsigned type to hold hash values. This should likely always be
    // uint64_t.
    using Hash = uint64_t;

    // How many items form a bucket.  This should be O(L^2/log(L)) - see
    // recommended_bucket_size below, don't use this value
    static constexpr Index kBucketSize = 16u * sizeof(CoeffRow);

    // The minimum number of buckets that should be processed per thread when
    // inserting elements in parallel. If there are too few buckets, the number
    // of threads is reduced.
    // Increasing this will possibly save a few bytes because there are fewer
    // thread boundaries, but may also increase the runtime.
    static constexpr Index kMinBucketsPerThread = 20000 / kBucketSize;

    // How many bits the retrieval data structure should store per key / how
    // many fingerprint bits to use in a filter.  When using interleaved
    // storage, values other than 8*sizeof(ResultRow) are efficiently supported.
    static constexpr Index kResultBits = 8u * sizeof(ResultRow);

    // Whether to use a shift-multiply hash function for arithmetic keys instead
    // of xxhash.  This can be problematic if keys aren't random.  Should likely
    // stay disabled.
    static constexpr bool kUseMultiplyShiftHash = false;

    // The hash function to use for mapping Key to Hash.  This should be a good
    // hash function that yields sufficiently independent values for different
    // seeds.
    static constexpr Hash HashFn(const Key &key, uint64_t seed) {
        if constexpr (std::is_arithmetic_v<Key>) {
            return XXH3_64bits_withSeed(reinterpret_cast<const char *>(&key),
                                        sizeof(Key), seed);
        } else {
            return XXH3_64bits_withSeed(key.data(), key.size(), seed);
        }
    }

    // Whether the data structure is used as a filter or for retrieval.
    static constexpr bool kIsFilter = true;
    // Whether the first coefficient should always be set.  This makes things
    // slightly faster.
    static constexpr bool kFirstCoeffAlwaysOne = true;
    // Which threshold compressor to use.  `twobit` is fast and pretty good,
    // `onebit` yields improved compression at the cost of some (query)
    // performance.  `normal` uses uncompressed thresholds and should not
    // normally be used.
    static constexpr ThreshMode kThreshMode = ThreshMode::twobit;
    // Whether to use sparse coefficients. This speeds up queries for
    // non-interleaved solution storage, but is not fully optimized - the
    // threshold parameters need separate tuning for the sparse case.  As is,
    // sparse mode is a prototype and should not be used productively.
    static constexpr bool kSparseCoeffs = false;
    // Whether to use interleaved storage.  This is usually faster than the
    // alternatives.
    static constexpr bool kUseInterleavedSol = false;
    // Whether to use contiguous storage with embedded metadata.  This is
    // cache-efficient but slow in practice and should likely not be used.
    static constexpr bool kUseCacheLineStorage = false;
    // Whether to use Master Hash Codes.  This causes keys to be hashed only
    // once during insertion and query, using hash remixing to derive the next
    // level's hash if the key was bumped.  This is almost always a good idea.
    static constexpr bool kUseMHC = true;
    // Whether to print timings and other information about the construction.
    static constexpr bool log = false;
    // Number of buckets to search to find the bucket in which the
    // minimum number of elements needs to be bumped
    // (when using the parallel version)
    // Set to 0 disable.
    static constexpr Index kBucketSearchRange = 50;
    // Mode to use to search for the best bucket in which to bump elements
    // between threads (in the parallel version).
    // `minbump` takes the bucket in which the smallest number of elements needs to
    // be bumped directly.
    // `maxprev` takes the bucket with the most elements in the kCoeffBits-1 start
    // positions before the start of the bucket.
    // `diff` takes the bucket that minimizes the number of elements that need to
    // be bumped directly minus the number of elements in the kCoeffBits-1 start
    // positions before the start of the bucket.
    static constexpr BucketSearchMode kBucketSearchMode = BucketSearchMode::diff;
};

namespace {
// Internal.  Whether compressed metadata should be used for Cache-Line
// Storage. For other storage types, the answer is always yes - for CLS, it
// depends how much metadata space is available and whether uncompressed
// thresholds would fit.  This is because we don't save anything by using less
// than the available space.
template <typename Index, Index kBucketSize, Index kResultBits>
struct shouldCompressMeta {
    // internal
    static constexpr Index _kBucketIdxBits = tlx::integer_log2_ceil(kBucketSize),
                           _kBucketBits = kBucketSize * kResultBits,
                           _clbits = 512,
                           _buckets_per_cl = (_kBucketBits > _clbits)
                                                 ? 1
                                                 : _clbits / _kBucketBits;
    static constexpr bool value = _kBucketIdxBits * _buckets_per_cl > kResultBits;
};
} // namespace

template <size_t coeff_bits, ThreshMode mode>
constexpr size_t recommended_bucket_size =
    // round down to next power of two
    size_t{1} << tlx::integer_log2_floor(
        // w * w / (factor * log(w))
        coeff_bits * coeff_bits /
        ((mode == ThreshMode::normal ? 2 : 4) * tlx::integer_log2_ceil(coeff_bits)));

// Reasonable default config, but specify sizes in bits not types and make
// smarter choices.  This exposes a lot of options that users may not want to
// think about, see below for some reasonable presets.
template <size_t coeff_bits, size_t result_bits, ThreshMode mode = ThreshMode::twobit,
          bool sparse = false, bool interleaved = false, bool cls = false,
          int bucket_sh = 0, typename Key = int, int min_buckets_per_thread = -1>
struct RConfig
    : public DefaultConfig<at_least_t<coeff_bits>, at_least_t<result_bits>, Key> {
    using Super =
        DefaultConfig<at_least_t<coeff_bits>, at_least_t<result_bits>, Key>;
    using Index = uint32_t;
    static constexpr Index kResultBits = result_bits;
    static constexpr ThreshMode kThreshMode =
        // if using CLS, override compression mode
        Super::kUseCacheLineStorage
            ? (shouldCompressMeta<Index, Super::kBucketSize, kResultBits>::value
                   ? ThreshMode::twobit
                   : ThreshMode::normal)
            : mode;

    // Bucket size w^2 / 2 log w (uncompressed) or w^2 / 4 log w (compressed),
    // rounded down to next power of two. Optionally shift by 'bucket_sh'
    static constexpr Index kBucketSize =
        bucket_sh < 0 ? (recommended_bucket_size<coeff_bits, mode> >> -bucket_sh)
                      : (recommended_bucket_size<coeff_bits, mode> << bucket_sh);
    static constexpr Index kMinBucketsPerThread =
        min_buckets_per_thread < 0 ? 20000 / kBucketSize : min_buckets_per_thread;
    static constexpr bool kSparseCoeffs = sparse,
                          kUseInterleavedSol = interleaved,
                          kUseCacheLineStorage = cls, kUseMHC = true;
};

/******************************************************************************/
// Suggested configurations for users

// A fast filter config.  This uses L=64, 2-bit thresholds, and interleaved
// storage with dense coefficients and master hash codes.  It is a good
// all-round preset for filters.  You can expect <0.5% overhead if result_bits
// is at least 4; we measured 1.6% overhead for result_bits = 1.  For
// result_bits = 8 we saw 0.23% overhead, declining further as result_bits is
// increased.
template <size_t result_bits, typename Key>
struct FastFilterConfig
    : public RConfig</* L */ 64, result_bits, ThreshMode::twobit, /* sparse */ false,
                     /* interleaved */ true, /* cls */ false, /* shift */ 0, Key> {
};

// A very space-efficient filter config.  This uses L=128 and 2-bit thresholds,
// with the other choices as for FastFilterConfig. Expect <0.1% overhead for
// result_bits >= 7, and <0.2% for result_bits between 3 and 6.  We measured
// 0.45% overhead for result_bits = 1 and 0.18% overhead for result_bits = 3.
// For result_bits = 8, we saw 0.09% overhead, declining further as result_bits
// is increased.
template <size_t result_bits, typename Key>
struct CompactFilterConfig
    : public RConfig</* L */ 128, result_bits, ThreshMode::twobit, /* sparse */ false,
                     /* interleaved */ true, /* cls */ false, /* shift */ 0, Key> {
};
// An extremely space-efficient filter config.  This uses L=128 and 1-bit
// thresholds, with the other choices as for FastFilterConfig. Expect <0.1%
// overhead for result_bits >= 4.  We measured 0.33% overhead for result_bits =
// 1 and 0.12% overhead for result_bits = 3.  For result_bits = 8, we saw 0.05%
// overhead, declining further as result_bits is increased.
template <size_t result_bits, typename Key>
struct UltraCompactFilterConfig
    : public RConfig</* L */ 128, result_bits, ThreshMode::onebit, /* sparse */ false,
                     /* interleaved */ true, /* cls */ false, /* shift */ 0, Key> {
};


// A fast retrieval config, see FastFilterConfig for details
template <size_t result_bits, typename Key>
struct FastRetrievalConfig : public FastFilterConfig<result_bits, Key> {
    static constexpr bool kIsFilter = false;
};

// A very space-efficient retrieval config, see CompactFilterConfig for details
template <size_t result_bits, typename Key>
struct CompactRetrievalConfig : public CompactFilterConfig<result_bits, Key> {
    static constexpr bool kIsFilter = false;
};

// An extremely space-efficient retrieval config, see UltraCompactFilterConfig
// for details
template <size_t result_bits, typename Key>
struct UltraCompactRetrievalConfig
    : public UltraCompactFilterConfig<result_bits, Key> {
    static constexpr bool kIsFilter = false;
};

} // namespace ribbon
