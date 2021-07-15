//  Copyright (c) Lorenz Hübschle-Schneider
//  Copyright (c) Facebook, Inc. and its affiliates.
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#define RIBBON_CHECK 1

#include "backsubst.hpp"
#include "construction.hpp"
#include "pcg-cpp/include/pcg_random.hpp"
#include "permute.hpp"
#include "query.hpp"
#include "ribbon.hpp"
#include "storage.hpp"
#include "test_helpers.hpp"
#include "thresh_compress.hpp"

#include <tlx/logger.hpp>

#include <gtest/gtest.h>

#include <limits>
#include <numeric>
#include <unordered_set>

using namespace ribbon;
using namespace ribbon::test;

struct HashHelper {
    template <typename T>
    size_t operator()(T p) const {
        return XXH3_64bits((*p).data(), (*p).size());
    }
    size_t operator()(ribbon::test::RetrievalInputGen p) const {
        return XXH3_64bits(p->first.data(), p->first.size());
    }
};

/******************************************************************************/

TEST(PermuterTest, PermuterInvertible) {
    using Config = ribbon::DefaultConfig<uint16_t, uint8_t, int>;
    IMPORT_RIBBON_CONFIG(Config);
    Permuter<Config> perm(42);

    // Check that the transformation works
    const Index starts = 1024 - kCoeffBits + 1;
    for (Key key = 0; key < 1000; key++) {
        auto h = perm.GetHash(key);
        auto start = perm.GetStart(h, starts);
        auto sort = perm.StartToSort(start);
        ASSERT_EQ(start, perm.SortToStart(sort));
        auto intra = perm.GetIntraBucket(sort);
        ASSERT_EQ(intra, perm.GetIntraBucketFromStart(start));
        ASSERT_LE(intra, kBucketSize);
    }
}

template <typename Config>
void CacheLineStorageTest() {
    using ResultRow = typename Config::ResultRow;
    const size_t num_slots = 1024, num_buckets = num_slots / Config::kBucketSize;
    const ResultRow mask = (1u << Config::kResultBits) - 1;
    CacheLineStorage<Config> s(num_slots);
    const ResultRow metamask = Config::kThreshMode == ThreshMode::onebit
                                   ? 1
                                   : (Config::kThreshMode == ThreshMode::twobit
                                          ? 3
                                          : Config::kBucketSize - 1);
    for (size_t i = 0; i < num_slots; i++) {
        s.SetResult(i, static_cast<ResultRow>(i) & mask);
    }
    for (size_t b = 0; b < num_buckets; b++) {
        s.SetMeta(b, (b + 42) & metamask);
    }
    for (size_t i = 0; i < num_slots; i++) {
        EXPECT_EQ(static_cast<ResultRow>(i) & mask, s.GetResult(i));
    }
    for (size_t b = 0; b < num_buckets; b++) {
        EXPECT_EQ((b + 42) & metamask, s.GetMeta(b));
    }
}

TEST(StorageTest, CacheLineStorage) {
    // CacheLineStorageTest<QuietRConfig<16, 2>>();
    CacheLineStorageTest<QuietRConfig<16, 4>>();
    CacheLineStorageTest<QuietRConfig<16, 8>>();
    CacheLineStorageTest<QuietRConfig<32, 2>>();
    CacheLineStorageTest<QuietRConfig<32, 4>>();
    CacheLineStorageTest<QuietRConfig<32, 8>>();
    CacheLineStorageTest<QuietRConfig<32, 16>>();
}

/******************************************************************************/

void BasicFilterTest(size_t num_items) {
    // static constexpr bool debug = true;
    using Config = ribbon::test::BasicConfig;
    IMPORT_RIBBON_CONFIG(Config);

    Index num_slots = num_items;
    Index num_to_add = num_items;

    ribbon::test::StandardKeyGen begin("in", 0), end("in", num_to_add);

    BasicStorage<Config> storage(num_slots);
    NormalThreshold<Config> hasher(42);

    std::vector<std::string> bumped;
    BandingAddRange(&storage, hasher, begin, end, &bumped);
    sLOG0 << "Bumped" << bumped.size() << "of" << num_to_add
          << "items = " << (100.0 * bumped.size() / num_to_add) << "%";
    // Should have bumped at most 5%
    EXPECT_GE(num_slots * 0.05, bumped.size());

    SimpleBackSubst(storage, &storage);

    std::unordered_set<std::string> bumpset(bumped.begin(), bumped.end());
    for (auto cur = begin; cur != end; ++cur) {
        auto [was_bumped, found] = SimpleFilterQuery(*cur, hasher, storage);
        if (was_bumped) {
            ASSERT_EQ(1, bumpset.count(*cur));
        } else {
            ASSERT_TRUE(found);
        }
    }
}

TEST(RibbonTest, FilterBasic) {
    BasicFilterTest(512);
    BasicFilterTest(12800);
}

/******************************************************************************/

template <uint8_t depth, size_t coeff_bits = 16, size_t result_bits = 8>
void BasicRibbonTest(size_t input_size) {
    // using Config = ribbon::DefaultConfig<uint16_t, uint8_t, int>;
    using Config = QuietRConfig<coeff_bits, result_bits>;

    std::vector<int> input(input_size);
    std::iota(input.begin(), input.end(), 0);

    ribbon::ribbon_filter<depth, Config> r(input_size, 0.95, 42);
    r.AddRange(input.begin(), input.end());
    r.BackSubst();

    for (const auto& v : input) {
        ASSERT_TRUE(r.QueryFilter(v));
    }
    // negative queries
    size_t fp_count = 0;
    const int num_queries = 3 * input_size;
    for (int v = input_size; v < (int)input_size + num_queries; v++) {
        fp_count += r.QueryFilter(v);
    }
    double expected_fp_count = num_queries * 1.0 / (1ul << result_bits);
    // For expected FP rate, also include false positives due to collisions
    // in Hash value. (Negligible for 64-bit, can matter for 32-bit.)
    double collision_fp_rate =
        (1.0 * input_size) / std::pow(256.0, sizeof(typename Config::Hash));
    double correction = num_queries * collision_fp_rate;

    // Allow 3 standard deviations
    EXPECT_LE(fp_count, PoissonUpperBound(expected_fp_count + correction, 3.0));
}


TEST(RibbonTest, RibbonFilterBasic) {
    BasicRibbonTest<0>(240);
    BasicRibbonTest<1>(12800);
    BasicRibbonTest<2>(131072);
    // test some other coeff_bits / result_bits combinations
    BasicRibbonTest<1, 16, 2>(12800);
    BasicRibbonTest<1, 32, 4>(25600);
    BasicRibbonTest<1, 32, 8>(25600);
    BasicRibbonTest<1, 32, 16>(25600);
    BasicRibbonTest<1, 64, 8>(65536);
}

/******************************************************************************/

template <uint8_t depth>
void BasicRibbonRetrievalTest(size_t input_size) {
    using Value = uint16_t;
    using Config = ribbon::test::DefaultRetrievalConfig<uint16_t, Value, int>;
    IMPORT_RIBBON_CONFIG(Config);

    // Generate random data to be stored
    pcg32 rng(42);
    const Value max = std::numeric_limits<Value>::max();
    std::vector<std::pair<int, Value>> input;
    input.reserve(input_size);
    for (Index i = 0; i < input_size; i++) {
        input.emplace_back(i, static_cast<Value>(rng(max)));
    }

    ribbon::ribbon_filter<depth, Config> r(input_size, 0.95, 42);
    r.AddRange(input.begin(), input.end());
    r.BackSubst();

    for (const auto& [key, val] : input) {
        ASSERT_EQ(val, r.QueryRetrieval(key));
    }
}


TEST(RibbonTest, RibbonRetrievalBasic) {
    BasicRibbonRetrievalTest<0>(976);
    BasicRibbonRetrievalTest<1>(12800);
    BasicRibbonRetrievalTest<2>(131072);
}

/******************************************************************************/

void BasicRetrievalTest(size_t num_items) {
    // static constexpr bool debug = true;
    using Config = ribbon::test::RetrievalConfig;
    IMPORT_RIBBON_CONFIG(Config);

    Index num_slots = num_items;
    Index num_to_add = num_items;

    ribbon::test::RetrievalInputGen begin("in", 0), end("in", num_to_add);

    BasicStorage<Config> storage(num_slots);
    NormalThreshold<Config> hasher(42);

    std::vector<std::pair<std::string, uint8_t>> bumped;
    BandingAddRange(&storage, hasher, begin, end, &bumped);
    sLOG0 << "Bumped" << bumped.size() << "of" << num_to_add
          << "items = " << (100.0 * bumped.size() / num_to_add) << "%";
    // Should have bumped at most 5%
    EXPECT_GE(num_slots * 0.05, bumped.size());

    SimpleBackSubst(storage, &storage);

    std::unordered_map<std::string, uint8_t> bumpmap(bumped.begin(), bumped.end());
    for (auto cur = begin; cur != end; ++cur) {
        auto [was_bumped, retrieved] =
            SimpleRetrievalQuery(cur->first, hasher, storage);
        if (was_bumped) {
            auto it = bumpmap.find(cur->first);
            ASSERT_NE(bumpmap.end(), it);
            ASSERT_EQ(cur->second, it->second);
        } else {
            ASSERT_EQ(cur->second, retrieved);
        }
    }
}

TEST(RibbonTest, RetrievalBasic) {
    BasicRetrievalTest(512);
    BasicRetrievalTest(12800);
}

/******************************************************************************/

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
#ifdef GFLAGS
    ParseCommandLineFlags(&argc, &argv, true);
#endif // GFLAGS
    return RUN_ALL_TESTS();
}
