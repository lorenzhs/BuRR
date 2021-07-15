//  Copyright (c) Lorenz Hübschle-Schneider
//  Copyright (c) Facebook, Inc. and its affiliates.
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "rocksdb/math.h"

#include <tlx/logger.hpp>
#include <tlx/logger/wrap_unprintable.hpp>

#include <bitset>
#include <cstddef>
#include <ios>
#include <utility>

namespace ribbon {

namespace {
template <typename Hasher, typename SolutionStorage>
inline bool CheckBumped([[maybe_unused]] typename Hasher::Index val,
                        typename Hasher::Index cval, typename Hasher::Index bucket,
                        const Hasher &hasher, const SolutionStorage &sol) {
    [[maybe_unused]] constexpr bool debug = false;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;
    // if constexpr because hasher.Get() doesn't exist for other threshold compressors
    if constexpr (oneBitThresh) {
        if (cval == 2) {
            sol.PrefetchMeta(bucket);
            const auto plusthresh = hasher.Get(bucket);
            sLOG << "plus bumping:" << val << (val >= plusthresh ? ">=" : "<")
                 << plusthresh << "bucket" << bucket;
            return (val >= plusthresh);
        }
    }
    return (cval >= sol.GetMeta(bucket));
}
} // namespace

// Common functionality for querying a key (already hashed) in
// SimpleSolutionStorage.
template <typename SimpleSolutionStorage, typename Hasher>
std::pair<bool, typename SimpleSolutionStorage::ResultRow> inline ShiftQueryHelper(
    const Hasher &hasher, typename Hasher::Hash hash,
    const SimpleSolutionStorage &sss) {
    using Index = typename SimpleSolutionStorage::Index;
    using CoeffRow = typename SimpleSolutionStorage::CoeffRow;
    using ResultRow = typename SimpleSolutionStorage::ResultRow;

    constexpr bool debug = false;

    const Index start_slot = hasher.GetStart(hash, sss.GetNumStarts());
    // prefetch result rows (or, for CLS, also metadata)
    sss.PrefetchQuery(start_slot);

    const Index bucket = hasher.GetBucket(start_slot);
    Index val = hasher.GetIntraBucketFromStart(start_slot),
          cval = hasher.Compress(val);
    CoeffRow cr = hasher.GetCoeffs(hash);

    if (CheckBumped(val, cval, bucket, hasher, sss)) {
        sLOG << "Item was bumped, hash" << hash << "start" << start_slot
             << "bucket" << bucket << "val" << val << cval << "thresh"
             << (size_t)sss.GetMeta(bucket);
        return std::make_pair(true, 0);
    }

    sLOG << "Searching in bucket" << bucket << "start" << start_slot << "val"
         << val << cval << "below thresh =" << (size_t)sss.GetMeta(bucket)
         << "coeffs" << std::hex << (uint64_t)cr << std::dec;

    ResultRow result = 0;
    while (cr) {
        CoeffRow lsb = cr & -cr; // get the lowest set bit
        int i = rocksdb::CountTrailingZeroBits(cr);
        result ^= sss.GetResult(start_slot + i);
        cr ^= lsb;
    }
    return std::make_pair(false, result);
}

// Common functionality for querying a key (already hashed) in
// SimpleSolutionStorage.
template <typename SimpleSolutionStorage, typename Hasher>
std::pair<bool, typename SimpleSolutionStorage::ResultRow>
SimpleQueryHelper(const Hasher &hasher, typename Hasher::Hash hash,
                  const SimpleSolutionStorage &sss) {
    using Index = typename SimpleSolutionStorage::Index;
    using CoeffRow = typename SimpleSolutionStorage::CoeffRow;
    using ResultRow = typename SimpleSolutionStorage::ResultRow;

    constexpr bool debug = false;
    constexpr unsigned kCoeffBits = static_cast<unsigned>(sizeof(CoeffRow) * 8U);

    const Index start_slot = hasher.GetStart(hash, sss.GetNumStarts());
    // prefetch result rows (or, for CLS, also metadata)
    sss.PrefetchQuery(start_slot);

    const Index bucket = hasher.GetBucket(start_slot);
    const Index val = hasher.GetIntraBucketFromStart(start_slot),
                cval = hasher.Compress(val);
    const CoeffRow cr = hasher.GetCoeffs(hash);

    if (CheckBumped(val, cval, bucket, hasher, sss)) {
        sLOG << "Item was bumped, hash" << hash << "start" << start_slot
             << "bucket" << bucket << "val" << val << cval << "thresh"
             << (size_t)sss.GetMeta(bucket);
        return std::make_pair(true, 0);
    }

    sLOG << "Searching in bucket" << bucket << "start" << start_slot << "val"
         << val << cval << "below thresh =" << (size_t)sss.GetMeta(bucket)
         << "coeffs" << std::hex << (uint64_t)cr << std::dec;

    ResultRow result = 0;
    auto state = sss.PrepareGetResult(start_slot);
    for (unsigned i = 0; i < kCoeffBits; ++i) {
        // Bit masking whole value is generally faster here than 'if'
        // if  ((cr >> i) & 1)
        //    result ^= sss.GetResult(start_slot + i);
#ifdef RIBBON_CHECK
        auto expected = sss.PrepareGetResult(start_slot + i);
        assert(state == expected);
#endif
        ResultRow row = sss.GetFromState(state);
        result ^= row & (ResultRow{0} -
                         (static_cast<ResultRow>(cr >> i) & ResultRow{1}));
        state = sss.AdvanceState(state);
        if (debug && (static_cast<ResultRow>(cr >> i) & ResultRow{1})) {
            LOG << "Coeff " << i << " set, using row " << std::hex
                << (uint64_t)row << std::dec;
        }
    }
    return std::make_pair(false, result);
}

// General retrieval query a key from SimpleSolutionStorage.
template <typename SimpleSolutionStorage, typename Hasher>
std::pair<bool, typename SimpleSolutionStorage::ResultRow>
SimpleRetrievalQuery(const typename HashTraits<Hasher>::mhc_or_key_t &key,
                     const Hasher &hasher, const SimpleSolutionStorage &sss) {
    const auto hash = hasher.GetHash(key);

    static_assert(sizeof(typename SimpleSolutionStorage::Index) ==
                      sizeof(typename Hasher::Index),
                  "must be same");
    static_assert(sizeof(typename SimpleSolutionStorage::CoeffRow) ==
                      sizeof(typename Hasher::CoeffRow),
                  "must be same");

    // don't query an empty ribbon, please
    assert(sss.GetNumSlots() >= Hasher::kCoeffBits);

    return ShiftQueryHelper(hasher, hash, sss);
}

// Filter query a key from SimpleSolutionStorage.
template <typename SimpleSolutionStorage, typename Hasher>
std::pair<bool, bool>
SimpleFilterQuery(const typename HashTraits<Hasher>::mhc_or_key_t &key,
                  const Hasher &hasher, const SimpleSolutionStorage &sss) {
    constexpr bool debug = false;
    const auto hash = hasher.GetHash(key);
    const typename SimpleSolutionStorage::ResultRow expected =
        hasher.GetResultRowFromHash(hash);

    static_assert(sizeof(typename SimpleSolutionStorage::Index) ==
                      sizeof(typename Hasher::Index),
                  "must be same");
    static_assert(sizeof(typename SimpleSolutionStorage::CoeffRow) ==
                      sizeof(typename Hasher::CoeffRow),
                  "must be same");
    static_assert(sizeof(typename SimpleSolutionStorage::ResultRow) ==
                      sizeof(typename Hasher::ResultRow),
                  "must be same");

    // don't query an empty filter, please
    assert(sss.GetNumSlots() >= Hasher::kCoeffBits);

    auto [bumped, retrieved] = ShiftQueryHelper(hasher, hash, sss);
    sLOG << "Key" << tlx::wrap_unprintable(key) << "b?" << bumped << "retrieved"
         << std::hex << (size_t)retrieved << "expected" << (size_t)expected
         << std::dec;
    return std::make_pair(bumped, retrieved == expected);
}


/******************************************************************************/


// General retrieval query a key from InterleavedSolutionStorage.
template <typename InterleavedSolutionStorage, typename Hasher>
std::pair<bool, typename InterleavedSolutionStorage::ResultRow>
InterleavedRetrievalQuery(const typename HashTraits<Hasher>::mhc_or_key_t &key,
                          const Hasher &hasher,
                          const InterleavedSolutionStorage &iss) {
    using Hash = typename Hasher::Hash;
    using Index = typename InterleavedSolutionStorage::Index;
    using CoeffRow = typename InterleavedSolutionStorage::CoeffRow;
    using ResultRow = typename InterleavedSolutionStorage::ResultRow;

    static_assert(sizeof(Index) == sizeof(typename Hasher::Index),
                  "must be same");
    static_assert(sizeof(CoeffRow) == sizeof(typename Hasher::CoeffRow),
                  "must be same");

    constexpr bool debug = false;
    constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U);
    constexpr Index num_columns = InterleavedSolutionStorage::kResultBits;

    // don't query an empty ribbon, please
    assert(iss.GetNumSlots() >= kCoeffBits);
    const Hash hash = hasher.GetHash(key);
    const Index start_slot = hasher.GetStart(hash, iss.GetNumStarts());
    const Index bucket = hasher.GetBucket(start_slot);

    const Index start_block_num = start_slot / kCoeffBits;
    Index segment = start_block_num * num_columns;
    iss.PrefetchQuery(segment);

    const Index val = hasher.GetIntraBucketFromStart(start_slot),
                cval = hasher.Compress(val);

    if (CheckBumped(val, cval, bucket, hasher, iss)) {
        sLOG << "Item was bumped, hash" << hash << "start" << start_slot
             << "bucket" << bucket << "val" << val << cval << "thresh"
             << (size_t)iss.GetMeta(bucket);
        return std::make_pair(true, 0);
    }

    sLOG << "Searching in bucket" << bucket << "start" << start_slot << "val"
         << val << cval << "below thresh =" << (size_t)iss.GetMeta(bucket);

    const Index start_bit = start_slot % kCoeffBits;
    const CoeffRow cr = hasher.GetCoeffs(hash);

    ResultRow sr = 0;
    const CoeffRow cr_left = cr << start_bit;
    for (Index i = 0; i < num_columns; ++i) {
        sr ^= rocksdb::BitParity(iss.GetSegment(segment + i) & cr_left) << i;
    }

    if (start_bit > 0) {
        segment += num_columns;
        const CoeffRow cr_right = cr >> (kCoeffBits - start_bit);
        for (Index i = 0; i < num_columns; ++i) {
            sr ^= rocksdb::BitParity(iss.GetSegment(segment + i) & cr_right) << i;
        }
    }

    return std::make_pair(false, sr);
}

// Filter query a key from InterleavedFilterQuery.
template <typename InterleavedSolutionStorage, typename Hasher>
std::pair<bool, bool>
InterleavedFilterQuery(const typename HashTraits<Hasher>::mhc_or_key_t &key,
                       const Hasher &hasher,
                       const InterleavedSolutionStorage &iss) {
    // BEGIN mostly copied from InterleavedRetrievalQuery
    using Index = typename InterleavedSolutionStorage::Index;
    using CoeffRow = typename InterleavedSolutionStorage::CoeffRow;
    using ResultRow = typename InterleavedSolutionStorage::ResultRow;

    static_assert(sizeof(Index) == sizeof(typename Hasher::Index),
                  "must be same");
    static_assert(sizeof(CoeffRow) == sizeof(typename Hasher::CoeffRow),
                  "must be same");

    constexpr bool debug = false;
    constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U);
    constexpr auto num_columns = InterleavedSolutionStorage::kResultBits;

    // don't query an empty filter, please
    assert(iss.GetNumSlots() >= kCoeffBits);
    const typename HashTraits<Hasher>::hash_t hash = hasher.GetHash(key);
    const Index start_slot = hasher.GetStart(hash, iss.GetNumStarts());
    const Index bucket = hasher.GetBucket(start_slot);

    const Index start_block_num = start_slot / kCoeffBits;
    const Index segment = start_block_num * num_columns;
    iss.PrefetchQuery(segment);

    const Index val = hasher.GetIntraBucketFromStart(start_slot),
                cval = hasher.Compress(val);

    if (CheckBumped(val, cval, bucket, hasher, iss)) {
        sLOG << "Item was bumped, hash" << hash << "start" << start_slot
             << "bucket" << bucket << "val" << val << cval << "thresh"
             << (size_t)iss.GetMeta(bucket);
        return std::make_pair(true, false);
    }

    sLOG << "Searching for" << tlx::wrap_unprintable(key) << "in bucket"
         << bucket << "start" << start_slot << "val" << val << cval
         << "below thresh =" << (size_t)iss.GetMeta(bucket);

    const Index start_bit = start_slot % kCoeffBits;
    const CoeffRow cr = hasher.GetCoeffs(hash);
    // END mostly copied from InterleavedRetrievalQuery.

    const ResultRow expected = hasher.GetResultRowFromHash(hash);


    sLOG << "\tSlot" << start_slot << "-> block" << start_block_num << "segment"
         << segment << "start_bit" << start_bit << "expecting" << std::hex
         << (size_t)expected << "="
         << std::bitset<sizeof(ResultRow) * 8u>(expected).to_string()
         << "coeffs" << cr << std::dec << "="
         << std::bitset<sizeof(CoeffRow) * 8u>(cr).to_string();

    const CoeffRow cr_left = cr << start_bit;
    const CoeffRow cr_right =
        cr >> static_cast<unsigned>((kCoeffBits - start_bit) % kCoeffBits);
    // This determines whether our two memory loads are to different
    // addresses (common) or the same address (1/kCoeffBits chance)
    const Index maybe_num_columns = (start_bit != 0) * num_columns;

    for (Index i = 0; i < num_columns; ++i) {
        CoeffRow soln_data =
            (iss.GetSegment(segment + i) & cr_left) |
            (iss.GetSegment(segment + maybe_num_columns + i) & cr_right);
        if (rocksdb::BitParity(soln_data) != (static_cast<int>(expected >> i) & 1)) {
            sLOG << "\tMismatch at bit" << i << "have"
                 << rocksdb::BitParity(soln_data) << "expect"
                 << ((expected >> i) & 1) << "soln_data"
                 << std::bitset<sizeof(CoeffRow) * 8u>(soln_data).to_string();
            return std::make_pair(false, false);
        }
    }
    // otherwise, all match
    LOG << "\tGot all the right bits";
    return std::make_pair(false, true);
}

} // namespace ribbon
