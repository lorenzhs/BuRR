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
template <typename Hasher, typename SolutionStorage>
inline bool CheckBumpedVLR([[maybe_unused]] typename Hasher::Index val,
                        typename Hasher::Index cval, typename Hasher::Index bucket,
                        const Hasher &hasher, const SolutionStorage &sol, size_t ribbon_idx) {
    [[maybe_unused]] constexpr bool debug = false;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;
    // if constexpr because hasher.Get() doesn't exist for other threshold compressors
    if constexpr (oneBitThresh) {
        if (cval == 2) {
            sol.PrefetchMeta(bucket, ribbon_idx);
            const auto plusthresh = hasher.Get(bucket, ribbon_idx);
            sLOG << "plus bumping:" << val << (val >= plusthresh ? ">=" : "<")
                 << plusthresh << "bucket" << bucket;
            return (val >= plusthresh);
        }
    }
    return (cval >= sol.GetMeta(bucket, ribbon_idx));
}
} // namespace

// Common functionality for querying a key (already hashed) in
// SimpleSolutionStorage.
template <typename SimpleSolutionStorage, typename Hasher>
std::pair<bool,
          std::conditional_t<SimpleSolutionStorage::kUseVLR,
                             typename SimpleSolutionStorage::ResultRowVLR,
                             typename SimpleSolutionStorage::ResultRow>> inline ShiftQueryHelper(
    typename Hasher::Hash hash, const Hasher &hasher,
    const SimpleSolutionStorage &sss, [[maybe_unused]] typename SimpleSolutionStorage::Index start_idx = 0,
    [[maybe_unused]] typename SimpleSolutionStorage::Index num_bits = 0) {

    constexpr bool kUseVLR = SimpleSolutionStorage::kUseVLR;
    constexpr bool kVLRShareMeta = SimpleSolutionStorage::kVLRShareMeta;
    constexpr bool kVLRFlipOutputBits = SimpleSolutionStorage::kVLRFlipOutputBits;
    using Index = typename SimpleSolutionStorage::Index;
    using CoeffRow = typename SimpleSolutionStorage::CoeffRow;
    using ResultRow = typename SimpleSolutionStorage::ResultRow;
    using ResultRowVLR = typename SimpleSolutionStorage::ResultRowVLR;

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
    if constexpr (kUseVLR && kVLRShareMeta) {
        ResultRowVLR result_vlr = 0;
        Index num_ribbons = sss.GetNumRibbons();
        Index num_vlr_columns = std::min(num_ribbons, static_cast<Index>(sizeof(ResultRowVLR) * 8U));
        Index cur_ribbon = hasher.GetVLRIndex(hash, num_ribbons);
        if (num_bits == 0) {
            start_idx = 0;
            num_bits = num_vlr_columns;
        }
        assert(num_bits + start_idx <= num_vlr_columns);
        Index idx;
        if constexpr (kVLRFlipOutputBits)
            idx = start_idx;
        else
            idx = sizeof(ResultRowVLR) * 8U - 1 - start_idx;
        for (Index _ = start_idx; _ < start_idx + num_bits; ++_) {
            result = 0;
            CoeffRow cr_copy = cr;
            // FIXME: see comment in ShiftQueryHelperVLR
            while (cr_copy) {
                CoeffRow lsb = cr_copy & -cr_copy; // get the lowest set bit
                int i = rocksdb::CountTrailingZeroBits(cr_copy);
                result ^= sss.GetResult(start_slot + i, cur_ribbon);
                cr_copy ^= lsb;
            }
            result_vlr |= static_cast<ResultRowVLR>(result) << idx;
            ++cur_ribbon;
            cur_ribbon %= num_ribbons;
            if constexpr (kVLRFlipOutputBits)
                ++idx;
            else
                --idx;
        }
        return std::make_pair(false, result_vlr);
    } else {
        while (cr) {
            CoeffRow lsb = cr & -cr; // get the lowest set bit
            int i = rocksdb::CountTrailingZeroBits(cr);
            result ^= sss.GetResult(start_slot + i);
            cr ^= lsb;
        }
        return std::make_pair(false, result);
    }
}

// Common functionality for querying a key (already hashed) in
// SimpleSolutionStorage.
// in return value, first element is bump mask, second element is (partial) stored value
template <typename SimpleSolutionStorage, typename Hasher>
std::pair<typename SimpleSolutionStorage::ResultRowVLR, typename SimpleSolutionStorage::ResultRowVLR> inline ShiftQueryHelperVLR(
    typename Hasher::Hash hash, const Hasher &hasher, const SimpleSolutionStorage &sss,
    typename SimpleSolutionStorage::ResultRowVLR bump_mask) {
    using Index = typename SimpleSolutionStorage::Index;
    using CoeffRow = typename SimpleSolutionStorage::CoeffRow;
    using ResultRow = typename SimpleSolutionStorage::ResultRow;
    using ResultRowVLR = typename SimpleSolutionStorage::ResultRowVLR;
    constexpr bool kVLRFlipOutputBits = SimpleSolutionStorage::kVLRFlipOutputBits;

    constexpr bool debug = false;

    const Index start_slot = hasher.GetStart(hash, sss.GetNumStarts());
    const Index num_ribbons = sss.GetNumRibbons();
    const Index start_ribbon = hasher.GetVLRIndex(hash, num_ribbons);
    // prefetch result rows
    sss.PrefetchQuery(start_slot, start_ribbon);

    const Index bucket = hasher.GetBucket(start_slot);
    Index val = hasher.GetIntraBucketFromStart(start_slot),
          cval = hasher.Compress(val);

    ResultRowVLR new_mask = 0;
    ResultRowVLR value = 0;
    const CoeffRow cr_orig = hasher.GetCoeffs(hash);
    while (bump_mask) {
        unsigned int bump_first;
        if constexpr (kVLRFlipOutputBits)
            bump_first = rocksdb::CountTrailingZeroBits(bump_mask);
        else
            bump_first = rocksdb::CountLeadingZeroBits(bump_mask);
        // FIXME: maybe avoid this if by just setting the mask properly in ribbon.hpp
        // values cannot contain more bits than there are ribbons
        if (bump_first >= num_ribbons)
            break;
        const Index ribbon_idx = (start_ribbon + bump_first) % num_ribbons;
        int index;
        if constexpr (kVLRFlipOutputBits)
            index = bump_first;
        else
            index = sizeof(ResultRowVLR) * 8U - bump_first - 1;
        if (CheckBumpedVLR(val, cval, bucket, hasher, sss, ribbon_idx)) {
            sLOG << "Item was bumped, hash" << hash << "start" << start_slot
                 << "bucket" << bucket << "val" << val << cval << "thresh"
                 << (size_t)sss.GetMeta(bucket) << "ribbon" << ribbon_idx;
            new_mask |= ResultRowVLR(1) << index;
            bump_mask &= ~(ResultRowVLR(1) << index);
            continue;

        }
        bump_mask &= ~(ResultRowVLR(1) << index);

        CoeffRow cr = cr_orig;
        sLOG << "Searching in bucket" << bucket << "start" << start_slot << "val"
             << val << cval << "below thresh =" << (size_t)sss.GetMeta(bucket)
             << "coeffs" << std::hex << (uint64_t)cr << std::dec
             << "ribbon" << ribbon_idx;

        // NOTE: This could be optimized by fetching an entire ResultRow of bits at
        // the same time instead of going bit-for-bit. It probably doesn't make much
        // sense to optimize this, though, since interleaved storage is normally
        // used anyways.
        ResultRow result = 0;
        while (cr) {
            CoeffRow lsb = cr & -cr; // get the lowest set bit
            int i = rocksdb::CountTrailingZeroBits(cr);
            result ^= sss.GetResult(start_slot + i, ribbon_idx);
            cr ^= lsb;
        }
        value |= static_cast<ResultRowVLR>(result) << index;
    }
    return std::make_pair(new_mask, value);
}

// General retrieval query a key from SimpleSolutionStorage.
template <typename SimpleSolutionStorage, typename Hasher>
std::pair<bool,
          std::conditional_t<SimpleSolutionStorage::kUseVLR,
                             typename SimpleSolutionStorage::ResultRowVLR,
                             typename SimpleSolutionStorage::ResultRow>>
SimpleRetrievalQuery(const typename HashTraits<Hasher>::mhc_or_key_t &key,
                     const Hasher &hasher, const SimpleSolutionStorage &sss,
                     [[maybe_unused]] typename SimpleSolutionStorage::Index start_idx = 0,
                     [[maybe_unused]] typename SimpleSolutionStorage::Index num_bits = 0) {
    const auto hash = hasher.GetHash(key);
    static_assert(sizeof(typename SimpleSolutionStorage::Index) ==
                      sizeof(typename Hasher::Index),
                  "must be same");
    static_assert(sizeof(typename SimpleSolutionStorage::CoeffRow) ==
                      sizeof(typename Hasher::CoeffRow),
                  "must be same");

    // don't query an empty ribbon, please
    assert(sss.GetNumSlots() >= Hasher::kCoeffBits);

    return ShiftQueryHelper(hash, hasher, sss, start_idx, num_bits);
}

// General retrieval query a key from SimpleSolutionStorage.
template <typename SimpleSolutionStorage, typename Hasher>
std::pair<typename SimpleSolutionStorage::ResultRowVLR, typename SimpleSolutionStorage::ResultRowVLR>
SimpleRetrievalQueryVLR(const typename HashTraits<Hasher>::mhc_or_key_t &key,
                     const Hasher &hasher, const SimpleSolutionStorage &sss,
                     typename SimpleSolutionStorage::ResultRowVLR bump_mask) {
    const auto hash = hasher.GetHash(key);
    static_assert(sizeof(typename SimpleSolutionStorage::Index) ==
                      sizeof(typename Hasher::Index),
                  "must be same");
    static_assert(sizeof(typename SimpleSolutionStorage::CoeffRow) ==
                      sizeof(typename Hasher::CoeffRow),
                  "must be same");

    // don't query an empty ribbon, please
    assert(sss.GetNumSlots() >= Hasher::kCoeffBits);

    return ShiftQueryHelperVLR(hash, hasher, sss, bump_mask);
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

    auto [bumped, retrieved] = ShiftQueryHelper(hash, hasher, sss);
    sLOG << "Key" << tlx::wrap_unprintable(key) << "b?" << bumped << "retrieved"
         << std::hex << (size_t)retrieved << "expected" << (size_t)expected
         << std::dec;
    return std::make_pair(bumped, retrieved == expected);
}


/******************************************************************************/


// General retrieval query a key from InterleavedSolutionStorage.
template <typename InterleavedSolutionStorage, typename Hasher>
std::pair<bool, std::conditional_t<InterleavedSolutionStorage::kUseVLR,
                                   typename InterleavedSolutionStorage::ResultRowVLR,
                                   typename InterleavedSolutionStorage::ResultRow>>
InterleavedRetrievalQuery(const typename HashTraits<Hasher>::mhc_or_key_t &key,
                          const Hasher &hasher,
                          const InterleavedSolutionStorage &iss,
                          [[maybe_unused]] typename InterleavedSolutionStorage::Index start_idx = 0,
                          [[maybe_unused]] typename InterleavedSolutionStorage::Index num_bits = 0) {
    constexpr bool kUseVLR = InterleavedSolutionStorage::kUseVLR;
    constexpr bool kVLRShareMeta = InterleavedSolutionStorage::kVLRShareMeta;
    constexpr bool kVLRFlipOutputBits = InterleavedSolutionStorage::kVLRFlipOutputBits;
    using Hash = typename Hasher::Hash;
    using Index = typename InterleavedSolutionStorage::Index;
    using CoeffRow = typename InterleavedSolutionStorage::CoeffRow;
    using ResultRow = typename InterleavedSolutionStorage::ResultRow;
    using ResultRowVLR = typename InterleavedSolutionStorage::ResultRowVLR;

    static_assert(sizeof(Index) == sizeof(typename Hasher::Index),
                  "must be same");
    static_assert(sizeof(CoeffRow) == sizeof(typename Hasher::CoeffRow),
                  "must be same");

    constexpr bool debug = false;
    constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U);

    // don't query an empty ribbon, please
    assert(iss.GetNumSlots() >= kCoeffBits);
    const Hash hash = hasher.GetHash(key);
    const Index start_slot = hasher.GetStart(hash, iss.GetNumStarts());
    const Index bucket = hasher.GetBucket(start_slot);

    const Index start_block_num = start_slot / kCoeffBits;
    constexpr Index num_columns = InterleavedSolutionStorage::kResultBits;
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

    if constexpr (kUseVLR && kVLRShareMeta) {
        Index num_ribbons = iss.GetNumRibbons();
        Index num_vlr_columns = std::min(num_ribbons, static_cast<Index>(sizeof(ResultRowVLR) * 8U));
        const Index start_ribbon = hasher.GetVLRIndex(hash, num_ribbons);
        ResultRowVLR value = 0;
        if (num_bits == 0) {
            start_idx = 0;
            num_bits = num_vlr_columns;
        }
        assert(num_bits + start_idx <= num_vlr_columns);
        int index;
        if constexpr (kVLRFlipOutputBits)
            index = start_idx;
        else
            index = sizeof(ResultRowVLR) * 8U - 1 - start_idx;
        for (Index bit = start_idx; bit < start_idx + num_bits; ++bit) {
            const Index ribbon_idx = (start_ribbon + bit) % num_ribbons;
            ResultRow sr = 0;
            const CoeffRow cr_left = cr << start_bit;
            sr ^= rocksdb::BitParity(iss.GetSegment(segment, ribbon_idx) & cr_left);
            // FIXME: move this to separate loop for cache efficiency
            if (start_bit > 0) {
                const CoeffRow cr_right = cr >> (kCoeffBits - start_bit);
                sr ^= rocksdb::BitParity(iss.GetSegment(segment + 1, ribbon_idx) & cr_right);
            }
            value |= static_cast<ResultRowVLR>(sr) << index;
            if constexpr (kVLRFlipOutputBits)
                ++index;
            else
                --index;
        }
        return std::make_pair(false, value);
    } else {
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

}

// General retrieval query a key from InterleavedSolutionStorage.
template <typename InterleavedSolutionStorage, typename Hasher>
std::pair<typename InterleavedSolutionStorage::ResultRowVLR, typename InterleavedSolutionStorage::ResultRowVLR>
InterleavedRetrievalQueryVLR(const typename HashTraits<Hasher>::mhc_or_key_t &key,
                             const Hasher &hasher, const InterleavedSolutionStorage &iss,
                             typename InterleavedSolutionStorage::ResultRowVLR bump_mask) {
    using Hash = typename Hasher::Hash;
    using Index = typename InterleavedSolutionStorage::Index;
    using CoeffRow = typename InterleavedSolutionStorage::CoeffRow;
    using ResultRow = typename InterleavedSolutionStorage::ResultRow;
    using ResultRowVLR = typename InterleavedSolutionStorage::ResultRowVLR;
    constexpr bool kVLRFlipOutputBits = InterleavedSolutionStorage::kVLRFlipOutputBits;

    static_assert(sizeof(Index) == sizeof(typename Hasher::Index),
                  "must be same");
    static_assert(sizeof(CoeffRow) == sizeof(typename Hasher::CoeffRow),
                  "must be same");

    constexpr bool debug = false;
    constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U);
    static_assert(InterleavedSolutionStorage::kResultBits == 1);

    // don't query an empty ribbon, please
    assert(iss.GetNumSlots() >= kCoeffBits);
    const Hash hash = hasher.GetHash(key);
    const Index start_slot = hasher.GetStart(hash, iss.GetNumStarts());
    const Index num_ribbons = iss.GetNumRibbons();
    const Index start_ribbon = hasher.GetVLRIndex(hash, num_ribbons);
    const Index bucket = hasher.GetBucket(start_slot);

    const Index segment = start_slot / kCoeffBits;
    iss.PrefetchQuery(segment);

    const Index val = hasher.GetIntraBucketFromStart(start_slot),
                cval = hasher.Compress(val);

    ResultRowVLR new_mask = 0;
    ResultRowVLR value = 0;
    while (bump_mask) {
        unsigned int bump_first;
        if constexpr (kVLRFlipOutputBits)
            bump_first = rocksdb::CountTrailingZeroBits(bump_mask);
        else
            bump_first = rocksdb::CountLeadingZeroBits(bump_mask);
        // FIXME: maybe avoid this if by just setting the mask properly in ribbon.hpp
        // values cannot contain more bits than there are ribbons
        if (bump_first >= num_ribbons)
            break;
        const Index ribbon_idx = (start_ribbon + bump_first) % num_ribbons;
        int index;
        if constexpr (kVLRFlipOutputBits)
            index = bump_first;
        else
            index = sizeof(ResultRowVLR) * 8U - bump_first - 1;
        if (CheckBumpedVLR(val, cval, bucket, hasher, iss, ribbon_idx)) {
            sLOG << "Item was bumped, hash" << hash << "start" << start_slot
                 << "bucket" << bucket << "val" << val << cval << "thresh"
                 << (size_t)iss.GetMeta(bucket) << "ribbon" << ribbon_idx;
            new_mask |= ResultRowVLR(1) << index;
            bump_mask &= ~(ResultRowVLR(1) << index);
            continue;

        }
        bump_mask &= ~(ResultRowVLR(1) << index);

        sLOG << "Searching in bucket" << bucket << "start" << start_slot << "val"
             << val << cval << "below thresh =" << (size_t)iss.GetMeta(bucket)
             << "ribbon" << ribbon_idx;

        const Index start_bit = start_slot % kCoeffBits;
        const CoeffRow cr = hasher.GetCoeffs(hash);

        ResultRow sr = 0;
        const CoeffRow cr_left = cr << start_bit;
        sr ^= rocksdb::BitParity(iss.GetSegment(segment, ribbon_idx) & cr_left);

        if (start_bit > 0) {
            const CoeffRow cr_right = cr >> (kCoeffBits - start_bit);
            sr ^= rocksdb::BitParity(iss.GetSegment(segment + 1, ribbon_idx) & cr_right);
        }
        value |= static_cast<ResultRowVLR>(sr) << index;
    }

    return std::make_pair(new_mask, value);
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
