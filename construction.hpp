//  Copyright (c) Lorenz Hübschle-Schneider
//  Copyright (c) Facebook, Inc. and its affiliates.
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "minimal_hasher.hpp"
#include "rocksdb/math.h"
#include "rocksdb/stop_watch.h"
#include "sorter.hpp"

#include <tlx/logger.hpp>
#include <tlx/logger/wrap_unprintable.hpp>
#include <tlx/define/likely.hpp>

#ifndef RIBBON_USE_STD_SORT
// Use in-place super-scalar radix sorter ips2ra, which is around 3x faster for
// the inputs used here
#include <ips2ra.hpp>
#endif

#include <algorithm>
#include <cassert>
#include <functional>
#include <tuple>
#include <vector>

namespace ribbon {

template <bool kFCA1, typename BandingStorage>
std::pair<bool, typename BandingStorage::Index>
BandingAdd(BandingStorage *bs, typename BandingStorage::Index start,
           typename BandingStorage::CoeffRow coeffs,
           typename BandingStorage::ResultRow result, size_t ribbon_idx = 0) {
    using Index = typename BandingStorage::Index;
    constexpr bool debug = false;

    Index pos = start;
    if constexpr (!kFCA1) {
        int tz = rocksdb::CountTrailingZeroBits(coeffs);
        pos += static_cast<size_t>(tz);
        coeffs >>= tz;
    }

    while (true) {
        assert(pos < bs->GetNumSlots());
        assert((coeffs & 1) == 1);

        typename BandingStorage::CoeffRow other;
        if constexpr (BandingStorage::kUseVLR)
            other = bs->GetCoeffs(pos, ribbon_idx);
        else
            other = bs->GetCoeffs(pos);
        if (other == 0) {
            // found an empty slot, insert
            if constexpr (BandingStorage::kUseVLR) {
                bs->SetCoeffs(pos, coeffs, ribbon_idx);
                bs->SetResult(pos, result, ribbon_idx);
            } else {
                bs->SetCoeffs(pos, coeffs);
                bs->SetResult(pos, result);
            }
            sLOG << "Insertion succeeded at position" << pos;
            return std::make_pair(true, pos);
        }

        assert((other & 1) == 1);
        coeffs ^= other;
        if constexpr (BandingStorage::kUseVLR)
            result ^= bs->GetResult(pos, ribbon_idx);
        else
            result ^= bs->GetResult(pos);
        if (coeffs == 0) {
            // linearly dependent!
            sLOG << "Insertion failed at position" << pos;
            return std::make_pair(false, pos);
        }

        // move to now-leading coefficient
        int tz = rocksdb::CountTrailingZeroBits(coeffs);
        pos += static_cast<size_t>(tz);
        coeffs >>= tz;
    }
    LOG << "Insertion hit the end of the loop";
    return std::pair(result == 0, pos);
}

// hack to prevent inlining of ips2ra, which is awful for compile time and
// produces ginormous binaries
template <typename Iterator>
__attribute__((noinline)) void my_sort(Iterator begin, Iterator end) {
#ifdef RIBBON_USE_STD_SORT
    // Use std::sort as a slow fallback
    std::sort(begin, end, [](const auto &a, const auto &b) {
        return std::get<0>(a) < std::get<0>(b);
    });
#else
    ips2ra::sort(begin, end, [](const auto &x) { return std::get<0>(x); });
#endif
}

// hack to prevent inlining of ips2ra, which is awful for compile time and
// produces ginormous binaries
template <typename Iterator, typename Hasher, typename Index>
__attribute__((noinline)) void my_sort(Iterator begin, Iterator end,
                                       const Hasher &h, Index num_starts) {
    unsigned sparse_shift = 0;
    if constexpr (Hasher::kSparseCoeffs) {
        sparse_shift = Hasher::shift_;
    }
    MinimalHasher<Index, Hasher::kSparseCoeffs> mh(Hasher::kBucketSize,
                                                   h.GetFactor(), sparse_shift);
    return Sorter<Index, Hasher::kIsFilter, Hasher::kSparseCoeffs,
                  typename std::iterator_traits<Iterator>::value_type>()
        .do_sort(begin, end, mh, num_starts);
}

template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
bool BandingAddRange(BandingStorage *bs, Hasher &hasher, Iterator begin,
                     Iterator end, BumpStorage *bump_vec, typename BandingStorage::Index num_ribbons = 1) {
    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using ResultRow = typename BandingStorage::ResultRow;
    using ResultRowVLR = typename BandingStorage::ResultRowVLR;
    using Hash = typename Hasher::Hash;
    constexpr bool kFCA1 = Hasher::kFirstCoeffAlwaysOne;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;
    constexpr bool kUseVLR = BandingStorage::kUseVLR;
    constexpr bool kVLRShareMeta = BandingStorage::kVLRShareMeta;
    constexpr bool kVLRFlipInputBits = BandingStorage::kVLRFlipInputBits;
    static_assert(!kUseVLR || kVLRShareMeta);

    constexpr bool debug = false;
    constexpr bool log = Hasher::log;

    if (begin == end)
        return true;

    rocksdb::StopWatchNano timer(true);
    const Index num_starts = bs->GetNumStarts();
    const Index num_buckets = bs->GetNumBuckets();
    sLOG << "Constructing ribbon with" << num_buckets
         << "buckets,  num_starts = " << num_starts;

    const auto num_items = end - begin; // std::distance(begin, end);
#ifdef RIBBON_PASS_HASH
    constexpr bool sparse = Hasher::kSparseCoeffs && Hasher::kCoeffBits < 128;
    auto input = std::make_unique<std::tuple<Index, Index, std::conditional_t<sparse, uint32_t, Hash>>[]>(num_items);
#else
    auto input = std::make_unique<std::pair<Index, Index>[]>(num_items);
#endif

    {
        sLOG << "Processing" << num_items << "items";

        for (Index i = 0; i < static_cast<Index>(num_items); i++) {
            const Hash h = hasher.GetHash(*(begin + i));
            const Index start = hasher.GetStart(h, num_starts);
            const Index sortpos = Hasher::StartToSort(start);
#ifdef RIBBON_PASS_HASH
            if constexpr (sparse) {
                uint32_t compact_hash = hasher.GetCompactHash(h);
                input[i] = std::make_tuple(sortpos, i, compact_hash);
            } else {
                input[i] = std::make_tuple(sortpos, i, h);
            }
#else
            input[i] = std::make_pair(sortpos, i);
#endif
        }
    }
    LOGC(log) << "\tInput transformation took "
              << timer.ElapsedNanos(true) / 1e6 << "ms";
    my_sort(input.get(), input.get() + num_items);
    LOGC(log) << "\tSorting took " << timer.ElapsedNanos(true) / 1e6 << "ms";

    [[maybe_unused]] const auto do_bump_vlr = [&](auto &rows, auto &indeces) {
        sLOG << "Bumping" << indeces.size() << "items";
        assert(kUseVLR);
        for (auto idx : indeces) {
            sLOG << "\tBumping item"
                 << tlx::wrap_unprintable(*(begin + idx));
            bump_vec->push_back(*(begin + idx));
        }
        indeces.clear();
        for (Index i = 0; i < rows.size(); ++i) {
            for (auto row : rows[i]) {
                bs->SetCoeffs(row, 0, i);
                bs->SetResult(row, 0, i);
            }
            rows[i].clear();
        }
    };
    [[maybe_unused]] const auto do_bump = [&](auto &vec) {
        sLOG << "Bumping" << vec.size() << "items";
        assert(!kUseVLR);
        for (auto [row, idx] : vec) {
            sLOG << "\tBumping row" << row << "item"
                 << tlx::wrap_unprintable(*(begin + idx));
            bs->SetCoeffs(row, 0);
            bs->SetResult(row, 0);
            bump_vec->push_back(*(begin + idx));
        }
        vec.clear();
    };

    Index last_bucket = 0;
    bool all_good = true;
    Index thresh = Hasher::NoBumpThresh();
    // Bump cache (row, input item) pairs that may have to be bumped retroactively
    Index last_cval = -1;
    [[maybe_unused]] std::conditional_t<kUseVLR, int, std::vector<std::pair<Index, Index>>> bump_cache;
    [[maybe_unused]] std::conditional_t<kUseVLR, std::vector<std::vector<Index>>, int> vlr_rows(num_ribbons);
    [[maybe_unused]] std::conditional_t<kUseVLR, std::vector<Index>, int> vlr_indeces;
    // For 1-bit thresholds, we also need an uncompressed bump cache for undoing
    // all insertions with the same uncompressed value if we end up in the
    // "plus" case with a separately stored threshold
    [[maybe_unused]] Index last_val = -1;
    [[maybe_unused]] std::conditional_t<oneBitThresh && !kUseVLR, decltype(bump_cache), int> unc_bump_cache;
    [[maybe_unused]] std::conditional_t<oneBitThresh && kUseVLR, decltype(vlr_rows), int> vlr_unc_rows(num_ribbons);
    [[maybe_unused]] std::conditional_t<oneBitThresh && kUseVLR, decltype(vlr_indeces), int> vlr_unc_indeces;

#ifndef RIBBON_PASS_HASH
    auto next = *(begin + input[0].second);
#endif

    for (Index i = 0; i < static_cast<Index>(num_items); ++i) {
#ifdef RIBBON_PASS_HASH
        const auto [sortpos, idx, hash] = input[i];
#else
        const auto [sortpos, idx] = input[i];
#endif
        const Index start = Hasher::SortToStart(sortpos),
                    bucket = Hasher::GetBucket(sortpos),
                    val = Hasher::GetIntraBucket(sortpos),
                    cval = hasher.Compress(val);
        assert(bucket >= last_bucket);
        assert(oneBitThresh || cval < Hasher::NoBumpThresh());

#ifndef RIBBON_PASS_HASH
        const Hash hash = hasher.GetHash(next);
        if (i + 1 < num_items)
            next = *(begin + input[i + 1].second);

        // prefetch the cache miss far in advance, assuming the iterator
        // is to contiguous storage
        if (TLX_LIKELY(i + 32 < num_items))
            __builtin_prefetch(&*begin + input[i + 32].second, 0, 1);
#endif

        if (bucket != last_bucket) {
            // moving to next bucket
            sLOG << "Moving to bucket" << bucket << "was" << last_bucket;
            if constexpr (oneBitThresh) {
                if constexpr (kUseVLR) {
                    for (auto &rows : vlr_unc_rows) {
                        rows.clear();
                    }
                    vlr_unc_indeces.clear();
                } else {
                    unc_bump_cache.clear();
                }
                last_val = val;
            }
            if (thresh == Hasher::NoBumpThresh()) {
                sLOG << "Bucket" << last_bucket << "has no bumped items";
                bs->SetMeta(last_bucket, thresh);
            }
            all_good = true;
            last_bucket = bucket;
            thresh = Hasher::NoBumpThresh(); // maximum == "no bumpage"
            last_cval = cval;
            if constexpr (kUseVLR) {
                for (auto &rows : vlr_rows) {
                    rows.clear();
                }
                vlr_indeces.clear();
            } else {
                bump_cache.clear();
            }
        } else if (!all_good) {
            // direct hard bump
            sLOG << "Directly bumping" << tlx::wrap_unprintable(*(begin + idx))
                 << "from bucket" << bucket << "val" << val << cval << "start"
                 << start << "sort" << sortpos << "hash" << std::hex << hash
                 << "data"
                 << (uint64_t)(Hasher::kIsFilter
                                   ? hasher.GetResultRowFromHash(hash)
                                   : hasher.GetResultRowFromInput(*(begin + idx)))
                 << std::dec;
            bump_vec->push_back(*(begin + idx));
            continue;
        } else if (cval != last_cval) {
            // clear bump cache
            sLOG << "Bucket" << bucket << "cval" << cval << "!=" << last_cval;
            if constexpr (kUseVLR) {
                for (auto &rows : vlr_rows) {
                    rows.clear();
                }
                vlr_indeces.clear();
            } else {
                bump_cache.clear();
            }
            last_cval = cval;
        }
        if constexpr (oneBitThresh) {
            // split into constexpr and normal if because unc_bump_cache isn't a
            // vector if !oneBitThresh
            if (val != last_val) {
                if constexpr (kUseVLR) {
                    for (auto &rows : vlr_unc_rows) {
                        rows.clear();
                    }
                    vlr_unc_indeces.clear();
                } else {
                    unc_bump_cache.clear();
                }
                last_val = val;
            }
        }


        const CoeffRow cr = hasher.GetCoeffs(hash);

        if constexpr (kUseVLR) {
            const Index ribbon_start = hasher.GetVLRIndex(hash, num_ribbons);
            const ResultRowVLR rr = hasher.GetResultRowVLRFromInput(*(begin + idx));

            // there must be at least a 1 to mark the beginning of the actual value
            assert(rr != 0);
            // the first 1 is not part of the actual value
            const int num_zeroes = rocksdb::CountLeadingZeroBits(rr) + 1;
            const int num_bits = (sizeof(ResultRowVLR) * 8) - num_zeroes;
            assert(num_bits != 0);

            vlr_indeces.push_back(idx);
            if constexpr (oneBitThresh)
                vlr_unc_indeces.push_back(idx);

            for (int bit = 0; bit < num_bits; ++bit) {
                const Index ribbon_idx = (ribbon_start + bit) % num_ribbons;
                int shift;
                if constexpr (kVLRFlipInputBits)
                    shift = bit;
                else
                    shift = num_bits - bit - 1;

                auto [success, row] = BandingAdd<kFCA1>(bs, start, cr, static_cast<ResultRow>((rr >> shift) & 0x1), ribbon_idx);
                if (!success) {
                    assert(all_good);
                    if (bump_vec == nullptr)
                        // bumping disabled, abort!
                        return false;
                    // if we got this far, this is the first failure in this bucket
                    // (for this ribbon_idx), and we need to undo insertions with the same cval
                    sLOG << "ribbon " << ribbon_idx << ": First failure in bucket" << bucket
                         << "val" << val << "start" << start << "sort" << sortpos << "hash"
                         << std::hex << hash << "data" << (uint64_t)rr << std::dec
                         << "for item" << tlx::wrap_unprintable(*(begin + idx))
                         << "-> threshold" << cval << "clash in row" << row;
                    thresh = cval;
                    if constexpr (oneBitThresh) {
                        if (cval == 2) {
                            sLOG << "ribbon " << ribbon_idx << ": First failure in bucket" << bucket
                                 << "val" << val << "is a 'plus' case (below threshold)";
                            // "plus" case: store uncompressed threshold in hash table
                            hasher.Set(bucket, val);
                            // Set meta to 0 (some bumpage) but keep thresh at 2 so that
                            // we don't retroactively bump everything when moving to the
                            // next bucket
                            bs->SetMeta(bucket, 0);
                            all_good = false;

                            // bump all items with the same uncompressed value
                            do_bump_vlr(vlr_unc_rows, vlr_unc_indeces);
                            // proceed to next item, don't do regular bumping (but only
                            // if cval == 2, not generally!)
                            break;
                        }
                    }
                    bs->SetMeta(bucket, thresh);
                    all_good = false;

                    do_bump_vlr(vlr_rows, vlr_indeces);
                    break;
                } else {
                    sLOG << "ribbon " << ribbon_idx << ": Insertion succeeded of item"
                         << tlx::wrap_unprintable(*(begin + idx)) << "in pos" << row
                         << "bucket" << bucket << "val" << val << cval << "start"
                         << start << "sort" << sortpos << "hash" << std::hex << hash
                         << "data" << (uint64_t)rr << std::dec;
                    vlr_rows[ribbon_idx].push_back(row);
                    if constexpr (oneBitThresh) {
                        // also record in bump cache for uncompressed values
                        vlr_unc_rows[ribbon_idx].push_back(row);
                    }
                }
            }
        } else {
            const ResultRow rr = Hasher::kIsFilter
                                     ? hasher.GetResultRowFromHash(hash)
                                     : hasher.GetResultRowFromInput(*(begin + idx));

            auto [success, row] = BandingAdd<kFCA1>(bs, start, cr, rr);
            if (!success) {
                assert(all_good);
                if (bump_vec == nullptr)
                    // bumping disabled, abort!
                    return false;
                // if we got this far, this is the first failure in this bucket,
                // and we need to undo insertions with the same cval
                sLOG << "First failure in bucket" << bucket << "val" << val
                     << "start" << start << "sort" << sortpos << "hash" << std::hex
                     << hash << "data" << (uint64_t)rr << std::dec << "for item"
                     << tlx::wrap_unprintable(*(begin + idx)) << "-> threshold"
                     << cval << "clash in row" << row;
                thresh = cval;
                if constexpr (oneBitThresh) {
                    if (cval == 2) {
                        sLOG << "First failure in bucket" << bucket << "val" << val
                             << "is a 'plus' case (below threshold)";
                        // "plus" case: store uncompressed threshold in hash table
                        hasher.Set(bucket, val);
                        // Set meta to 0 (some bumpage) but keep thresh at 2 so that
                        // we don't retroactively bump everything when moving to the
                        // next bucket
                        bs->SetMeta(bucket, 0);
                        all_good = false;

                        // bump all items with the same uncompressed value
                        do_bump(unc_bump_cache);
                        sLOG << "Also bumping"
                             << tlx::wrap_unprintable(*(begin + idx));
                        bump_vec->push_back(*(begin + idx));
                        // proceed to next item, don't do regular bumping (but only
                        // if cval == 2, not generally!)
                        continue;
                    }
                }
                bs->SetMeta(bucket, thresh);
                all_good = false;

                do_bump(bump_cache);
                bump_vec->push_back(*(begin + idx));
            } else {
                sLOG << "Insertion succeeded of item"
                     << tlx::wrap_unprintable(*(begin + idx)) << "in pos" << row
                     << "bucket" << bucket << "val" << val << cval << "start"
                     << start << "sort" << sortpos << "hash" << std::hex << hash
                     << "data" << (uint64_t)rr << std::dec;
                bump_cache.emplace_back(row, idx);
                if constexpr (oneBitThresh) {
                    // also record in bump cache for uncompressed values
                    unc_bump_cache.emplace_back(row, idx);
                }
            }
        }
    }
    // set final threshold
    if (thresh == Hasher::NoBumpThresh()) {
        bs->SetMeta(last_bucket, thresh);
    }

    // migrate thresholds to hash table
    if constexpr (oneBitThresh) {
        hasher.Finalise(num_buckets);
    }

    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return true;
}

template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
bool BandingAddRangeVLR(BandingStorage *bs, Hasher &hasher, Iterator begin, Iterator end,
                        BumpStorage *bump_vec, typename BandingStorage::Index num_ribbons) {
    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using Key = typename BandingStorage::Key;
    using ResultRow = typename BandingStorage::ResultRow;
    using ResultRowVLR = typename BandingStorage::ResultRowVLR;
    using Hash = typename Hasher::Hash;
    constexpr bool kFCA1 = Hasher::kFirstCoeffAlwaysOne;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;
    // in toplevel ribbon, there is no bump mask
    constexpr bool toplevel = std::is_same_v<typename std::iterator_traits<Iterator>::value_type, std::pair<Key, ResultRowVLR>>;
    constexpr bool kVLRFlipInputBits = BandingStorage::kVLRFlipInputBits;

    constexpr bool debug = false;
    constexpr bool log = Hasher::log;

    if (begin == end)
        return true;

    rocksdb::StopWatchNano timer(true);
    const Index num_starts = bs->GetNumStarts();
    const Index num_buckets = bs->GetNumBuckets();
    sLOG << "Constructing ribbon with" << num_buckets
         << "buckets,  num_starts = " << num_starts;

    const auto num_items = end - begin; // std::distance(begin, end);

#ifdef RIBBON_PASS_HASH
    constexpr bool sparse = Hasher::kSparseCoeffs && Hasher::kCoeffBits < 128;
    auto input = std::make_unique<
        std::tuple<Index, Index, std::conditional_t<sparse, uint32_t, Hash>>[]>(
        num_items);
#else
    auto input = std::make_unique<std::pair<Index, Index>[]>(num_items);
#endif

    {
        sLOG << "Processing" << num_items << "items";

        for (Index i = 0; i < static_cast<Index>(num_items); i++) {
            const Hash h = hasher.GetHash(*(begin + i));
            const Index start = hasher.GetStart(h, num_starts);
            const Index sortpos = Hasher::StartToSort(start);
#ifdef RIBBON_PASS_HASH
            if constexpr (sparse) {
                uint32_t compact_hash = hasher.GetCompactHash(h);
                input[i] = std::make_tuple(sortpos, i, compact_hash);
            } else {
                input[i] = std::make_tuple(sortpos, i, h);
            }
#else
            input[i] = std::make_pair(sortpos, i);
#endif
        }
    }
    LOGC(log) << "\tInput transformation took "
              << timer.ElapsedNanos(true) / 1e6 << "ms";
    my_sort(input.get(), input.get() + num_items);
    LOGC(log) << "\tSorting took " << timer.ElapsedNanos(true) / 1e6 << "ms";

    const auto do_bump = [&](auto &vec) {
        for (auto& [idx, start_pos, mask, num_bits] : vec) {
            if (mask) {
                const auto itm = *(begin + idx);
                bump_vec->emplace_back(std::get<0>(itm), std::get<1>(itm), mask);
            }
        }
        vec.clear();
    };

    const auto adjust_mask = [num_ribbons](auto &vec, auto &mask_vec, size_t start, size_t stop) {
        for (size_t i = start; i < stop; ++i) {
            unsigned int new_num_bits = std::get<3>(vec[i]);
            auto new_start_pos = std::get<1>(vec[i]);
            while (new_num_bits > 0) {
                auto index = new_start_pos / 64;
                uint64_t num = mask_vec[index];
                auto sz = (index == mask_vec.size() - 1 && num_ribbons % 64) ? num_ribbons % 64 : 64;
                // should always be the case because new_start_pos < num_ribbons
                assert(new_start_pos % 64 < sz);
                if (sz - (new_start_pos % 64) >= new_num_bits) {
                    std::get<2>(vec[i]) |= static_cast<ResultRowVLR>((num << new_start_pos % 64) >> (64 - new_num_bits));
                    new_num_bits = 0;
                } else {
                    auto cur_num_bits = sz - (new_start_pos % 64);
                    std::get<2>(vec[i]) |= static_cast<ResultRowVLR>((num << new_start_pos % 64) >> (64 - cur_num_bits)) << (new_num_bits - cur_num_bits);
                    new_start_pos = (new_start_pos + cur_num_bits) % num_ribbons;
                    new_num_bits -= cur_num_bits;
                }
            }
        }
    };

    Index last_bucket = 0;
    std::vector<uint64_t> already_bumped((num_ribbons + 63) / 64, 0);
    [[maybe_unused]] std::conditional_t<oneBitThresh, std::vector<uint64_t>, int> unc_already_bumped;
    if constexpr (oneBitThresh)
        unc_already_bumped.resize((num_ribbons + 63) / 64, 0);
    std::vector<Index> thresh(num_ribbons, Hasher::NoBumpThresh());

    Index last_cval = -1;
    std::vector<std::tuple<Index, size_t, ResultRowVLR, int>> bump_cache;
    // the rows need to be kept separately for each 1-bit ribbon because they will
    // generally be different, so they aren't stored in bump_cache directly
    std::vector<std::vector<Index>> row_cache(num_ribbons);
    // For 1-bit thresholds, we also need an uncompressed bump cache for undoing
    // all insertions with the same uncompressed value if we end up in the
    // "plus" case with a separately stored threshold
    [[maybe_unused]] Index last_val = -1;
    [[maybe_unused]] std::conditional_t<oneBitThresh, std::vector<size_t>, int> unc_row_num;
    if constexpr (oneBitThresh)
        unc_row_num.resize(num_ribbons, 0);
    [[maybe_unused]] size_t num_same_val = 0;

#ifndef RIBBON_PASS_HASH
    auto next = *(begin + input[0].second);
#endif

    for (Index i = 0; i < static_cast<Index>(num_items); ++i) {
#ifdef RIBBON_PASS_HASH
        const auto [sortpos, idx, hash] = input[i];
#else
        const auto [sortpos, idx] = input[i];
#endif
        const Index start = Hasher::SortToStart(sortpos),
                    bucket = Hasher::GetBucket(sortpos),
                    val = Hasher::GetIntraBucket(sortpos),
                    cval = hasher.Compress(val);
        assert(bucket >= last_bucket);
        assert(oneBitThresh || cval < Hasher::NoBumpThresh());

#ifndef RIBBON_PASS_HASH
        const Hash hash = hasher.GetHash(next);
        if (i + 1 < num_items)
            next = *(begin + input[i + 1].second);

        // prefetch the cache miss far in advance, assuming the iterator
        // is to contiguous storage
        if (TLX_LIKELY(i + 32 < num_items))
            __builtin_prefetch(&*begin + input[i + 32].second, 0, 1);
#endif

        if (bucket != last_bucket) {
            // moving to next bucket
            sLOG << "Moving to bucket" << bucket << "was" << last_bucket;
            if constexpr (oneBitThresh) {
                last_val = val;
                num_same_val = 0;
                std::fill(unc_already_bumped.begin(), unc_already_bumped.end(), 0);
            }
            do_bump(bump_cache);
            for (Index ribbon_idx = 0; ribbon_idx < num_ribbons; ++ribbon_idx) {
                if (thresh[ribbon_idx] == Hasher::NoBumpThresh()) {
                    sLOG << "ribbon " << ribbon_idx << ": Bucket" << last_bucket << "has no bumped items";
                    bs->SetMeta(last_bucket, thresh[ribbon_idx], ribbon_idx);
                }
                row_cache[ribbon_idx].clear();
                if constexpr (oneBitThresh)
                    unc_row_num[ribbon_idx] = 0;
            }
            std::fill(already_bumped.begin(), already_bumped.end(), 0);
            last_bucket = bucket;
            // maximum == "no bumpage"
            std::fill(thresh.begin(), thresh.end(), Hasher::NoBumpThresh());
            last_cval = cval;
        } else if (cval != last_cval) {
            // clear bump cache
            sLOG << "Bucket" << bucket << "cval" << cval << "!=" << last_cval;
            do_bump(bump_cache);
            for (Index ribbon_idx = 0; ribbon_idx < num_ribbons; ++ribbon_idx) {
                row_cache[ribbon_idx].clear();
            }
            last_cval = cval;
        }
        if constexpr (oneBitThresh) {
            // split into constexpr and normal if because unc_row_num isn't a
            // vector if !oneBitThresh
            if (val != last_val) {
                last_val = val;
                num_same_val = 0;
                std::fill(unc_row_num.begin(), unc_row_num.end(), 0);
                std::fill(unc_already_bumped.begin(), unc_already_bumped.end(), 0);
            }
        }

        ResultRowVLR cur_bucket_mask = 0;
        const CoeffRow cr = hasher.GetCoeffs(hash);
        const ResultRowVLR rr = hasher.GetResultRowVLRFromInput(*(begin + idx));
        // there must be at least a 1 to mark the beginning of the actual value
        assert(rr != 0);
        // the first 1 is not part of the actual value
        const int num_zeroes = rocksdb::CountLeadingZeroBits(rr) + 1;
        const int num_bits = (sizeof(ResultRowVLR) * 8) - num_zeroes;
        assert(num_bits != 0);
        // in top-level, prev_mask is nullptr since nothing has been bumped before
        ResultRowVLR prev_bumped;
        if constexpr (toplevel)
            prev_bumped = ~ResultRowVLR(0);
        else
            prev_bumped = std::get<2>(*(begin + idx));

        const Index ribbon_start = hasher.GetVLRIndex(hash, num_ribbons);

        bool any_bumped = false;
        [[maybe_unused]] bool unc_any_bumped = false;

        for (int bit = 0; bit < num_bits; ++bit) {
            const Index ribbon_idx = (ribbon_start + bit) % num_ribbons;
            int shift;
            // the bump mask is still in the same bit order to make adjust_mask simpler
            // (if it was flipped as well, adjust_mask and already_bumped would also
            //  need to be changed as well, which would make everything more complicated)
            int bump_shift;
            if constexpr (kVLRFlipInputBits) {
                shift = bit;
                bump_shift = num_bits - bit - 1;
            } else {
                shift = num_bits - bit - 1;
                bump_shift = shift;
            }
            // if the element wasn't bumped previously, it doesn't need to be inserted
            if (!((prev_bumped >> bump_shift) & 0x1)) {
                continue;
            } else if ((already_bumped[ribbon_idx / 64] >> (63 - (ribbon_idx % 64))) & 0x1) {
                // direct hard bump for this ribbon_idx
                cur_bucket_mask |= ResultRowVLR(1) << bump_shift;
                continue;
            }

            auto [success, row] = BandingAdd<kFCA1>(bs, start, cr, static_cast<ResultRow>((rr >> shift) & 0x1), ribbon_idx);
            if (!success) {
                if (bump_vec == nullptr)
                    // bumping disabled, abort!
                    return false;
                // if we got this far, this is the first failure in this bucket
                // (for this ribbon_idx), and we need to undo insertions with the same cval
                sLOG << "ribbon " << ribbon_idx << ": First failure in bucket" << bucket
                     << "val" << val << "start" << start << "sort" << sortpos << "hash"
                     << std::hex << hash << "data" << (uint64_t)rr << std::dec
                     << "for item" << tlx::wrap_unprintable(*(begin + idx))
                     << "-> threshold" << cval << "clash in row" << row;
                thresh[ribbon_idx] = cval;
                if constexpr (oneBitThresh) {
                    if (cval == 2) {
                        sLOG << "ribbon " << ribbon_idx << ": First failure in bucket" << bucket
                             << "val" << val << "is a 'plus' case (below threshold)";
                        // "plus" case: store uncompressed threshold in hash table
                        hasher.Set(bucket, val, ribbon_idx);
                        // Set meta to 0 (some bumpage) but keep thresh at 2 so that
                        // we don't retroactively bump everything when moving to the
                        // next bucket
                        bs->SetMeta(bucket, 0, ribbon_idx);
                        unc_already_bumped[ribbon_idx / 64] |= uint64_t(1) << (63 - (ribbon_idx % 64));
                        unc_any_bumped = true;

                        // bump all items with the same uncompressed value
                        const size_t sz = row_cache[ribbon_idx].size();
                        for (size_t j = sz - unc_row_num[ribbon_idx]; j < sz; ++j) {
                            bs->SetCoeffs(row_cache[ribbon_idx][j], 0, ribbon_idx);
                            bs->SetResult(row_cache[ribbon_idx][j], 0, ribbon_idx);
                        }
                        sLOG << "Also bumping"
                             << tlx::wrap_unprintable(*(begin + idx));
                        // proceed to next item, don't do regular bumping (but only
                        // if cval == 2, not generally!)
                        continue;
                    }
                }
                bs->SetMeta(bucket, thresh[ribbon_idx], ribbon_idx);
                already_bumped[ribbon_idx / 64] |= uint64_t(1) << (63 - (ribbon_idx % 64));

                for (auto row : row_cache[ribbon_idx]) {
                    bs->SetCoeffs(row, 0, ribbon_idx);
                    bs->SetResult(row, 0, ribbon_idx);
                }
                any_bumped = true;
            } else {
                sLOG << "ribbon " << ribbon_idx << ": Insertion succeeded of item"
                     << tlx::wrap_unprintable(*(begin + idx)) << "in pos" << row
                     << "bucket" << bucket << "val" << val << cval << "start"
                     << start << "sort" << sortpos << "hash" << std::hex << hash
                     << "data" << (uint64_t)rr << std::dec;
                row_cache[ribbon_idx].emplace_back(row);
                if constexpr (oneBitThresh) {
                    // also record in bump cache for uncompressed values
                    unc_row_num[ribbon_idx]++;
                }
            }
        }
        // we need to save num_bits as well so adjust_mask knows which parts
        // are actually relevant (without having to redo a bunch of calculations)
        bump_cache.emplace_back(idx, ribbon_start, cur_bucket_mask, num_bits);
        if constexpr (oneBitThresh)
            num_same_val++;
        if constexpr (oneBitThresh) {
            if (any_bumped) {
                adjust_mask(bump_cache, already_bumped, 0, bump_cache.size() - num_same_val);
            }
            if (unc_any_bumped) {
                for (size_t index = 0; index < already_bumped.size(); ++index) {
                    already_bumped[index] |= unc_already_bumped[index];
                }
            }
            if (any_bumped || unc_any_bumped) {
                adjust_mask(bump_cache, already_bumped, bump_cache.size() - num_same_val, bump_cache.size());
            }
        } else {
            if (any_bumped)
                adjust_mask(bump_cache, already_bumped, 0, bump_cache.size());
        }
    }

    // set final threshold
    for (Index i = 0; i < num_ribbons; ++i) {
        if (thresh[i] == Hasher::NoBumpThresh()) {
            bs->SetMeta(last_bucket, thresh[i], i);
        }
    }
    // bump elements with non-zero mask from last bucket
    do_bump(bump_cache);

    // migrate thresholds to hash table
    if constexpr (oneBitThresh) {
        hasher.Finalise(num_buckets);
    }

    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return true;
}


template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename Hasher::mhc_t>>
bool BandingAddRangeMHC(BandingStorage *bs, Hasher &hasher, Iterator begin,
                        Iterator end, BumpStorage *bump_vec, typename BandingStorage::Index num_ribbons = 1) {
    static_assert(Hasher::kUseMHC, "you called the wrong method");

    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using ResultRow = typename BandingStorage::ResultRow;
    using ResultRowVLR = typename BandingStorage::ResultRowVLR;
    constexpr bool kFCA1 = Hasher::kFirstCoeffAlwaysOne;
    constexpr bool oneBitThresh = (Hasher::kThreshMode == ThreshMode::onebit);
    constexpr bool kUseVLR = BandingStorage::kUseVLR;
    constexpr bool kVLRShareMeta = BandingStorage::kVLRShareMeta;
    static_assert(!kUseVLR || kVLRShareMeta);
    constexpr bool kVLRFlipInputBits = BandingStorage::kVLRFlipInputBits;

    constexpr bool debug = false;
    constexpr bool log = Hasher::log;

    if (begin == end)
        return true;

    rocksdb::StopWatchNano timer(true);
    const auto num_items = end - begin;
    const Index num_starts = bs->GetNumStarts();
    const Index num_buckets = bs->GetNumBuckets();
    sLOG << "Constructing ribbon (MHC) with" << num_buckets
         << "buckets, num_starts =" << num_starts;

    my_sort(begin, end, hasher, num_starts);
    // MHCs should be unique, if not, fail construction
    if constexpr (Hasher::kIsFilter) {
        assert(std::adjacent_find(begin, end) == end);
    } else {
        assert(std::adjacent_find(begin, end, [](const auto &a, const auto &b) {
                   return a.first == b.first;
               }) == end);
    }

    LOGC(log) << "\tSorting took " << timer.ElapsedNanos(true) / 1e6 << "ms";

    [[maybe_unused]] const auto do_bump_vlr = [&](auto &rows, auto &indeces) {
        sLOG << "Bumping" << indeces.size() << "items";
        assert(kUseVLR);
        for (auto idx : indeces) {
            sLOG << "\tBumping item"
                 << tlx::wrap_unprintable(*(begin + idx));
            bump_vec->push_back(*(begin + idx));
        }
        indeces.clear();
        for (Index i = 0; i < rows.size(); ++i) {
            for (auto row : rows[i]) {
                bs->SetCoeffs(row, 0, i);
                bs->SetResult(row, 0, i);
            }
            rows[i].clear();
        }
    };
    [[maybe_unused]] const auto do_bump = [&](auto &vec) {
        sLOG << "Bumping" << vec.size() << "items";
        for (auto [row, idx] : vec) {
            sLOG << "\tBumping row" << row << "item"
                 << tlx::wrap_unprintable(*(begin + idx));
            bs->SetCoeffs(row, 0);
            bs->SetResult(row, 0);
            bump_vec->push_back(*(begin + idx));
        }
        vec.clear();
    };

    Index last_bucket = 0;
    bool all_good = true;
    Index thresh = Hasher::NoBumpThresh();
    // Bump cache (row, input item) pairs that may have to be bumped retroactively
    Index last_cval = -1;
    [[maybe_unused]] std::conditional_t<kUseVLR, int, std::vector<std::pair<Index, Index>>> bump_cache;
    [[maybe_unused]] std::conditional_t<kUseVLR, std::vector<std::vector<Index>>, int> vlr_rows(num_ribbons);
    [[maybe_unused]] std::conditional_t<kUseVLR, std::vector<Index>, int> vlr_indeces;

    // For 1-bit thresholds, we also need an uncompressed bump cache for undoing
    // all insertions with the same uncompressed value if we end up in the
    // "plus" case with a separately stored threshold
    [[maybe_unused]] Index last_val = -1;
    [[maybe_unused]] std::conditional_t<oneBitThresh && !kUseVLR, decltype(bump_cache), int> unc_bump_cache;
    [[maybe_unused]] std::conditional_t<oneBitThresh && kUseVLR, decltype(vlr_rows), int> vlr_unc_rows(num_ribbons);
    [[maybe_unused]] std::conditional_t<oneBitThresh && kUseVLR, decltype(vlr_indeces), int> vlr_unc_indeces;

    for (Index i = 0; i < num_items; ++i) {
        const auto mhc = *(begin + i);
        const auto hash = hasher.GetHash(mhc);
        const Index start = hasher.GetStart(hash, num_starts),
                    sortpos = Hasher::SortToStart(start),
                    bucket = Hasher::GetBucket(sortpos),
                    val = Hasher::GetIntraBucket(sortpos),
                    cval = hasher.Compress(val);
        assert(bucket >= last_bucket);
        assert(oneBitThresh || cval < Hasher::NoBumpThresh());

        if (bucket != last_bucket) {
            // moving to next bucket
            sLOG << "Moving to bucket" << bucket << "was" << last_bucket;
            if constexpr (oneBitThresh) {
                if constexpr (kUseVLR) {
                    for (auto &rows : vlr_unc_rows) {
                        rows.clear();
                    }
                    vlr_unc_indeces.clear();
                } else {
                    unc_bump_cache.clear();
                }
                last_val = val;
            }
            if (thresh == Hasher::NoBumpThresh()) {
                sLOG << "Bucket" << last_bucket << "has no bumped items";
                bs->SetMeta(last_bucket, thresh);
            }
            all_good = true;
            last_bucket = bucket;
            thresh = Hasher::NoBumpThresh(); // maximum == "no bumpage"
            last_cval = cval;
            if constexpr (kUseVLR) {
                for (auto &rows : vlr_rows) {
                    rows.clear();
                }
                vlr_indeces.clear();
            } else {
                bump_cache.clear();
            }
        } else if (!all_good) {
            // direct hard bump
            sLOG << "Directly bumping" << tlx::wrap_unprintable(*(begin + i))
                 << "from bucket" << bucket << "val" << val << cval << "start"
                 << start << "sort" << sortpos << "hash" << std::hex << hash
                 << std::dec;
            bump_vec->push_back(*(begin + i));
            continue;
        } else if (cval != last_cval) {
            // clear bump cache
            sLOG << "Bucket" << bucket << "cval" << cval << "!=" << last_cval;
            if constexpr (kUseVLR) {
                for (auto &rows : vlr_rows) {
                    rows.clear();
                }
                vlr_indeces.clear();
            } else {
                bump_cache.clear();
            }
            last_cval = cval;
        }
        if constexpr (oneBitThresh) {
            // split into constexpr and normal if because unc_bump_cache isn't a
            // vector if !oneBitThresh
            if (val != last_val) {
                if constexpr (kUseVLR) {
                    for (auto &rows : vlr_unc_rows) {
                        rows.clear();
                    }
                    vlr_unc_indeces.clear();
                } else {
                    unc_bump_cache.clear();
                }
                last_val = val;
            }
        }


        const CoeffRow cr = hasher.GetCoeffs(hash);

        if constexpr (kUseVLR) {
            const Index ribbon_start = hasher.GetVLRIndex(hash, num_ribbons);
            const ResultRowVLR rr = hasher.GetResultRowVLRFromInput(*(begin + i));

            // there must be at least a 1 to mark the beginning of the actual value
            assert(rr != 0);
            // the first 1 is not part of the actual value
            const int num_zeroes = rocksdb::CountLeadingZeroBits(rr) + 1;
            const int num_bits = (sizeof(ResultRowVLR) * 8) - num_zeroes;
            assert(num_bits != 0);

            vlr_indeces.push_back(i);
            if constexpr (oneBitThresh)
                vlr_unc_indeces.push_back(i);

            for (int bit = 0; bit < num_bits; ++bit) {
                const Index ribbon_idx = (ribbon_start + bit) % num_ribbons;
                int shift;
                if constexpr (kVLRFlipInputBits)
                    shift = bit;
                else
                    shift = num_bits - bit - 1;

                auto [success, row] = BandingAdd<kFCA1>(bs, start, cr, static_cast<ResultRow>((rr >> shift) & 0x1), ribbon_idx);
                if (!success) {
                    assert(all_good);
                    if (bump_vec == nullptr)
                        // bumping disabled, abort!
                        return false;
                    // if we got this far, this is the first failure in this bucket
                    // (for this ribbon_idx), and we need to undo insertions with the same cval
                    sLOG << "ribbon " << ribbon_idx << ": First failure in bucket" << bucket
                         << "val" << val << "start" << start << "sort" << sortpos << "hash"
                         << std::hex << hash << "data" << (uint64_t)rr << std::dec
                         << "for item" << tlx::wrap_unprintable(*(begin + i))
                         << "-> threshold" << cval << "clash in row" << row;
                    thresh = cval;
                    if constexpr (oneBitThresh) {
                        if (cval == 2) {
                            sLOG << "ribbon " << ribbon_idx << ": First failure in bucket" << bucket
                                 << "val" << val << "is a 'plus' case (below threshold)";
                            // "plus" case: store uncompressed threshold in hash table
                            hasher.Set(bucket, val);
                            // Set meta to 0 (some bumpage) but keep thresh at 2 so that
                            // we don't retroactively bump everything when moving to the
                            // next bucket
                            bs->SetMeta(bucket, 0);
                            all_good = false;

                            // bump all items with the same uncompressed value
                            do_bump_vlr(vlr_unc_rows, vlr_unc_indeces);
                            // proceed to next item, don't do regular bumping (but only
                            // if cval == 2, not generally!)
                            break;
                        }
                    }
                    bs->SetMeta(bucket, thresh);
                    all_good = false;

                    do_bump_vlr(vlr_rows, vlr_indeces);
                    break;
                } else {
                    sLOG << "ribbon " << ribbon_idx << ": Insertion succeeded of item"
                         << tlx::wrap_unprintable(*(begin + i)) << "in pos" << row
                         << "bucket" << bucket << "val" << val << cval << "start"
                         << start << "sort" << sortpos << "hash" << std::hex << hash
                         << "data" << (uint64_t)rr << std::dec;
                    vlr_rows[ribbon_idx].push_back(row);
                    if constexpr (oneBitThresh) {
                        // also record in bump cache for uncompressed values
                        vlr_unc_rows[ribbon_idx].push_back(row);
                    }
                }
            }
        } else {
            const ResultRow rr = Hasher::kIsFilter
                                     ? hasher.GetResultRowFromHash(hash)
                                     : hasher.GetResultRowFromInput(*(begin + i));

            auto [success, row] = BandingAdd<kFCA1>(bs, start, cr, rr);
            if (!success) {
                assert(all_good);
                if (bump_vec == nullptr)
                    // bumping disabled, abort!
                    return false;
                // if we got this far, this is the first failure in this bucket,
                // and we need to undo insertions with the same cval
                sLOG << "First failure in bucket" << bucket << "val" << val
                     << "start" << start << "sort" << sortpos << "hash" << std::hex
                     << hash << "data" << (uint64_t)rr << std::dec << "for item"
                     << tlx::wrap_unprintable(*(begin + i)) << "-> threshold"
                     << cval << "clash in row" << row;
                thresh = cval;
                if constexpr (oneBitThresh) {
                    if (cval == 2) {
                        sLOG << "First failure in bucket" << bucket << "val" << val
                             << "is a 'plus' case (below threshold)";
                        // "plus" case: store uncompressed threshold in hash table
                        hasher.Set(bucket, val);
                        // Set meta to 0 (some bumpage) but keep thresh at 2 so that
                        // we don't retroactively bump everything when moving to the
                        // next bucket
                        bs->SetMeta(bucket, 0);
                        all_good = false;

                        // bump all items with the same uncompressed value
                        do_bump(unc_bump_cache);
                        sLOG << "Also bumping"
                             << tlx::wrap_unprintable(*(begin + i));
                        bump_vec->push_back(*(begin + i));
                        // proceed to next item, don't do regular bumping (but only
                        // if cval == 2, not generally!)
                        continue;
                    }
                }
                bs->SetMeta(bucket, thresh);
                all_good = false;

                do_bump(bump_cache);
                bump_vec->push_back(*(begin + i));
            } else {
                sLOG << "Insertion succeeded of item"
                     << tlx::wrap_unprintable(*(begin + i)) << "in pos" << row
                     << "bucket" << bucket << "val" << val << cval << "start"
                     << start << "sort" << sortpos << "hash" << std::hex << hash
                     << "data" << (uint64_t)rr << std::dec;
                bump_cache.emplace_back(row, i);
                if constexpr (oneBitThresh) {
                    // also record in bump cache for uncompressed values
                    unc_bump_cache.emplace_back(row, i);
                }
            }
        }
    }
    // set final threshold
    if (thresh == Hasher::NoBumpThresh()) {
        bs->SetMeta(last_bucket, thresh);
    }

    // migrate thresholds to hash table
    if constexpr (oneBitThresh) {
        hasher.Finalise(num_buckets);
    }

    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return true;
}

template <typename BandingStorage, typename Hasher, typename Iterator, typename BumpStorage>
bool BandingAddRangeMHCVLR(BandingStorage *bs, Hasher &hasher, Iterator begin, Iterator end,
                           BumpStorage *bump_vec, typename BandingStorage::Index num_ribbons) {
    static_assert(Hasher::kUseMHC && Hasher::kUseVLR && !Hasher::kVLRShareMeta && !Hasher::kIsFilter, "you called the wrong method");

    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using ResultRow = typename BandingStorage::ResultRow;
    using ResultRowVLR = typename BandingStorage::ResultRowVLR;
    constexpr bool kFCA1 = Hasher::kFirstCoeffAlwaysOne;
    constexpr bool oneBitThresh = (Hasher::kThreshMode == ThreshMode::onebit);
    // in toplevel ribbon, there is no bump mask
    constexpr bool toplevel = std::is_same_v<typename std::iterator_traits<Iterator>::value_type, std::pair<typename Hasher::mhc_t, ResultRowVLR>>;
    constexpr bool kVLRFlipInputBits = BandingStorage::kVLRFlipInputBits;

    constexpr bool debug = false;
    constexpr bool log = Hasher::log;

    if (begin == end)
        return true;

    rocksdb::StopWatchNano timer(true);
    const auto num_items = end - begin;
    const Index num_starts = bs->GetNumStarts();
    const Index num_buckets = bs->GetNumBuckets();
    sLOG << "Constructing ribbon (MHC) with" << num_buckets
         << "buckets, num_starts =" << num_starts;

    my_sort(begin, end, hasher, num_starts);
    // MHCs should be unique, if not, fail construction
    assert(std::adjacent_find(begin, end, [](const auto &a, const auto &b) {
               return std::get<0>(a) == std::get<0>(b);
           }) == end);

    LOGC(log) << "\tSorting took " << timer.ElapsedNanos(true) / 1e6 << "ms";

    const auto do_bump = [&](auto &vec) {
        for (auto& [idx, start_pos, mask, num_bits] : vec) {
            if (mask) {
                const auto mhc = *(begin + idx);
                bump_vec->emplace_back(std::get<0>(mhc), std::get<1>(mhc), mask);
            }
        }
        vec.clear();
    };

    const auto adjust_mask = [num_ribbons](auto &vec, auto &mask_vec, size_t start, size_t stop) {
        for (size_t i = start; i < stop; ++i) {
            unsigned int new_num_bits = std::get<3>(vec[i]);
            auto new_start_pos = std::get<1>(vec[i]);
            while (new_num_bits > 0) {
                auto index = new_start_pos / 64;
                uint64_t num = mask_vec[index];
                auto sz = (index == mask_vec.size() - 1 && num_ribbons % 64) ? num_ribbons % 64 : 64;
                // should always be the case because new_start_pos < num_ribbons
                assert(new_start_pos % 64 < sz);
                if (sz - (new_start_pos % 64) >= new_num_bits) {
                    std::get<2>(vec[i]) |= static_cast<ResultRowVLR>((num << new_start_pos % 64) >> (64 - new_num_bits));
                    new_num_bits = 0;
                } else {
                    auto cur_num_bits = sz - (new_start_pos % 64);
                    std::get<2>(vec[i]) |= static_cast<ResultRowVLR>((num << new_start_pos % 64) >> (64 - cur_num_bits)) << (new_num_bits - cur_num_bits);
                    new_start_pos = (new_start_pos + cur_num_bits) % num_ribbons;
                    new_num_bits -= cur_num_bits;
                }
            }
        }
    };

    Index last_bucket = 0;
    std::vector<uint64_t> already_bumped((num_ribbons + 63) / 64, 0);
    [[maybe_unused]] std::conditional_t<oneBitThresh, std::vector<uint64_t>, int> unc_already_bumped;
    if constexpr (oneBitThresh)
        unc_already_bumped.resize((num_ribbons + 63) / 64, 0);
    std::vector<Index> thresh(num_ribbons, Hasher::NoBumpThresh());

    Index last_cval = -1;
    std::vector<std::tuple<Index, size_t, ResultRowVLR, int>> bump_cache;
    // the rows need to be kept separately for each 1-bit ribbon because they will
    // generally be different, so they aren't stored in bump_cache directly
    std::vector<std::vector<Index>> row_cache(num_ribbons);
    // For 1-bit thresholds, we also need an uncompressed bump cache for undoing
    // all insertions with the same uncompressed value if we end up in the
    // "plus" case with a separately stored threshold
    [[maybe_unused]] Index last_val = -1;
    [[maybe_unused]] std::conditional_t<oneBitThresh, std::vector<size_t>, int> unc_row_num;
    if constexpr (oneBitThresh)
        unc_row_num.resize(num_ribbons, 0);
    [[maybe_unused]] size_t num_same_val = 0;

    for (Index i = 0; i < num_items; ++i) {
        const auto mhc = *(begin + i);
        const auto hash = hasher.GetHash(mhc);
        const Index start = hasher.GetStart(hash, num_starts),
                    sortpos = Hasher::SortToStart(start),
                    bucket = Hasher::GetBucket(sortpos),
                    val = Hasher::GetIntraBucket(sortpos),
                    cval = hasher.Compress(val);
        assert(bucket >= last_bucket);
        assert(oneBitThresh || cval < Hasher::NoBumpThresh());

        if (bucket != last_bucket) {
            // moving to next bucket
            sLOG << "Moving to bucket" << bucket << "was" << last_bucket;
            if constexpr (oneBitThresh) {
                last_val = val;
                num_same_val = 0;
                std::fill(unc_already_bumped.begin(), unc_already_bumped.end(), 0);
            }
            do_bump(bump_cache);
            for (Index ribbon_idx = 0; ribbon_idx < num_ribbons; ++ribbon_idx) {
                if (thresh[ribbon_idx] == Hasher::NoBumpThresh()) {
                    sLOG << "ribbon " << ribbon_idx << ": Bucket" << last_bucket << "has no bumped items";
                    bs->SetMeta(last_bucket, thresh[ribbon_idx], ribbon_idx);
                }
                row_cache[ribbon_idx].clear();
                if constexpr (oneBitThresh)
                    unc_row_num[ribbon_idx] = 0;
            }
            std::fill(already_bumped.begin(), already_bumped.end(), 0);
            last_bucket = bucket;
            // maximum == "no bumpage"
            std::fill(thresh.begin(), thresh.end(), Hasher::NoBumpThresh());
            last_cval = cval;
        } else if (cval != last_cval) {
            // clear bump cache
            sLOG << "Bucket" << bucket << "cval" << cval << "!=" << last_cval;
            do_bump(bump_cache);
            for (Index ribbon_idx = 0; ribbon_idx < num_ribbons; ++ribbon_idx) {
                row_cache[ribbon_idx].clear();
            }
            last_cval = cval;
        }
        if constexpr (oneBitThresh) {
            // split into constexpr and normal if because unc_row_num isn't a
            // vector if !oneBitThresh
            if (val != last_val) {
                last_val = val;
                num_same_val = 0;
                std::fill(unc_row_num.begin(), unc_row_num.end(), 0);
                std::fill(unc_already_bumped.begin(), unc_already_bumped.end(), 0);
            }
        }

        ResultRowVLR cur_bucket_mask = 0;
        const CoeffRow cr = hasher.GetCoeffs(hash);
        const ResultRowVLR rr = hasher.GetResultRowVLRFromInput(mhc);
        // there must be at least a 1 to mark the beginning of the actual value
        assert(rr != 0);
        // the first 1 is not part of the actual value
        const int num_zeroes = rocksdb::CountLeadingZeroBits(rr) + 1;
        const int num_bits = (sizeof(ResultRowVLR) * 8) - num_zeroes;
        assert(num_bits != 0);
        // in top-level, prev_mask is nullptr since nothing has been bumped before
        ResultRowVLR prev_bumped;
        // in toplevel, everything has to be inserted, otherwise the input contains the bump mask
        if constexpr (toplevel)
            prev_bumped = ~ResultRowVLR(0);
        else
            prev_bumped = std::get<2>(mhc);
        const Index ribbon_start = hasher.GetVLRIndex(hash, num_ribbons);

        bool any_bumped = false;
        [[maybe_unused]] bool unc_any_bumped = false;

        for (int bit = 0; bit < num_bits; ++bit) {
            const Index ribbon_idx = (ribbon_start + bit) % num_ribbons;
            int shift;
            // the bump mask is still in the same bit order to make adjust_mask simpler
            // (if it was flipped as well, adjust_mask and already_bumped would also
            //  need to be changed as well, which would make everything more complicated)
            int bump_shift;
            if constexpr (kVLRFlipInputBits) {
                shift = bit;
                bump_shift = num_bits - bit - 1;
            } else {
                shift = num_bits - bit - 1;
                bump_shift = shift;
            }
            // if the element wasn't bumped previously, it doesn't need to be inserted
            if (!((prev_bumped >> bump_shift) & 0x1)) {
                continue;
            } else if ((already_bumped[ribbon_idx / 64] >> (63 - (ribbon_idx % 64))) & 0x1) {
                // direct hard bump for this ribbon_idx
                cur_bucket_mask |= ResultRowVLR(1) << bump_shift;
                continue;
            }

            auto [success, row] = BandingAdd<kFCA1>(bs, start, cr, static_cast<ResultRow>((rr >> shift) & 0x1), ribbon_idx);
            if (!success) {
                if (bump_vec == nullptr)
                    // bumping disabled, abort!
                    return false;
                // if we got this far, this is the first failure in this bucket
                // (for this ribbon_idx), and we need to undo insertions with the same cval
                sLOG << "ribbon " << ribbon_idx << ": First failure in bucket" << bucket
                     << "val" << val << "start" << start << "sort" << sortpos << "hash"
                     << std::hex << hash << "data" << (uint64_t)rr << std::dec
                     << "for item" << tlx::wrap_unprintable(mhc)
                     << "-> threshold" << cval << "clash in row" << row;
                thresh[ribbon_idx] = cval;
                if constexpr (oneBitThresh) {
                    if (cval == 2) {
                        sLOG << "ribbon " << ribbon_idx << ": First failure in bucket" << bucket
                             << "val" << val << "is a 'plus' case (below threshold)";
                        // "plus" case: store uncompressed threshold in hash table
                        hasher.Set(bucket, val, ribbon_idx);
                        // Set meta to 0 (some bumpage) but keep thresh at 2 so that
                        // we don't retroactively bump everything when moving to the
                        // next bucket
                        bs->SetMeta(bucket, 0, ribbon_idx);
                        unc_already_bumped[ribbon_idx / 64] |= uint64_t(1) << (63 - (ribbon_idx % 64));
                        unc_any_bumped = true;

                        // bump all items with the same uncompressed value
                        const size_t sz = row_cache[ribbon_idx].size();
                        for (size_t j = sz - unc_row_num[ribbon_idx]; j < sz; ++j) {
                            bs->SetCoeffs(row_cache[ribbon_idx][j], 0, ribbon_idx);
                            bs->SetResult(row_cache[ribbon_idx][j], 0, ribbon_idx);
                        }
                        sLOG << "Also bumping"
                             << tlx::wrap_unprintable(mhc);
                        // proceed to next item, don't do regular bumping (but only
                        // if cval == 2, not generally!)
                        continue;
                    }
                }
                bs->SetMeta(bucket, thresh[ribbon_idx], ribbon_idx);
                already_bumped[ribbon_idx / 64] |= uint64_t(1) << (63 - (ribbon_idx % 64));

                for (auto row : row_cache[ribbon_idx]) {
                    bs->SetCoeffs(row, 0, ribbon_idx);
                    bs->SetResult(row, 0, ribbon_idx);
                }
                any_bumped = true;
            } else {
                sLOG << "ribbon " << ribbon_idx << ": Insertion succeeded of item"
                     << tlx::wrap_unprintable(mhc) << "in pos" << row
                     << "bucket" << bucket << "val" << val << cval << "start"
                     << start << "sort" << sortpos << "hash" << std::hex << hash
                     << "data" << (uint64_t)rr << std::dec;
                row_cache[ribbon_idx].emplace_back(row);
                if constexpr (oneBitThresh) {
                    // also record in bump cache for uncompressed values
                    unc_row_num[ribbon_idx]++;
                }
            }
        }
        // we need to save num_bits as well so adjust_mask knows which parts
        // are actually relevant (without having to redo a bunch of calculations)
        bump_cache.emplace_back(i, ribbon_start, cur_bucket_mask, num_bits);
        if constexpr (oneBitThresh)
            num_same_val++;
        if constexpr (oneBitThresh) {
            if (any_bumped) {
                adjust_mask(bump_cache, already_bumped, 0, bump_cache.size() - num_same_val);
            }
            if (unc_any_bumped) {
                for (size_t index = 0; index < already_bumped.size(); ++index) {
                    already_bumped[index] |= unc_already_bumped[index];
                }
            }
            if (any_bumped || unc_any_bumped) {
                adjust_mask(bump_cache, already_bumped, bump_cache.size() - num_same_val, bump_cache.size());
            }
        } else {
            if (any_bumped)
                adjust_mask(bump_cache, already_bumped, 0, bump_cache.size());
        }
    }

    // set final threshold
    for (Index i = 0; i < num_ribbons; ++i) {
        if (thresh[i] == Hasher::NoBumpThresh()) {
            bs->SetMeta(last_bucket, thresh[i], i);
        }
    }
    // bump elements with non-zero mask from last bucket
    do_bump(bump_cache);

    // migrate thresholds to hash table
    if constexpr (oneBitThresh) {
        hasher.Finalise(num_buckets);
    }

    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return true;
}

} // namespace ribbon
