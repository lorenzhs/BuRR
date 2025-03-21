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
#else
#include <execution>
#endif

#include <algorithm>
#include <cassert>
#include <functional>
#include <tuple>
#include <vector>

#ifdef _REENTRANT
#include <mutex>
#include <thread>
#endif

namespace ribbon {

template <bool kFCA1, typename BandingStorage>
std::pair<bool, typename BandingStorage::Index>
BandingAdd(BandingStorage *bs, typename BandingStorage::Index start,
           typename BandingStorage::CoeffRow coeffs,
           typename BandingStorage::ResultRow result) {
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

        auto other = bs->GetCoeffs(pos);
        if (other == 0) {
            // found an empty slot, insert
            bs->SetCoeffs(pos, coeffs);
            bs->SetResult(pos, result);
            sLOG << "Insertion succeeded at position" << pos;
            return std::make_pair(true, pos);
        }

        assert((other & 1) == 1);
        coeffs ^= other;
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
__attribute__((noinline)) void my_sort(Iterator begin, Iterator end, std::size_t num_threads = 0) {
#ifdef RIBBON_USE_STD_SORT
    // Use std::sort as a slow fallback
    if (num_threads <= 1) {
        std::sort(begin, end, [](const auto &a, const auto &b) {
            return std::get<0>(a) < std::get<0>(b);
        });
    } else {
        std::sort(std::execution::par_unseq, begin, end, [](const auto &a, const auto &b) {
            return std::get<0>(a) < std::get<0>(b);
        });
    }
#else
    if (num_threads <= 1)
        ips2ra::sort(begin, end, [](const auto &x) { return std::get<0>(x); });
    #ifdef _REENTRANT
    else
        ips2ra::parallel::sort(begin, end, [](const auto &x) { return std::get<0>(x); }, num_threads);
    #else
    else {
        std::cerr << "Parallel version called but not compiled in. This should be impossible.\n";
        abort(); /* should never happen */
    }
    #endif
#endif
}

// hack to prevent inlining of ips2ra, which is awful for compile time and
// produces ginormous binaries
template <typename Iterator, typename Hasher, typename Index>
__attribute__((noinline)) void my_sort(Iterator begin, Iterator end,
                                       const Hasher &h, Index num_starts,
                                       std::size_t num_threads = 0) {
    unsigned sparse_shift = 0;
    if constexpr (Hasher::kSparseCoeffs) {
        sparse_shift = Hasher::shift_;
    }
    MinimalHasher<Index, Hasher::kSparseCoeffs> mh(Hasher::kBucketSize,
                                                   h.GetFactor(), sparse_shift);
    return Sorter<Index, Hasher::kIsFilter, Hasher::kSparseCoeffs,
                  std::conditional_t<Hasher::kIsFilter, SorterDummyData,
                                     typename Hasher::ResultRow>>()
        .do_sort(begin, end, mh, num_starts, num_threads);
}

template <typename BandingStorage, typename Hasher, typename Iterator, typename Input>
void ProcessInput(
    BandingStorage *bs, Hasher &hasher, Input &input, Iterator begin,
    typename BandingStorage::Index start_idx,
    typename BandingStorage::Index end_idx) {

    using Index = typename BandingStorage::Index;
    using Hash = typename Hasher::Hash;
    const Index num_starts = bs->GetNumStarts();
#ifdef RIBBON_PASS_HASH
    constexpr bool sparse = Hasher::kSparseCoeffs && Hasher::kCoeffBits < 128;
#endif
    for (Index i = start_idx; i < end_idx; ++i) {
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

template <bool parallel, typename BandingStorage, typename Hasher, typename Iterator, typename Input,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
bool AddRangeInternal(
    BandingStorage *bs, Hasher &hasher,
    typename BandingStorage::Index start_bucket,
    typename BandingStorage::Index end_bucket,
    typename BandingStorage::Index start_index,
    typename BandingStorage::Index end_index,
    Input &input, Iterator begin, BumpStorage *bump_vec, std::size_t thread_index,
    std::size_t num_threads,
    std::conditional_t<parallel, std::vector<std::mutex> &, int> border_mutexes,
    std::conditional_t<parallel && Hasher::kThreshMode == ThreshMode::onebit, std::mutex &, int> hash_mtx) {

    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using ResultRow = typename BandingStorage::ResultRow;
    using Hash = typename Hasher::Hash;
    constexpr bool kFCA1 = Hasher::kFirstCoeffAlwaysOne;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;
    [[maybe_unused]] constexpr Index kBucketSize = BandingStorage::kBucketSize;
    [[maybe_unused]] constexpr Index kCoeffBits = BandingStorage::kCoeffBits;

    constexpr bool debug = false;

    [[maybe_unused]] Index safe_start_bucket;
    [[maybe_unused]] Index safe_end_bucket;
    if constexpr (parallel) {
        safe_start_bucket = bs->GetNextSafeStart(start_bucket);
        safe_end_bucket = bs->GetPrevSafeEnd(end_bucket);
    }

    Index last_bucket = start_bucket;
    bool all_good = true;
    Index thresh = Hasher::NoBumpThresh();
    // Bump cache (row, input item) pairs that may have to be bumped retroactively
    Index last_cval = -1;
    std::vector<std::pair<Index, Index>> bump_cache;
    // For 1-bit thresholds, we also need an uncompressed bump cache for undoing
    // all insertions with the same uncompressed value if we end up in the
    // "plus" case with a separately stored threshold
    [[maybe_unused]] Index last_val = -1;
    [[maybe_unused]] std::conditional_t<oneBitThresh, decltype(bump_cache), int> unc_bump_cache;
    [[maybe_unused]] bool start_locked = false;
    [[maybe_unused]] bool end_locked = false;
    [[maybe_unused]] Index bump_start;
    if constexpr (parallel) {
        if (thread_index > 0) {
            border_mutexes[thread_index - 1].lock();
            start_locked = true;
        }
        bump_start = thread_index == 0
                     ? start_bucket * BandingStorage::kBucketSize
                     : start_bucket * BandingStorage::kBucketSize + BandingStorage::kCoeffBits - 1;
    }

    const auto do_bump = [&](auto &vec) {
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

#ifndef RIBBON_PASS_HASH
    auto next = *(begin + input[start_index].second);
#endif

    for (Index i = start_index; i < end_index; ++i) {
#ifdef RIBBON_PASS_HASH
        const auto [sortpos, idx, hash] = input[i];
#else
        const auto [sortpos, idx] = input[i];
#endif
        const Index start = Hasher::SortToStart(sortpos),
                    val = Hasher::GetIntraBucket(sortpos),
                    cval = hasher.Compress(val),
                    bucket = Hasher::GetBucket(sortpos);
        assert(bucket >= last_bucket);
        assert(oneBitThresh || cval < Hasher::NoBumpThresh());

#ifndef RIBBON_PASS_HASH
        const Hash hash = hasher.GetHash(next);
        if (i + 1 < end_index)
            next = *(begin + input[i + 1].second);

        // prefetch the cache miss far in advance, assuming the iterator
        // is to contiguous storage
        if (TLX_LIKELY(i + 32 < end_index))
            __builtin_prefetch(&*begin + input[i + 32].second, 0, 1);
#endif

        if (bucket != last_bucket) {
            // moving to next bucket
            sLOG << "Moving to bucket" << bucket << "was" << last_bucket;
            if constexpr (parallel) {
                if (start_locked && bucket >= safe_start_bucket) {
                    border_mutexes[thread_index - 1].unlock();
                    start_locked = false;
                }
                if (thread_index < num_threads - 1 && !end_locked && bucket > safe_end_bucket) {
                    border_mutexes[thread_index].lock();
                    end_locked = true;
                }
            }
            if constexpr (oneBitThresh) {
                unc_bump_cache.clear();
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
            bump_cache.clear();
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
            bump_cache.clear();
            last_cval = cval;
        }
        if constexpr (oneBitThresh) {
            // split into constexpr and normal if because unc_bump_cache isn't a
            // vector if !oneBitThresh
            if (val != last_val) {
                unc_bump_cache.clear();
                last_val = val;
            }
        }


        const CoeffRow cr = hasher.GetCoeffs(hash);
        const ResultRow rr = Hasher::kIsFilter
                                 ? hasher.GetResultRowFromHash(hash)
                                 : hasher.GetResultRowFromInput(*(begin + idx));

        /* The first kCoeffBits-1 rows of the first bucket are bumped in the parallel case */
        bool success;
        Index row;
        if constexpr (parallel) {
            std::tie(success, row) = start < bump_start
                                  ? std::make_pair(false, start)
                                  : BandingAdd<kFCA1>(bs, start, cr, rr);
        } else {
            std::tie(success, row) = BandingAdd<kFCA1>(bs, start, cr, rr);
        }
        /*
        const auto [success, row] = [&]() {
            if constexpr (parallel) {
                return start < bump_start
                       ? std::make_pair(false, start)
                       : BandingAdd<kFCA1>(bs, start, cr, rr);
            } else {
                return BandingAdd<kFCA1>(bs, start, cr, rr);
            }
        }();
        */

        if (!success) {
            assert(all_good);
            if constexpr (!parallel) {
                // bumping disabled, abort!
                if (bump_vec == nullptr) {
                    return false;
                }
            }
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
                    if constexpr (parallel) {
                        std::scoped_lock lock(hash_mtx);
                        hasher.Set(bucket, val);
                    } else {
                        hasher.Set(bucket, val);
                    }
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
    // set final threshold
    if (thresh == Hasher::NoBumpThresh()) {
        bs->SetMeta(last_bucket, thresh);
    }

    if constexpr (parallel) {
        if (end_locked) {
            border_mutexes[thread_index].unlock();
        }
        if (start_locked) {
            border_mutexes[thread_index - 1].unlock();
        }
    }

    return true;
}

template <bool parallel, typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
bool AddRangeInternalMHC(
    BandingStorage *bs, Hasher &hasher,
    typename BandingStorage::Index start_bucket,
    typename BandingStorage::Index end_bucket,
    typename BandingStorage::Index start_index,
    typename BandingStorage::Index end_index,
    Iterator begin, BumpStorage *bump_vec, std::size_t thread_index,
    std::size_t num_threads,
    std::conditional_t<parallel, std::vector<std::mutex> &, int> border_mutexes,
    std::conditional_t<parallel && Hasher::kThreshMode == ThreshMode::onebit, std::mutex &, int> hash_mtx) {

    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using ResultRow = typename BandingStorage::ResultRow;
    constexpr bool kFCA1 = Hasher::kFirstCoeffAlwaysOne;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;
    [[maybe_unused]] constexpr Index kBucketSize = BandingStorage::kBucketSize;
    [[maybe_unused]] constexpr Index kCoeffBits = BandingStorage::kCoeffBits;

    constexpr bool debug = false;

    [[maybe_unused]] Index safe_start_bucket;
    [[maybe_unused]] Index safe_end_bucket;
    if constexpr (parallel) {
        safe_start_bucket = bs->GetNextSafeStart(start_bucket);
        safe_end_bucket = bs->GetPrevSafeEnd(end_bucket);
    }
    Index num_starts = bs->GetNumStarts();

    const auto do_bump = [&](auto &vec) {
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

    Index last_bucket = start_bucket;
    bool all_good = true;
    Index thresh = Hasher::NoBumpThresh();
    // Bump cache (row, input item) pairs that may have to be bumped retroactively
    Index last_cval = -1;
    std::vector<std::pair<Index, Index>> bump_cache;
    // For 1-bit thresholds, we also need an uncompressed bump cache for undoing
    // all insertions with the same uncompressed value if we end up in the
    // "plus" case with a separately stored threshold
    [[maybe_unused]] Index last_val = -1;
    [[maybe_unused]] std::conditional_t<oneBitThresh, decltype(bump_cache), int> unc_bump_cache;
    [[maybe_unused]] bool start_locked = false;
    [[maybe_unused]] bool end_locked = false;
    [[maybe_unused]] Index bump_start;
    if constexpr (parallel) {
        if (thread_index > 0) {
            border_mutexes[thread_index - 1].lock();
            start_locked = true;
        }
        bump_start = thread_index == 0
                     ? start_bucket * BandingStorage::kBucketSize
                     : start_bucket * BandingStorage::kBucketSize + BandingStorage::kCoeffBits - 1;
    }

    for (Index i = start_index; i < end_index; ++i) {
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
            if constexpr (parallel) {
                if (start_locked && bucket >= safe_start_bucket) {
                    border_mutexes[thread_index - 1].unlock();
                    start_locked = false;
                }
                if (thread_index < num_threads - 1 && !end_locked && bucket > safe_end_bucket) {
                    border_mutexes[thread_index].lock();
                    end_locked = true;
                }
            }
            if constexpr (oneBitThresh) {
                unc_bump_cache.clear();
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
            bump_cache.clear();
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
            bump_cache.clear();
            last_cval = cval;
        }
        if constexpr (oneBitThresh) {
            // split into constexpr and normal if because unc_bump_cache isn't a
            // vector if !oneBitThresh
            if (val != last_val) {
                unc_bump_cache.clear();
                last_val = val;
            }
        }


        const CoeffRow cr = hasher.GetCoeffs(hash);
        const ResultRow rr = Hasher::kIsFilter
                                 ? hasher.GetResultRowFromHash(hash)
                                 : hasher.GetResultRowFromInput(*(begin + i));

        /* The first kCoeffBits-1 rows of the first bucket are bumped in the parallel case */
        bool success;
        Index row;
        if constexpr (parallel) {
            std::tie(success, row) = start < bump_start
                                  ? std::make_pair(false, start)
                                  : BandingAdd<kFCA1>(bs, start, cr, rr);
        } else {
            std::tie(success, row) = BandingAdd<kFCA1>(bs, start, cr, rr);
        }
        if (!success) {
            assert(all_good);
            if constexpr (!parallel) {
                // bumping disabled, abort!
                if (bump_vec == nullptr) {
                    return false;
                }
            }
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
                    if constexpr (parallel) {
                        std::scoped_lock lock(hash_mtx);
                        hasher.Set(bucket, val);
                    } else {
                        hasher.Set(bucket, val);
                    }
                    // Set meta to 0 (some bumpage) but keep thresh at 2 so that
                    // we don't retroactively bump everything when moving to the
                    // next bucket
                    bs->SetMeta(bucket, 0);
                    all_good = false;

                    // bump all items with the same uncompressed value
                    do_bump(unc_bump_cache);
                    sLOG << "Also bumping" << tlx::wrap_unprintable(*(begin + i));
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
    // set final threshold
    if (thresh == Hasher::NoBumpThresh()) {
        bs->SetMeta(last_bucket, thresh);
    }

    if constexpr (parallel) {
        if (end_locked) {
            border_mutexes[thread_index].unlock();
        }
        if (start_locked) {
            border_mutexes[thread_index - 1].unlock();
        }
    }

    return true;
}

template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>, typename F, typename G>
std::tuple<bool, size_t, uint64_t> BandingAddRangeBase(BandingStorage *bs, Hasher &hasher, Iterator begin,
                     Iterator end, BumpStorage *bump_vec, F process_func, G sort_func) {
    using Index = typename BandingStorage::Index;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;

    constexpr bool debug = false;
    constexpr bool log = Hasher::log;

    if (begin == end)
        return std::make_tuple(true, 1, 0);

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
        process_func(bs, hasher, input, begin, end);
    }
    LOGC(log) << "\tInput transformation took "
              << timer.ElapsedNanos(true) / 1e6 << "ms";
    sort_func(input.get(), input.get() + num_items);
    auto sort_time = timer.ElapsedNanos(true);
    LOGC(log) << "\tSorting took " << sort_time / 1e6 << "ms";

    bool success = AddRangeInternal<false>(
        bs, hasher, 0, num_buckets, 0, num_items, input,
        begin, bump_vec, 0, 0, 0, 0
    );
    if (!success)
        return std::make_tuple(false, 1, sort_time);

    // migrate thresholds to hash table
    if constexpr (oneBitThresh) {
        hasher.Finalise(num_buckets);
    }

    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return std::make_tuple(true, 1, sort_time);
}

template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
std::tuple<bool, size_t, uint64_t> BandingAddRange(BandingStorage *bs, Hasher &hasher, Iterator begin,
                     Iterator end, BumpStorage *bump_vec) {
    return BandingAddRangeBase(bs, hasher, begin, end, bump_vec,
        [&](BandingStorage *bs, Hasher &hasher, auto &input, Iterator begin, Iterator end) {
        const auto num_items = end - begin;
        ProcessInput(bs, hasher, input, begin, 0, num_items);
        }, [](auto begin, auto end){my_sort(begin, end);});
}

#ifdef _REENTRANT
template <typename BandingStorage, typename Hasher, typename Iterator, typename Input,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
inline void AddRangeParallelInternal(
    BandingStorage *bs, Hasher &hasher, Input input, Iterator begin, Iterator end,
    BumpStorage *bump_vec, std::size_t num_threads) {

    using Index = typename BandingStorage::Index;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;
    [[maybe_unused]] constexpr Index kBucketSize = BandingStorage::kBucketSize;
    constexpr Index kBucketSearchRange = BandingStorage::kBucketSearchRange;
    constexpr BucketSearchMode kBucketSearchMode = BandingStorage::kBucketSearchMode;

    const Index num_buckets = bs->GetNumBuckets();
    [[maybe_unused]] const Index num_starts = bs->GetNumStarts();

    const auto num_items = end - begin;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    /* FIXME: decide if there's a better reserve size here */
    const std::size_t bump_reserve = bump_vec->capacity() / num_threads;
    /* FIXME: make hasher.Set concurrent to remove this mutex */
    [[maybe_unused]] std::conditional_t<oneBitThresh, std::mutex, int> hash_mtx;
    std::vector<std::mutex> border_mutexes(num_threads - 1);
    [[maybe_unused]] std::conditional_t<kBucketSearchRange <= 0, int, std::vector<Index>> thread_ends(num_threads - 1);
    if constexpr (kBucketSearchRange > 0) {
        bs->SetNumThreadBorders(num_threads);
    }
    /* used for synchronizing the search for the bucket in which to bump between threads */
    [[maybe_unused]] std::conditional_t<kBucketSearchRange <= 0, int, std::vector<std::mutex>> search_mtx(num_threads - 1);
    [[maybe_unused]] std::conditional_t<kBucketSearchRange <= 0, int, std::vector<std::condition_variable>> search_cv(num_threads - 1);
    [[maybe_unused]] std::conditional_t<kBucketSearchRange <= 0, int, std::vector<char>> search_status(num_threads - 1);
    /* used for synchronizing the copying of the local bump vectors into the global bump vector */
    std::mutex vec_mtx;
    std::condition_variable vec_cv;
    std::size_t vec_rem = num_threads;
    std::vector<std::size_t> bump_vec_start_pos(num_threads + 1);

    const auto resize_bump_vec = [&]() {
        /* I guess this is probably always 0 anyways */
        bump_vec_start_pos[0] = bump_vec->size();
        for (std::size_t i = 0; i < num_threads; ++i) {
            bump_vec_start_pos[i + 1] += bump_vec_start_pos[i];
        }
        bump_vec->resize(bump_vec_start_pos[num_threads]);
    };

    /* If we return from a thread early because the range starts after the
       last bucket or item, we still need to set some things to avoid
       issues with other threads.

       FIXME: It might be a better idea to guard the code below against this
       condition instead of having all this duplicated code that has to
       be run before returning. */
    const auto return_safe = [&](std::size_t ti) {
        if constexpr (kBucketSearchRange > 0) {
            if (ti > 0) {
                bs->SetThreadBorderBucket(ti - 1, num_buckets);
                thread_ends[ti - 1] = num_items;
                std::unique_lock l(search_mtx[ti - 1]);
                search_status[ti - 1] = 1;
                l.unlock();
                search_cv[ti - 1].notify_all();
            }
        }
        std::unique_lock l(vec_mtx);
        --vec_rem;
        if (vec_rem == 0) {
            resize_bump_vec();
            l.unlock();
            vec_cv.notify_all();
        }
    };

    for (std::size_t ti = 0; ti < num_threads; ++ti) {
        threads.emplace_back([&, ti]() {
            BumpStorage local_bump_vec;
            local_bump_vec.reserve(bump_reserve);

            /* NOTE: This is assumed to never be 0. That could only happen if
               kMinBucketsPerThread is also 0, which is prevented by a static_assert */
            const Index local_num_buckets =
                num_buckets / num_threads +
                (ti < num_buckets % num_threads);
            Index start_bucket =
                ti * (num_buckets / num_threads) +
                (ti < num_buckets % num_threads ? ti : num_buckets % num_threads);

            Index end_bucket = start_bucket + local_num_buckets - 1;
            Index start_index = 0, end_index = 0;
            if constexpr (Hasher::kUseMHC) {
                auto start_it = std::lower_bound(
                    begin, end, start_bucket, [&hasher, num_starts](const auto &e, auto v) {
                    const auto hash = hasher.GetHash(e);
                    const Index start = hasher.GetStart(hash, num_starts);
                    return Hasher::GetBucket(start) < v;
                });
                auto end_it = std::upper_bound(
                    begin, end, end_bucket, [&hasher, num_starts](auto v, const auto &e) {
                    const auto hash = hasher.GetHash(e);
                    const Index start = hasher.GetStart(hash, num_starts);
                    return v < Hasher::GetBucket(start);
                });
                if (start_it == end) {
                    return_safe(ti);
                    return;
                }
                start_index = start_it - begin;
                end_index = end_it - begin;
            } else {
                auto start_it = std::lower_bound(
                    input.get(), input.get() + num_items, start_bucket, [](const auto &e, auto v) {
                    return Hasher::GetBucket(std::get<0>(e)) < v;
                });
                auto end_it = std::upper_bound(
                    input.get(), input.get() + num_items, end_bucket, [](auto v, const auto &e) {
                    return v < Hasher::GetBucket(std::get<0>(e));
                });
                if (start_it == input.get() + num_items) {
                    return_safe(ti);
                    return;
                }
                start_index = start_it - input.get();
                end_index = end_it - input.get();
            }

            /* NOTE; It is important that we do not return immediately even if the item range is
               empty because it's theoretically possible that this is the case but through the
               bucket search, the end index is moved, causing the range to become non-empty.
               This case will probably never occur in practice, especially when kMinBucketsPerThread
               is set to a sensible value, but it doesn't hurt to be careful. Of course, this hasn't
               been tested properly, so it might blow up anyways when that happens. */

            if constexpr (kBucketSearchRange > 0) {
                if (ti > 0) {
                    Index cur_bucket = start_bucket;
                    Index min_bucket = start_bucket;
                    int min_elems = std::numeric_limits<int>::max();
                    Index min_bucket_start = start_index;
                    int cur_elems = 0;
                    [[maybe_unused]] int old_next_elems = 0;
                    [[maybe_unused]] int next_elems = 0;
                    Index cur_bucket_start = start_index;
                    [[maybe_unused]] const Index count_val = BandingStorage::kCoeffBits - 1;
                    const Index bump_val = kBucketSize - BandingStorage::kCoeffBits + 1;
                    const Index bump_cval = hasher.Compress(bump_val);
                    if constexpr (kBucketSearchMode == BucketSearchMode::diff || kBucketSearchMode == BucketSearchMode::maxprev) {
                        for (Index i = start_index; i-- > 0;) {
                            Index sortpos;
                            if constexpr (Hasher::kUseMHC) {
                                const auto hash = hasher.GetHash(*(begin + i));
                                sortpos = Hasher::SortToStart(hasher.GetStart(hash, num_starts));
                            } else {
                                sortpos = std::get<0>(input[i]);
                            }
                            const Index bucket = Hasher::GetBucket(sortpos);
                            const Index val = Hasher::GetIntraBucket(sortpos);
                            if (bucket < start_bucket - 1)
                                break;
                            else if (val < count_val)
                                ++old_next_elems;
                        }
                    }
                    for (Index i = start_index; i < end_index; ++i) {
                        Index sortpos;
                        if constexpr (Hasher::kUseMHC) {
                            const auto hash = hasher.GetHash(*(begin + i));
                            sortpos = Hasher::SortToStart(hasher.GetStart(hash, num_starts));
                        } else {
                            sortpos = std::get<0>(input[i]);
                        }
                        const Index bucket = Hasher::GetBucket(sortpos);
                        const Index val = Hasher::GetIntraBucket(sortpos);
                        const Index cval = hasher.Compress(val);
                        if (bucket != cur_bucket) {
                            int diff;
                            if constexpr (kBucketSearchMode == BucketSearchMode::diff) {
                                diff = cur_elems - old_next_elems;
                            } else if constexpr (kBucketSearchMode == BucketSearchMode::maxprev) {
                                diff = -old_next_elems;
                            } else {
                                diff = cur_elems;
                            }
                            if (diff < min_elems) {
                                min_elems = diff;
                                min_bucket = cur_bucket;
                                min_bucket_start = cur_bucket_start;
                            }
                            if (bucket >= start_bucket + kBucketSearchRange)
                                break;
                            cur_elems = 0;
                            cur_bucket_start = i;
                            cur_bucket = bucket;
                            if constexpr (kBucketSearchMode == BucketSearchMode::diff || kBucketSearchMode == BucketSearchMode::maxprev) {
                                old_next_elems = next_elems;
                                next_elems = 0;
                            }
                        }
                        if constexpr (oneBitThresh) {
                            if (bump_cval == 2) {
                                if (val >= bump_val)
                                    ++cur_elems;
                            } else if (cval >= bump_cval) {
                                ++cur_elems;
                            }
                        } else {
                            if (cval >= bump_cval)
                                ++cur_elems;
                        }
                        if constexpr (kBucketSearchMode == BucketSearchMode::diff || kBucketSearchMode == BucketSearchMode::maxprev) {
                            if (val < count_val)
                                ++next_elems;
                        }
                    }
                    int diff;
                    if constexpr (kBucketSearchMode == BucketSearchMode::diff) {
                        diff = cur_elems - old_next_elems;
                    } else if (kBucketSearchMode == BucketSearchMode::maxprev) {
                        diff = -old_next_elems;
                    } else {
                        diff = cur_elems;
                    }
                    /* this should only happen if end_index == start_index or
                       the loop ran all the way to end_index - 1, both of
                       which are edge cases that shouldn't usually happen when
                       the parameters are chosen properly */
                    if (diff < min_elems) {
                        min_bucket = cur_bucket;
                        min_bucket_start = cur_bucket_start;
                    }

                    bs->SetThreadBorderBucket(ti - 1, min_bucket);
                    thread_ends[ti - 1] = min_bucket_start;
                    std::unique_lock l(search_mtx[ti - 1]);
                    search_status[ti - 1] = 1;
                    l.unlock();
                    search_cv[ti - 1].notify_all();
                }
                if (ti < num_threads - 1) {
                    std::unique_lock l(search_mtx[ti]);
                    while (!search_status[ti]) {
                        search_cv[ti].wait(l);
                    }
                }
                if (ti < num_threads - 1) {
                    end_bucket = bs->GetThreadBorderBucket(ti);
                    /* end_bucket == 0 would cause problems,
                       but that shouldn't happen anyways */
                    if (end_bucket > 0)
                        --end_bucket;
                    end_index = thread_ends[ti];
                }
                if (ti > 0) {
                    start_bucket = bs->GetThreadBorderBucket(ti - 1);
                    start_index = thread_ends[ti - 1];
                }
            }

            /* NOTE: It is theoretically possible for the range to be empty,
               although this will probably never happen in practice. In that
               case, the loop in AddRangeInternal[MHC] will not run, so only
               the threshold of start_bucket will be set to NoBumpThresh().
               This will only affect the behavior of negative queries, though. */
            if constexpr (Hasher::kUseMHC) {
                AddRangeInternalMHC<true>(
                    bs, hasher, start_bucket, end_bucket, start_index, end_index,
                    begin, &local_bump_vec, ti, num_threads, border_mutexes, hash_mtx
                );
            } else {
                AddRangeInternal<true>(
                    bs, hasher, start_bucket, end_bucket, start_index, end_index, input,
                    begin, &local_bump_vec, ti, num_threads, border_mutexes, hash_mtx
                );
            }

            bump_vec_start_pos[ti + 1] = local_bump_vec.size();
            {
                std::unique_lock l(vec_mtx);
                --vec_rem;
                if (vec_rem > 0) {
                    do {
                        vec_cv.wait(l);
                    } while (vec_rem > 0);
                } else {
                    resize_bump_vec();
                    l.unlock();
                    vec_cv.notify_all();
                }
            }
            std::copy(
                local_bump_vec.begin(),
                local_bump_vec.end(),
                bump_vec->begin() + bump_vec_start_pos[ti]
            );
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    // migrate thresholds to hash table
    if constexpr (oneBitThresh) {
        hasher.Finalise(num_buckets);
    }
}
#endif

#ifdef _REENTRANT
template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
std::tuple<bool, size_t, uint64_t> BandingAddRangeParallel(BandingStorage *bs, Hasher &hasher, Iterator begin,
                     Iterator end, BumpStorage *bump_vec, std::size_t num_threads) {
    using Index = typename BandingStorage::Index;
    constexpr Index kMinBucketsPerThread = BandingStorage::kMinBucketsPerThread;
    constexpr Index kBucketSize = BandingStorage::kBucketSize;
    constexpr Index kCoeffBits = BandingStorage::kCoeffBits;
    constexpr Index kBucketSearchRange = BandingStorage::kBucketSearchRange;

    [[maybe_unused]] constexpr Index bump_buckets =
        kCoeffBits <= 1 ? 1 : (kCoeffBits - 1 + kBucketSize - 1) / kBucketSize;
    /* less than this wouldn't really make sense */
    static_assert(kMinBucketsPerThread >= 1 + bump_buckets);
    assert(num_threads > 0);
    if constexpr (kBucketSearchRange > 0) {
        /* FIXME: this is not a static_assert because the template is
           currently initialized with invalid arguments in some cases */
        if (bump_buckets > 1) {
            std::cerr << "kBucketSize < kCoeffBits - 1 is currently not supported with search range!\n";
            abort();
        }
    }

    constexpr bool debug = false;
    constexpr bool log = Hasher::log;

    if (begin == end)
        return std::make_tuple(true, num_threads, 0);

    /* These two lambdas are used to still perform the preprocessing and sorting in
       parallel even when the sequential insertion is used (i.e. when we are in the
       base case ribbon or there are too few slots) */
    /* this uses the original number of threads before it is possibly reduced */
    const auto process_input = [num_threads](BandingStorage *bs, Hasher &hasher, auto &input, Iterator begin, Iterator end) {
        const auto num_items = end - begin;
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (std::size_t ti = 0; ti < num_threads; ++ti) {
            threads.emplace_back([&, ti]() {
                const Index local_num_items =
                    num_items / num_threads +
                    (ti < num_items % num_threads);
                Index start_idx =
                    ti * (num_items / num_threads) +
                    (ti < num_items % num_threads ? ti : num_items % num_threads);
                Index end_idx = start_idx + local_num_items;
                ProcessInput(bs, hasher, input, begin, start_idx, end_idx);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    };
    const auto sort_parallel = [num_threads](auto begin, auto end) {
        my_sort(begin, end, num_threads);
    };

    if (!bump_vec) {
        return BandingAddRangeBase(bs, hasher, begin, end, bump_vec, process_input, sort_parallel);
    }

    rocksdb::StopWatchNano timer(true);
    const Index num_buckets = bs->GetNumBuckets();
    const Index num_starts = bs->GetNumStarts();

    const Index buckets_per_thread = num_buckets / num_threads;
    if (buckets_per_thread < kMinBucketsPerThread) {
        num_threads = num_buckets / kMinBucketsPerThread;
        LOGC(log) << "Reducing to " << num_threads << " threads.";
        if (num_threads <= 1)
            return BandingAddRangeBase(bs, hasher, begin, end, bump_vec, process_input, sort_parallel);
    }

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
        process_input(bs, hasher, input, begin, end);
    }
    LOGC(log) << "\tInput transformation took "
              << timer.ElapsedNanos(true) / 1e6 << "ms";
    sort_parallel(input.get(), input.get() + num_items);
    auto sort_time = timer.ElapsedNanos(true);
    LOGC(log) << "\tSorting took " << sort_time / 1e6 << "ms";
    AddRangeParallelInternal(bs, hasher, std::move(input), begin, end, bump_vec, num_threads);
    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return std::make_tuple(true, num_threads, sort_time);
}
#endif

template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename Hasher::mhc_t>, typename F>
std::tuple<bool, size_t, uint64_t> BandingAddRangeBaseMHC(BandingStorage *bs, Hasher &hasher, Iterator begin,
                        Iterator end, BumpStorage *bump_vec, F sort_func) {
    static_assert(Hasher::kUseMHC, "you called the wrong method");

    using Index = typename BandingStorage::Index;
    constexpr bool oneBitThresh = (Hasher::kThreshMode == ThreshMode::onebit);

    constexpr bool debug = false;
    constexpr bool log = Hasher::log;

    if (begin == end)
        return std::make_tuple(true, 1, 0);

    rocksdb::StopWatchNano timer(true);
    const auto num_items = end - begin;
    const Index num_starts = bs->GetNumStarts();
    const Index num_buckets = bs->GetNumBuckets();
    sLOG << "Constructing ribbon (MHC) with" << num_buckets
         << "buckets, num_starts =" << num_starts;

    sort_func(begin, end, hasher, num_starts);
    // MHCs should be unique, if not, fail construction
    if constexpr (Hasher::kIsFilter) {
        assert(std::adjacent_find(begin, end) == end);
    } else {
        assert(std::adjacent_find(begin, end, [](const auto &a, const auto &b) {
                   return a.first == b.first;
               }) == end);
    }

    auto sort_time = timer.ElapsedNanos(true);

    LOGC(log) << "\tSorting took " << sort_time / 1e6 << "ms";

    bool success = AddRangeInternalMHC<false>(
        bs, hasher, 0, num_buckets, 0, num_items,
        begin, bump_vec, 0, 0, 0, 0
    );
    if (!success)
        return std::make_tuple(false, 1, sort_time);

    // migrate thresholds to hash table
    if constexpr (oneBitThresh) {
        hasher.Finalise(num_buckets);
    }

    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return std::make_tuple(true, 1, sort_time);
}

template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
std::tuple<bool, size_t, uint64_t> BandingAddRangeMHC(BandingStorage *bs, Hasher &hasher, Iterator begin,
                     Iterator end, BumpStorage *bump_vec) {
    static_assert(Hasher::kUseMHC, "you called the wrong method");
    return BandingAddRangeBaseMHC(
        bs, hasher, begin, end, bump_vec,
        [](auto begin, auto end, auto &hasher, auto num_starts){my_sort(begin, end, hasher, num_starts);}
    );
}

#ifdef _REENTRANT
template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
std::tuple<bool, size_t, uint64_t> BandingAddRangeParallelMHC(BandingStorage *bs, Hasher &hasher, Iterator begin,
                     Iterator end, BumpStorage *bump_vec, std::size_t num_threads) {
    static_assert(Hasher::kUseMHC, "you called the wrong method");
    using Index = typename BandingStorage::Index;
    constexpr Index kMinBucketsPerThread = BandingStorage::kMinBucketsPerThread;
    constexpr Index kBucketSize = BandingStorage::kBucketSize;
    constexpr Index kCoeffBits = BandingStorage::kCoeffBits;
    constexpr Index kBucketSearchRange = BandingStorage::kBucketSearchRange;

    [[maybe_unused]] constexpr Index bump_buckets =
        kCoeffBits <= 1 ? 1 : (kCoeffBits - 1 + kBucketSize - 1) / kBucketSize;
    /* less than this wouldn't really make sense */
    static_assert(kMinBucketsPerThread >= 1 + bump_buckets);
    assert(num_threads > 0);
    if constexpr (kBucketSearchRange > 0) {
        /* FIXME: this is not a static_assert because the template is
           currently initialized with invalid arguments in some cases */
        if (bump_buckets > 1) {
            std::cerr << "kBucketSize < kCoeffBits - 1 is currently not supported with search range!\n";
            abort();
        }
    }

    constexpr bool debug = false;
    constexpr bool log = Hasher::log;

    if (begin == end)
        return std::make_tuple(true, num_threads, 0);

    const auto sort_parallel = [num_threads](auto begin, auto end, auto &hasher, auto num_starts) {
        my_sort(begin, end, hasher, num_starts, num_threads);
    };

    if (!bump_vec) {
        return BandingAddRangeBaseMHC(bs, hasher, begin, end, bump_vec, sort_parallel);
    }

    const Index num_buckets = bs->GetNumBuckets();
    const Index num_starts = bs->GetNumStarts();

    const Index buckets_per_thread = num_buckets / num_threads;
    if (buckets_per_thread < kMinBucketsPerThread) {
        num_threads = num_buckets / kMinBucketsPerThread;
        LOGC(log) << "Reducing to " << num_threads << " threads.";
        if (num_threads <= 1)
            return BandingAddRangeBaseMHC(bs, hasher, begin, end, bump_vec, sort_parallel);
    }

    sLOG << "Constructing ribbon (MHC) with" << num_buckets
         << "buckets, num_starts =" << num_starts;

    rocksdb::StopWatchNano timer(true);

    sort_parallel(begin, end, hasher, num_starts);
    // MHCs should be unique, if not, fail construction
    if constexpr (Hasher::kIsFilter) {
        assert(std::adjacent_find(begin, end) == end);
    } else {
        assert(std::adjacent_find(begin, end, [](const auto &a, const auto &b) {
                   return a.first == b.first;
               }) == end);
    }
    auto sort_time = timer.ElapsedNanos(true);
    LOGC(log) << "\tSorting took " << sort_time / 1e6 << "ms";
    /* 0 is just a dummy value where the non-MHC version gets the input */
    AddRangeParallelInternal(bs, hasher, 0, begin, end, bump_vec, num_threads);
    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return std::make_tuple(true, num_threads, sort_time);;
}
#endif

} // namespace ribbon
