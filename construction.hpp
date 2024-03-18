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
#include <mutex>
#include <thread>

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

template <typename Iterator>
__attribute__((noinline)) void my_sort_parallel(Iterator begin, Iterator end, std::size_t num_threads) {
#ifdef RIBBON_USE_STD_SORT
    // Use std::sort as a slow fallback
    std::sort(std::execution::par_unseq, begin, end, [](const auto &a, const auto &b) {
        return std::get<0>(a) < std::get<0>(b);
    });
#else
    ips2ra::parallel::sort(begin, end, [](const auto &x) { return std::get<0>(x); }, num_threads);
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
                  std::conditional_t<Hasher::kIsFilter, SorterDummyData,
                                     typename Hasher::ResultRow>>()
        .do_sort(begin, end, mh, num_starts);
}


template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename std::iterator_traits<Iterator>::value_type>>
bool BandingAddRange(BandingStorage *bs, Hasher &hasher, Iterator begin,
                     Iterator end, BumpStorage *bump_vec) {
    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using ResultRow = typename BandingStorage::ResultRow;
    using Hash = typename Hasher::Hash;
    constexpr bool kFCA1 = Hasher::kFirstCoeffAlwaysOne;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;

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
    std::vector<std::pair<Index, Index>> bump_cache;
    // For 1-bit thresholds, we also need an uncompressed bump cache for undoing
    // all insertions with the same uncompressed value if we end up in the
    // "plus" case with a separately stored threshold
    [[maybe_unused]] Index last_val = -1;
    [[maybe_unused]] std::conditional_t<oneBitThresh, decltype(bump_cache), int> unc_bump_cache;

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
bool BandingAddRangeParallel(BandingStorage *bs, Hasher &hasher, Iterator begin,
                     Iterator end, BumpStorage *bump_vec, std::size_t num_threads) {
    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using ResultRow = typename BandingStorage::ResultRow;
    using Hash = typename Hasher::Hash;
    constexpr bool kFCA1 = Hasher::kFirstCoeffAlwaysOne;
    constexpr bool oneBitThresh = Hasher::kThreshMode == ThreshMode::onebit;
    constexpr Index kMinBucketsPerThread = BandingStorage::kMinBucketsPerThread;
    constexpr Index kBucketSize = BandingStorage::kBucketSize;
    constexpr Index kCoeffBits = BandingStorage::kCoeffBits;
    constexpr Index kBucketSearchRange = BandingStorage::kBucketSearchRange;
    constexpr BucketSearchMode kBucketSearchMode = BandingStorage::kBucketSearchMode;

    [[maybe_unused]] constexpr Index bump_buckets =
        kCoeffBits <= 1 ? 1 : (kCoeffBits - 1 + kBucketSize - 1) / kBucketSize;
    /* less than this wouldn't really make sense */
    static_assert(kMinBucketsPerThread >= 1 + bump_buckets);

    constexpr bool debug = false;
    constexpr bool log = Hasher::log;

    if (begin == end)
        return true;

    if (!bump_vec)
        return BandingAddRange(bs, hasher, begin, end, bump_vec);

    rocksdb::StopWatchNano timer(true);
    const Index num_starts = bs->GetNumStarts();
    const Index num_buckets = bs->GetNumBuckets();

    std::size_t orig_num_threads = num_threads;
    /* NOTE: Here, we round down in order to make sure that all threads
       get at least kMinBucketsPerThread buckets. Below, we round up
       for the actual distribution. */
    /* FIXME: should num_threads be a power of 2? */
    std::size_t buckets_per_thread = num_buckets / num_threads;
    if (buckets_per_thread < kMinBucketsPerThread) {
        num_threads = num_buckets / kMinBucketsPerThread;
        assert(num_threads < orig_num_threads);
        LOGC(log) << "reducing to " << num_threads << " threads.\n";
        if (num_threads <= 1)
            return BandingAddRange(bs, hasher, begin, end, bump_vec);
    }
    buckets_per_thread = (num_buckets + num_threads - 1) / num_threads;

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

        std::size_t items_per_thread = (num_items + num_threads - 1) / num_threads;
        std::vector<std::thread> threads;
        threads.reserve(orig_num_threads);
        for (std::size_t ti = 0; ti < orig_num_threads; ++ti) {
            threads.emplace_back([&, ti]() {
                Index start_idx = static_cast<Index>(ti * items_per_thread);
                Index end_idx = static_cast<Index>(std::min((ti + 1) * items_per_thread, static_cast<std::size_t>(num_items)));
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
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    LOGC(log) << "\tInput transformation took "
              << timer.ElapsedNanos(true) / 1e6 << "ms";
    my_sort_parallel(input.get(), input.get() + num_items, orig_num_threads);
    LOGC(log) << "\tSorting took " << timer.ElapsedNanos(true) / 1e6 << "ms";

    const auto do_bump = [&](auto &thread_bump_vec, auto &vec) {
        sLOG << "Bumping" << vec.size() << "items";
        for (auto [row, idx] : vec) {
            sLOG << "\tBumping row" << row << "item"
                 << tlx::wrap_unprintable(*(begin + idx));
            bs->SetCoeffs(row, 0);
            bs->SetResult(row, 0);
            thread_bump_vec.push_back(*(begin + idx));
        }
        vec.clear();
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    /* FIXME: decide if there's a better reserve size here */
    const std::size_t bump_reserve = bump_vec->capacity() / num_threads;
    /* FIXME: make hasher.Set concurrent to remove this mutex */
    [[maybe_unused]] std::conditional_t<oneBitThresh, std::mutex, int> hash_mtx;
    std::vector<std::mutex> border_mutexes(num_threads - 1);
    [[maybe_unused]] std::conditional_t<kBucketSearchRange <= 0, int, std::vector<Index>> thread_ends(num_threads - 1);
    [[maybe_unused]] std::conditional_t<kBucketSearchRange <= 0, int, std::vector<Index>> thread_starts(num_threads - 1);
    if constexpr (kBucketSearchRange > 0) {
        bs->SetNumThreads(num_threads);
    }
    /* used for synchronizing the search for the bucket in which to bump between threads */
    [[maybe_unused]] std::conditional_t<kBucketSearchRange <= 0, int, std::mutex> search_mtx;
    [[maybe_unused]] std::conditional_t<kBucketSearchRange <= 0, int, std::condition_variable> search_cv;
    [[maybe_unused]] std::size_t search_rem = num_threads;
    /* used for synchronizing the copying of the local bump vectors into the global bump vector */
    std::mutex vec_mtx;
    std::condition_variable vec_cv;
    std::size_t vec_rem = num_threads;
    std::vector<std::size_t> bump_vec_start_pos(num_threads + 1);
    for (std::size_t ti = 0; ti < num_threads; ++ti) {
        threads.emplace_back([&, ti]() {
            BumpStorage local_bump_vec;
            local_bump_vec.reserve(bump_reserve);
            Index start_bucket = ti * buckets_per_thread;
            if (start_bucket >= num_buckets)
                return;
            Index end_bucket = start_bucket + buckets_per_thread - 1;
            if (end_bucket >= num_buckets)
                end_bucket = num_buckets - 1;
            auto start_it = std::lower_bound(
                input.get(), input.get() + num_items, start_bucket, [](const auto& e, auto v) {
                return Hasher::GetBucket(std::get<0>(e)) < v;
            });
            auto end_it = std::upper_bound(
                input.get(), input.get() + num_items, end_bucket, [](auto v, const auto& e) {
                return v < Hasher::GetBucket(std::get<0>(e));
            });
            if (start_it == input.get() + num_items) {
                return;
            }
            Index start_index = start_it - input.get();
            Index end_index = end_it - input.get();

            if constexpr (kBucketSearchRange > 0) {
                if constexpr (bump_buckets == 1) {
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
                                Index sortpos = std::get<0>(input[i]);
                                Index bucket = Hasher::GetBucket(sortpos);
                                Index val = Hasher::GetIntraBucket(sortpos);
                                if (bucket < start_bucket - 1)
                                    break;
                                else if (val < count_val)
                                    ++old_next_elems;
                            }
                        }
                        for (Index i = start_index; i < end_index; ++i) {
                            Index sortpos = std::get<0>(input[i]);
                            Index bucket = Hasher::GetBucket(sortpos);
                            Index val = Hasher::GetIntraBucket(sortpos);
                            Index cval = hasher.Compress(val);
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
                           which are edge cases that should never happen when the
                           parameters are chosen properly */
                        if (diff < min_elems) {
                            min_bucket = cur_bucket;
                            min_bucket_start = cur_bucket_start;
                        }
                        bs->SetThreadBorderBucket(ti - 1, min_bucket);
                        thread_starts[ti - 1] = min_bucket_start;
                        thread_ends[ti - 1] = min_bucket_start;
                    }
                } else {
                    std::cerr << "kBucketSize < kCoeffBits - 1 is currently not supported with search range!\n";
                    abort();
                }
                {
                    std::unique_lock l(search_mtx);
                    --search_rem;
                    if (search_rem > 0)
                        search_cv.wait(l);
                    else
                        search_cv.notify_all();
                }
                if (ti < num_threads - 1) {
                    end_bucket = bs->GetThreadBorderBucket(ti);
                    /* FIXME: end_bucket == 0 would cause problems,
                       but that would be really weird anyways */
                    if (end_bucket > 0)
                        --end_bucket;
                    end_index = thread_ends[ti];
                }
                if (ti > 0) {
                    start_bucket = bs->GetThreadBorderBucket(ti - 1);
                    start_index = thread_starts[ti - 1];
                }
            }

            Index safe_start_bucket = bs->GetNextSafeStart(start_bucket);
            Index safe_end_bucket = bs->GetPrevSafeEnd(end_bucket);

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
            bool start_locked = false;
            bool end_locked = false;
            if (ti > 0) {
                border_mutexes[ti - 1].lock();
                start_locked = true;
            }
            [[maybe_unused]] Index bump_start = ti == 0
                                   ? start_bucket * BandingStorage::kBucketSize
                                   : start_bucket * BandingStorage::kBucketSize + BandingStorage::kCoeffBits - 1;

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
                    if (ti > 0 && last_bucket < safe_start_bucket && bucket >= safe_start_bucket) {
                        border_mutexes[ti - 1].unlock();
                        start_locked = false;
                    }
                    if (ti < num_threads - 1 && last_bucket <= safe_end_bucket && bucket > safe_end_bucket) {
                        border_mutexes[ti].lock();
                        end_locked = true;
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
                    local_bump_vec.push_back(*(begin + idx));
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

                /* The first kCoeffBits rows of the first bucket are bumped */
                const auto [success, row] = start < bump_start
                                          ? std::make_pair(false, start)
                                          : BandingAdd<kFCA1>(bs, start, cr, rr);

                if (!success) {
                    assert(all_good);
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
                            {
                                std::scoped_lock lock(hash_mtx);
                                hasher.Set(bucket, val);
                            }
                            // Set meta to 0 (some bumpage) but keep thresh at 2 so that
                            // we don't retroactively bump everything when moving to the
                            // next bucket
                            bs->SetMeta(bucket, 0);
                            all_good = false;

                            // bump all items with the same uncompressed value
                            do_bump(local_bump_vec, unc_bump_cache);
                            sLOG << "Also bumping"
                                 << tlx::wrap_unprintable(*(begin + idx));
                            local_bump_vec.push_back(*(begin + idx));
                            // proceed to next item, don't do regular bumping (but only
                            // if cval == 2, not generally!)
                            continue;
                        }
                    }
                    bs->SetMeta(bucket, thresh);
                    all_good = false;

                    do_bump(local_bump_vec, bump_cache);
                    local_bump_vec.push_back(*(begin + idx));
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

            if (end_locked) {
                border_mutexes[ti].unlock();
            }
            if (start_locked) {
                border_mutexes[ti - 1].unlock();
            }
            bump_vec_start_pos[ti + 1] = local_bump_vec.size();
            {
                std::unique_lock l(vec_mtx);
                --vec_rem;
                if (vec_rem > 0) {
                    vec_cv.wait(l);
                } else {
                    /* I guess this is probably always 0 anyways */
                    bump_vec_start_pos[0] = bump_vec->size();
                    for (std::size_t i = 0; i < num_threads; ++i) {
                        bump_vec_start_pos[i + 1] += bump_vec_start_pos[i];
                    }
                    bump_vec->resize(bump_vec_start_pos[num_threads]);
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

    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return true;
}


template <typename BandingStorage, typename Hasher, typename Iterator,
          typename BumpStorage = std::vector<typename Hasher::mhc_t>>
bool BandingAddRangeMHC(BandingStorage *bs, Hasher &hasher, Iterator begin,
                        Iterator end, BumpStorage *bump_vec) {
    static_assert(Hasher::kUseMHC, "you called the wrong method");

    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using ResultRow = typename BandingStorage::ResultRow;
    constexpr bool kFCA1 = Hasher::kFirstCoeffAlwaysOne;
    constexpr bool oneBitThresh = (Hasher::kThreshMode == ThreshMode::onebit);

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

    Index last_bucket = 0;
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

    // migrate thresholds to hash table
    if constexpr (oneBitThresh) {
        hasher.Finalise(num_buckets);
    }

    LOGC(log) << "\tActual insertion took " << timer.ElapsedNanos(true) / 1e6
              << "ms";
    return true;
}

} // namespace ribbon
