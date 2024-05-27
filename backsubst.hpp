//  Copyright (c) Lorenz Hübschle-Schneider
//  Copyright (c) Facebook, Inc. and its affiliates.
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "config.hpp"
#include "rocksdb/math.h"

#include <tlx/logger.hpp>

#include <array>
#include <cassert>
#include <memory>

#ifdef RIBBON_DUMP
#include <bitset>
#include <iomanip>
#include <ios>
#endif

#include <iostream>

#ifdef _REENTRANT
#include <mutex>
#include <thread>
#endif

namespace ribbon {

template <typename BandingStorage, typename SolutionStorage>
void SimpleBackSubst(const BandingStorage &bs, SolutionStorage *sol) {
    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    // use uint32_t instead of uint16_t because gcc is bad with uint16_t
    using ResultRow = make_fast_t<typename BandingStorage::ResultRow>;

    constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U);
    constexpr auto kResultBits = static_cast<Index>(sizeof(ResultRow) * 8U);


    // A column-major buffer of the solution matrix, containing enough
    // recently-computed solution data to compute the next solution row
    // (based also on banding data).
    std::array<CoeffRow, kResultBits> state;
    state.fill(0);

    const Index num_starts = bs.GetNumStarts();
    // sss->PrepareForNumStarts(num_starts);
    const Index num_slots = num_starts + kCoeffBits - 1;

    for (Index i = num_slots; i > 0;) {
        --i;
        CoeffRow cr = bs.GetCoeffs(i);
        ResultRow rr = bs.GetResult(i);
        // solution row
        ResultRow sr = 0;
        for (Index j = 0; j < kResultBits; ++j) {
            // Compute next solution bit at row i, column j (see derivation below)
            CoeffRow tmp = state[j] << 1;
            bool bit = (rocksdb::BitParity(tmp & cr) ^ ((rr >> j) & 1)) != 0;
            tmp |= bit ? CoeffRow{1} : CoeffRow{0};

            // Now tmp is solution at column j from row i for next kCoeffBits
            // more rows. Thus, for valid solution, the dot product of the
            // solution column with the coefficient row has to equal the result
            // at that column,
            //   BitParity(tmp & cr) == ((rr >> j) & 1)

            // Update state.
            state[j] = tmp;
            // add to solution row
            sr |= (bit ? ResultRow{1} : ResultRow{0}) << j;
        }
        sol->SetResult(i, sr);
    }

#ifdef RIBBON_DUMP
    sLOG1 << num_slots << "slots";
    for (Index i = 0; i < num_slots; i++) {
        const ResultRow r = sol->GetResult(i);
        LOG1 << "Row " << std::setw(2) << i << " = " << std::hex << std::setw(2)
             << (uint64_t)r << std::dec << " = "
             << std::bitset<sizeof(ResultRow) * 8u>(r).to_string();
    }
#endif
}

#ifdef _REENTRANT
/* NOTE: It is crucial that this is not called by the base case ribbon
   since that does not support bumping */
template <typename BandingStorage, typename SolutionStorage>
void SimpleBackSubstParallel(const BandingStorage &bs, SolutionStorage *sol, std::size_t num_threads) {
    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    // use uint32_t instead of uint16_t because gcc is bad with uint16_t
    using ResultRow = make_fast_t<typename BandingStorage::ResultRow>;

    constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U);
    constexpr auto kResultBits = static_cast<Index>(sizeof(ResultRow) * 8U);
    constexpr Index kMinBucketsPerThread = BandingStorage::kMinBucketsPerThread;
    constexpr Index kBucketSearchRange = BandingStorage::kBucketSearchRange;

    const Index num_starts = bs.GetNumStarts();
    const Index num_buckets = bs.GetNumBuckets();

    /* NOTE: It is crucial that this is calculated the same as in BandingAddParallel
       so there are no conflicts. */
    std::size_t buckets_per_thread = num_buckets / num_threads;
    if (buckets_per_thread < kMinBucketsPerThread) {
        [[maybe_unused]] std::size_t orig_num_threads = num_threads;
        num_threads = num_buckets / kMinBucketsPerThread;
        assert(num_threads < orig_num_threads);
        if (num_threads <= 1) {
            SimpleBackSubst(bs, sol);
            return;
        }
    }

    // sss->PrepareForNumStarts(num_starts);
    const Index num_slots = num_starts + kCoeffBits - 1;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    [[maybe_unused]] std::conditional_t<BandingStorage::kUseCacheLineStorage, std::vector<std::mutex>, std::size_t> mutexes(num_threads - 1);
    std::mutex output_mutex;
    for (std::size_t ti = 0; ti < num_threads; ++ti) {
        threads.emplace_back([&, ti]() {
            // A column-major buffer of the solution matrix, containing enough
            // recently-computed solution data to compute the next solution row
            // (based also on banding data).
            std::array<CoeffRow, kResultBits> state;
            state.fill(0);

            const Index local_num_buckets =
                num_buckets / num_threads +
                (ti < num_buckets % num_threads);
            Index start_bucket =
                ti * (num_buckets / num_threads) +
                (ti < num_buckets % num_threads ? ti : num_buckets % num_threads);
            Index end_bucket = start_bucket + local_num_buckets;
            if constexpr (kBucketSearchRange > 0) {
                if (ti < num_threads - 1) {
                    end_bucket = bs.GetThreadBorderBucket(ti);
                }
                if (ti > 0) {
                    start_bucket = bs.GetThreadBorderBucket(ti - 1);
                }
            }
            Index start_slot, end_slot;
            start_slot = ti == 0
                                   ? start_bucket * BandingStorage::kBucketSize
                                   : start_bucket * BandingStorage::kBucketSize + kCoeffBits - 1;
            end_slot = end_bucket >= num_buckets
                                 ? num_slots
                                 : end_bucket * BandingStorage::kBucketSize + kCoeffBits - 1;
            [[maybe_unused]] Index safe_end = 0;
            [[maybe_unused]] Index safe_start = 0;
            if constexpr (BandingStorage::kUseCacheLineStorage) {
                safe_end = sol->GetPrevSafeRowEnd(end_slot - 1);
                safe_start = sol->GetNextSafeRowStart(start_slot);
            }
            [[maybe_unused]] bool start_locked = false;
            [[maybe_unused]] bool end_locked = false;
            if constexpr (BandingStorage::kUseCacheLineStorage) {
                if (ti < num_threads - 1) {
                    mutexes[ti].lock();
                    end_locked = true;
                }
            }
            /* FIXME: maybe only perform this check inside an assert() so it
               is disabled with NDEBUG? */
            /* if this isn't the last thread, check if the necessary amount
               of empty rows is present for parallel processing to work
               (if the back substitution is called with the same amount of
               threads as the insertion, this should always be the case) */
            if (end_bucket < num_buckets) {
                for (Index i = 1; i < kCoeffBits; ++i) {
                    CoeffRow cr = bs.GetCoeffs(end_slot - i);
                    if ((cr >> i) != 0) {
                        std::scoped_lock lk(output_mutex);
                        std::cerr << "ERROR: Thread " << ti << " found non-zero " <<
                                     "coefficient row in back substitution!\n";
                        /* FIXME: better error handling? */
                        abort();
                    }
                }
            }
            for (Index i = end_slot; i > start_slot;) {
                --i;
                CoeffRow cr = bs.GetCoeffs(i);
                ResultRow rr = bs.GetResult(i);
                // solution row
                ResultRow sr = 0;
                for (Index j = 0; j < kResultBits; ++j) {
                    // Compute next solution bit at row i, column j (see derivation below)
                    CoeffRow tmp = state[j] << 1;
                    bool bit = (rocksdb::BitParity(tmp & cr) ^ ((rr >> j) & 1)) != 0;
                    tmp |= bit ? CoeffRow{1} : CoeffRow{0};

                    // Now tmp is solution at column j from row i for next kCoeffBits
                    // more rows. Thus, for valid solution, the dot product of the
                    // solution column with the coefficient row has to equal the result
                    // at that column,
                    //   BitParity(tmp & cr) == ((rr >> j) & 1)

                    // Update state.
                    state[j] = tmp;
                    // add to solution row
                    sr |= (bit ? ResultRow{1} : ResultRow{0}) << j;
                }
                if constexpr (BandingStorage::kUseCacheLineStorage) {
                    if (ti > 0 && !start_locked && i < safe_start) {
                        mutexes[ti - 1].lock();
                        start_locked = true;
                    }
                    if (ti < num_threads - 1 && end_locked && i <= safe_end) {
                        mutexes[ti].unlock();
                        end_locked = false;
                    }
                }
                sol->SetResult(i, sr);
            }
            if constexpr (BandingStorage::kUseCacheLineStorage) {
                if (start_locked)
                    mutexes[ti - 1].unlock();
                if (end_locked)
                    mutexes[ti].unlock();
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

#ifdef RIBBON_DUMP
    sLOG1 << num_slots << "slots";
    for (Index i = 0; i < num_slots; i++) {
        const ResultRow r = sol->GetResult(i);
        LOG1 << "Row " << std::setw(2) << i << " = " << std::hex << std::setw(2)
             << (uint64_t)r << std::dec << " = "
             << std::bitset<sizeof(ResultRow) * 8u>(r).to_string();
    }
#endif
}
#endif

// A helper for InterleavedBackSubst.
template <typename BandingStorage>
inline void BackSubstBlock(typename BandingStorage::CoeffRow *state,
                           typename BandingStorage::Index num_columns,
                           const BandingStorage &bs,
                           typename BandingStorage::Index start_slot) {
    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;
    using ResultRow = typename BandingStorage::ResultRow;

    constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U);

    for (Index i = start_slot + kCoeffBits; i > start_slot;) {
        --i;
        CoeffRow cr = bs.GetCoeffs(i);
        ResultRow rr = bs.GetResult(i);
        for (Index j = 0; j < num_columns; ++j) {
            // Compute next solution bit at row i, column j (see derivation below)
            CoeffRow tmp = state[j] << 1;
            int bit = rocksdb::BitParity(tmp & cr) ^ ((rr >> j) & 1);
            tmp |= static_cast<CoeffRow>(bit);

            // Now tmp is solution at column j from row i for next kCoeffBits
            // more rows. Thus, for valid solution, the dot product of the
            // solution column with the coefficient row has to equal the result
            // at that column,
            //   BitParity(tmp & cr) == ((rr >> j) & 1)

            // Update state.
            state[j] = tmp;
        }
    }
}

template <typename BandingStorage, typename SolutionStorage>
void InterleavedBackSubst(const BandingStorage &bs, SolutionStorage *sol) {
    using CoeffRow = typename BandingStorage::CoeffRow;
    using Index = typename BandingStorage::Index;

    static_assert(sizeof(Index) == sizeof(typename SolutionStorage::Index),
                  "must be same");
    static_assert(sizeof(CoeffRow) == sizeof(typename SolutionStorage::CoeffRow),
                  "must be same");

    constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U),
                   kResultBits = SolutionStorage::kResultBits;

    constexpr bool debug = false;
    const Index num_slots = bs.GetNumSlots();
    // num_slots *MUST* be a multiple of kCoeffBits
    assert(num_slots >= kCoeffBits && num_slots % kCoeffBits == 0);
    sol->Prepare(num_slots);

    const Index num_blocks = sol->GetNumBlocks();
    const Index num_segments = sol->GetNumSegments();

    // We should be utilizing all available segments
    assert(num_segments == num_blocks * kResultBits);

    sLOG << "Backsubstitution: have" << num_blocks << "blocks," << num_segments
         << "segments for" << num_slots << "slots, kResultBits=" << kResultBits;

    // TODO: consider fixed-column specializations with stack-allocated state

    // A column-major buffer of the solution matrix, containing enough
    // recently-computed solution data to compute the next solution row
    // (based also on banding data).
    std::unique_ptr<CoeffRow[]> state{new CoeffRow[kResultBits]()};

    Index block = num_blocks;
    Index segment = num_segments;
    while (block > 0) {
        --block;
        sLOG << "Backsubstituting block" << block << "segment" << segment;
        BackSubstBlock(state.get(), kResultBits, bs, block * kCoeffBits);
        segment -= kResultBits;
        for (Index i = 0; i < kResultBits; ++i) {
            sol->SetSegment(segment + i, state[i]);
        }
    }
    // Verify everything processed
    assert(block == 0);
    assert(segment == 0);


#ifdef RIBBON_DUMP
    sLOG1 << sol->GetNumSegments() << "segments in" << sol->GetNumBlocks()
          << "blocks";
    for (Index i = 0; i < sol->GetNumSegments(); i++) {
        if (i % (sol->GetNumSegments() / sol->GetNumBlocks()) == 0)
            LOG1 << "================================";
        const CoeffRow seg = sol->GetSegment(i);
        LOG1 << "Seg " << std::setw(2) << i << " = " << std::hex << std::setw(4)
             << (uint64_t)seg << std::dec << " = "
             << std::bitset<sizeof(CoeffRow) * 8u>(seg).to_string();
    }
#endif
}

#ifdef _REENTRANT
template <typename BandingStorage, typename SolutionStorage>
void InterleavedBackSubstParallel(const BandingStorage &bs, SolutionStorage *sol, std::size_t num_threads) {
    using CoeffRow = typename BandingStorage::CoeffRow;
    using ResultRow = typename BandingStorage::ResultRow;
    using Index = typename BandingStorage::Index;

    static_assert(sizeof(Index) == sizeof(typename SolutionStorage::Index),
                  "must be same");
    static_assert(sizeof(CoeffRow) == sizeof(typename SolutionStorage::CoeffRow),
                  "must be same");

    constexpr auto kCoeffBits = static_cast<Index>(sizeof(CoeffRow) * 8U),
                   kResultBits = SolutionStorage::kResultBits;
    constexpr Index kMinBucketsPerThread = BandingStorage::kMinBucketsPerThread;
    constexpr Index kBucketSearchRange = BandingStorage::kBucketSearchRange;

    constexpr bool debug = false;
    const Index num_slots = bs.GetNumSlots();
    // num_slots *MUST* be a multiple of kCoeffBits
    assert(num_slots >= kCoeffBits && num_slots % kCoeffBits == 0);
    sol->Prepare(num_slots);

    const Index num_blocks = sol->GetNumBlocks();
    const Index num_segments = sol->GetNumSegments();

    // We should be utilizing all available segments
    assert(num_segments == num_blocks * kResultBits);

    sLOG << "Backsubstitution: have" << num_blocks << "blocks," << num_segments
         << "segments for" << num_slots << "slots, kResultBits=" << kResultBits;

    const Index num_buckets = bs.GetNumBuckets();
    /* NOTE: It is crucial that this is calculated the same as in BandingAddParallel
       so there are no conflicts. */
    std::size_t buckets_per_thread = num_buckets / num_threads;
    if (buckets_per_thread < kMinBucketsPerThread) {
        [[maybe_unused]] std::size_t orig_num_threads = num_threads;
        num_threads = num_buckets / kMinBucketsPerThread;
        assert(num_threads < orig_num_threads);
        if (num_threads <= 1) {
            InterleavedBackSubst(bs, sol);
            return;
        }
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    std::mutex output_mutex;
    for (std::size_t ti = 0; ti < num_threads; ++ti) {
        threads.emplace_back([&, ti]() {
            // TODO: consider fixed-column specializations with stack-allocated state

            // A column-major buffer of the solution matrix, containing enough
            // recently-computed solution data to compute the next solution row
            // (based also on banding data).
            std::unique_ptr<CoeffRow[]> state{new CoeffRow[kResultBits]()};

            const Index local_num_buckets =
                num_buckets / num_threads +
                (ti < num_buckets % num_threads);
            Index start_bucket =
                ti * (num_buckets / num_threads) +
                (ti < num_buckets % num_threads ? ti : num_buckets % num_threads);
            Index end_bucket = start_bucket + local_num_buckets;
            if constexpr (kBucketSearchRange > 0) {
                if (ti < num_threads - 1) {
                    end_bucket = bs.GetThreadBorderBucket(ti);
                }
                if (ti > 0) {
                    start_bucket = bs.GetThreadBorderBucket(ti - 1);
                }
            }

            Index start_slot, end_slot;
            start_slot = ti == 0
                                   ? start_bucket * BandingStorage::kBucketSize
                                   : start_bucket * BandingStorage::kBucketSize + kCoeffBits - 1;
            end_slot = end_bucket >= num_buckets
                                 ? num_slots
                                 : end_bucket * BandingStorage::kBucketSize + kCoeffBits - 1;

            Index start_block = start_slot / kCoeffBits;
            Index block = end_slot / kCoeffBits;
            Index segment = block * kResultBits;
            Index real_end_pos = block * kCoeffBits;
            /* if this isn't the last thread, check if the necessary amount
               of empty rows is present for parallel processing to work
               (if the back substitution is called with the same amount of
               threads as the insertion, this should always be the case) */
            if (end_bucket < num_buckets) {
                for (Index i = 1; i < kCoeffBits; ++i) {
                    CoeffRow cr = bs.GetCoeffs(end_slot - i);
                    if ((cr >> i) != 0) {
                        std::scoped_lock lk(output_mutex);
                        std::cerr << "ERROR: Thread " << ti << " found non-zero " <<
                                     "coefficient row in back substitution!\n";
                        /* FIXME: better error handling? */
                        abort();
                    }

                    /* we need to initialize state with the rows that are in the
                       bumped area but are processed by the next thread because
                       only complete blocks are processed */
                    if (end_slot - i >= real_end_pos) {
                        ResultRow rr = bs.GetResult(end_slot - i);
                        for (Index j = 0; j < kResultBits; ++j) {
                            CoeffRow tmp = state[j] << 1;
                            int bit = rocksdb::BitParity(tmp & cr) ^ ((rr >> j) & 1);
                            tmp |= static_cast<CoeffRow>(bit);
                            state[j] = tmp;
                        }
                    }

                }
            }

            while (block > start_block) {
                --block;
                /* FIXME: this debug statement won't work properly
                   when running this in parallel */
                sLOG << "Backsubstituting block" << block << "segment" << segment;
                BackSubstBlock(state.get(), kResultBits, bs, block * kCoeffBits);
                segment -= kResultBits;
                for (Index i = 0; i < kResultBits; ++i) {
                    sol->SetSegment(segment + i, state[i]);
                }
            }
            assert(segment == start_block * kResultBits);
        });
    }
    for (auto& t : threads) {
        t.join();
    }


#ifdef RIBBON_DUMP
    sLOG1 << sol->GetNumSegments() << "segments in" << sol->GetNumBlocks()
          << "blocks";
    for (Index i = 0; i < sol->GetNumSegments(); i++) {
        if (i % (sol->GetNumSegments() / sol->GetNumBlocks()) == 0)
            LOG1 << "================================";
        const CoeffRow seg = sol->GetSegment(i);
        LOG1 << "Seg " << std::setw(2) << i << " = " << std::hex << std::setw(4)
             << (uint64_t)seg << std::dec << " = "
             << std::bitset<sizeof(CoeffRow) * 8u>(seg).to_string();
    }
#endif
}
#endif

} // namespace ribbon
