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

} // namespace ribbon
