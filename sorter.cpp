//  Copyright (c) Lorenz Hübschle-Schneider
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#include "sorter.hpp"

#ifndef RIBBON_USE_STD_SORT
// Use in-place super-scalar radix sorter ips2ra, which is around 3x faster for
// the inputs used here
#include <ips2ra.hpp>
#endif

#include <bits/stdint-uintn.h>

#include <algorithm>
#include <functional>

template <typename Index, bool IsFilter, bool sparse, typename ResultRow>
void ribbon::Sorter<Index, IsFilter, sparse, ResultRow>::do_sort(
    ribbon::Sorter<Index, IsFilter, sparse, ResultRow>::data_t *begin,
    ribbon::Sorter<Index, IsFilter, sparse, ResultRow>::data_t *end,
    const ribbon::MinimalHasher<Index, sparse> &mh, Index num_starts) {
    using data_t = ribbon::Sorter<Index, IsFilter, sparse, ResultRow>::data_t;
    auto KeyEx = [&mh, num_starts](const data_t &mhc) -> Index {
        const auto hash = mh.GetHash(mhc);
        const auto start = mh.GetStart(hash, num_starts);
        return mh.StartToSort(start);
    };
#ifdef RIBBON_USE_STD_SORT
    // Use std::sort as a slow fallback
    std::sort(begin, end, [&KeyEx](const auto &a, const auto &b) {
        return KeyEx(a) < KeyEx(b);
    });
#else
    // prioritise speed over compile time
    ips2ra::sort(begin, end, KeyEx);
#endif
}

// Explicit ips2ra instantiations
template class ribbon::Sorter<uint32_t, true, false, ribbon::SorterDummyData>;
template class ribbon::Sorter<uint32_t, false, false, uint8_t>;
template class ribbon::Sorter<uint32_t, false, false, uint16_t>;
template class ribbon::Sorter<uint64_t, true, false, ribbon::SorterDummyData>;
template class ribbon::Sorter<uint64_t, false, false, uint8_t>;
template class ribbon::Sorter<uint64_t, false, false, uint16_t>;

// sparse configs - filter only for now
template class ribbon::Sorter<uint32_t, true, true, ribbon::SorterDummyData>;
template class ribbon::Sorter<uint64_t, true, true, ribbon::SorterDummyData>;
/*
template class ribbon::Sorter<uint32_t, false, true, uint8_t>;
template class ribbon::Sorter<uint32_t, false, true, uint16_t>;
*/
