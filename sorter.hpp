//  Copyright (c) Lorenz Hübschle-Schneider
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "minimal_hasher.hpp"

namespace ribbon {

struct SorterDummyData {};

template <typename Index, bool IsFilter, bool sparse, typename ResultRow>
class Sorter {
public:
    using data_t =
        std::conditional_t<IsFilter, uint64_t, std::pair<uint64_t, ResultRow>>;
    void do_sort(data_t *begin, data_t *end, const MinimalHasher<Index, sparse> &mh,
                 Index num_starts);
};
} // namespace ribbon
