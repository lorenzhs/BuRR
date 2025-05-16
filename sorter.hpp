//  Copyright (c) Lorenz Hübschle-Schneider
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "minimal_hasher.hpp"

namespace ribbon {

template <typename Index, bool IsFilter, bool sparse, typename data_t>
class Sorter {
public:
    void do_sort(data_t *begin, data_t *end, const MinimalHasher<Index, sparse> &mh, Index num_starts);
};
} // namespace ribbon
