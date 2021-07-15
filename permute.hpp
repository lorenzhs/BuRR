//  Copyright (c) Lorenz Hübschle-Schneider
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "config.hpp"
#include "hasher.hpp"

namespace ribbon {

template <typename Config>
class Permuter : public ChooseHasher<Config> {
public:
    using Index = typename Config::Index;

    using ChooseHasher<Config>::ChooseHasher;

    static constexpr Index StartToSort(Index startpos) {
        return (startpos ^ (Config::kBucketSize - 1));
    }

    // the transformation is identical
    static constexpr Index SortToStart(Index sortpos) {
        return StartToSort(sortpos);
    }

    static constexpr Index GetBucket(Index sortOrStartPos) {
        return sortOrStartPos / Config::kBucketSize;
    }

    static constexpr Index GetIntraBucket(Index sortpos) {
        // it's a compile-time power-of-two constant -> fast
        return sortpos % Config::kBucketSize;
    }

    static constexpr Index GetIntraBucketFromStart(Index startpos) {
        // it's a compile-time power-of-two constant -> fast
        return Config::kBucketSize - 1 - (startpos % Config::kBucketSize);
    }
};

} // namespace ribbon
