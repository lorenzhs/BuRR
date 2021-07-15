//  Copyright (c) Lorenz Hübschle-Schneider
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "rocksdb/fastrange.h"

#include <bits/stdint-uintn.h>

#include <utility>

namespace ribbon {

template <typename Index, bool sparse>
class MinimalHasher {
public:
    using hash_t = uint64_t;
    using mhc_t = uint64_t;

    explicit MinimalHasher(Index kBucketSize, uint64_t fct, unsigned shift = 0)
        : kBSm1_(kBucketSize - 1), fct_(fct), shift_(shift) {}

    inline hash_t GetHash(mhc_t mhc) const {
        // return middle bits of the product (Knuth Multiplicative Hashing)
        // XXX why middle? why not most significant?
        return static_cast<hash_t>(
            (static_cast<__uint128_t>(mhc) * static_cast<__uint128_t>(fct_)) >> 32);
    }
    template <typename ResultRow>
    inline hash_t GetHash(const std::pair<mhc_t, ResultRow> &p) const {
        return GetHash(p.first);
    }

    inline Index GetStart(hash_t h, Index num_starts) const {
        if constexpr (sparse) {
            return rocksdb::FastRangeGeneric(h, num_starts >> shift_) << shift_;
        } else {
            return rocksdb::FastRange64(h, num_starts);
        }
    }

    inline Index StartToSort(Index startpos) const {
        return (startpos ^ kBSm1_);
    }

protected:
    Index kBSm1_; // kBucketSize minus 1
    uint64_t fct_;
    unsigned shift_;
};

} // namespace ribbon
