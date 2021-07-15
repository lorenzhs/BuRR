//  Copyright (c) Lorenz Hübschle-Schneider
//  Copyright (c) Facebook, Inc. and its affiliates.
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "config.hpp"
#include "rocksdb/coding.h"
#include "rocksdb/slice.h"

#include <xxhash.h>

#include <string>

namespace ribbon::test {

struct BasicConfig : ribbon::DefaultConfig<uint32_t, uint8_t, rocksdb::Slice> {
    static constexpr ThreshMode kThreshMode = ThreshMode::normal;
    static constexpr bool log = false;
};

struct RetrievalConfig : public BasicConfig {
    static constexpr bool kIsFilter = false;
};

template <typename CoeffRow, typename ResultRow, typename Key>
struct DefaultRetrievalConfig : public DefaultConfig<CoeffRow, ResultRow, Key> {
    static constexpr bool kIsFilter = false;
    static constexpr ThreshMode kThreshMode = ThreshMode::normal;
    static constexpr bool log = false;
};


// Default config, but specify sizes in bits not types
template <size_t coeff_bits, size_t result_bits>
struct QuietRConfig : public RConfig<coeff_bits, result_bits> {
    static constexpr bool log = false;
};


// Generate semi-sequential keys
struct StandardKeyGen {
    StandardKeyGen(const std::string& prefix, uint64_t id)
        : id_(id), str_(prefix) {
        rocksdb::PutFixed64(&str_, /*placeholder*/ 0);
    }

    // Prefix (only one required)
    StandardKeyGen& operator++() {
        ++id_;
        return *this;
    }

    // Prefix (only one required)
    StandardKeyGen operator+(uint64_t i) {
        StandardKeyGen copy = *this;
        copy += i;
        return copy;
    }

    StandardKeyGen& operator+=(uint64_t i) {
        id_ += i;
        return *this;
    }

    const std::string& operator*() {
        // Use multiplication to mix things up a little in the key
        rocksdb::EncodeFixed64(&str_[str_.size() - 8],
                               id_ * uint64_t{0x1500000001});
        return str_;
    }

    bool operator==(const StandardKeyGen& other) const {
        // Same prefix is assumed
        return id_ == other.id_;
    }
    bool operator!=(const StandardKeyGen& other) const {
        // Same prefix is assumed
        return id_ != other.id_;
    }
    ssize_t operator-(const StandardKeyGen& other) const {
        return id_ - other.id_;
    }

    uint64_t id_;
    std::string str_;
};

// Generate small sequential keys, that can misbehave with sequential seeds
// as in https://github.com/Cyan4973/xxHash/issues/469.
// These keys are only heuristically unique, but that's OK with 64 bits,
// for testing purposes.
struct SmallKeyGen {
    SmallKeyGen(const std::string& prefix, uint64_t id) : id_(id) {
        // Hash the prefix for a heuristically unique offset
        id_ += XXH3_64bits(prefix.c_str(), prefix.size());
        rocksdb::PutFixed64(&str_, id_);
    }

    // Prefix (only one required)
    SmallKeyGen& operator++() {
        ++id_;
        return *this;
    }

    SmallKeyGen operator+(uint64_t i) {
        SmallKeyGen copy = *this;
        copy += i;
        return copy;
    }

    SmallKeyGen& operator+=(uint64_t i) {
        id_ += i;
        return *this;
    }

    const std::string& operator*() {
        rocksdb::EncodeFixed64(&str_[str_.size() - 8], id_);
        return str_;
    }

    bool operator==(const SmallKeyGen& other) const {
        return id_ == other.id_;
    }
    bool operator!=(const SmallKeyGen& other) const {
        return id_ != other.id_;
    }

    uint64_t id_;
    std::string str_;
};


struct RetrievalInputGen {
    RetrievalInputGen(const std::string& prefix, uint64_t id) : id_(id) {
        val_.first = prefix;
        rocksdb::PutFixed64(&val_.first, /*placeholder*/ 0);
    }

    // Prefix (only one required)
    RetrievalInputGen& operator++() {
        ++id_;
        return *this;
    }

    // Prefix (only one required)
    RetrievalInputGen operator+(uint64_t i) {
        RetrievalInputGen copy = *this;
        copy += i;
        return copy;
    }

    RetrievalInputGen& operator+=(uint64_t i) {
        id_ += i;
        return *this;
    }

    const std::pair<std::string, uint8_t>& operator*() {
        // Use multiplication to mix things up a little in the key
        rocksdb::EncodeFixed64(&val_.first[val_.first.size() - 8],
                               id_ * uint64_t{0x1500000001});
        // Occasionally repeat values etc.
        val_.second = static_cast<uint8_t>(id_ * 7 / 8);
        return val_;
    }

    const std::pair<std::string, uint8_t>* operator->() {
        return &**this;
    }

    ssize_t operator-(const RetrievalInputGen& other) const {
        return id_ - other.id_;
    }

    bool operator==(const RetrievalInputGen& other) const {
        // Same prefix is assumed
        return id_ == other.id_;
    }
    bool operator!=(const RetrievalInputGen& other) const {
        // Same prefix is assumed
        return id_ != other.id_;
    }

    uint64_t id_;
    std::pair<std::string, uint8_t> val_;
};

// Copied from rocksdb util/ribbon_test.cpp:
// For testing Poisson-distributed (or similar) statistics, get value for
// `stddevs_allowed` standard deviations above expected mean
// `expected_count`.
// (Poisson approximates Binomial only if probability of a trial being
// in the count is low.)
uint64_t PoissonUpperBound(double expected_count, double stddevs_allowed) {
    return static_cast<uint64_t>(
        expected_count + stddevs_allowed * std::sqrt(expected_count) + 1.0);
}

} // namespace ribbon::test
