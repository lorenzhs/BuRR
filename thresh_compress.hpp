//  Copyright (c) Lorenz Hübschle-Schneider
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "config.hpp"
#include "permute.hpp"
#include "serialization.hpp"

#include <include/cuckoo_simple.h>

#include <memory>
#include <istream>
#include <ostream>

namespace ribbon {


template <typename Config>
class NormalThreshold : public Permuter<Config> {
public:
    using Permuter_ = Permuter<Config>;
    using Permuter_::Permuter;

    IMPORT_RIBBON_CONFIG(Config);

    void Prepare(Index, double) {
        sLOGC(Config::log) << "Using uncompressed thresholds";
    }

    static constexpr Index NoBumpThresh() {
        if constexpr (kSparseCoeffs) {
            return (kBucketSize >> Permuter_::shift_) - 1;
        } else {
            return kBucketSize - 1;
        }
    }

    inline Index Compress(Index intra_bucket) const {
        if constexpr (kSparseCoeffs) {
            // all intra_bucket values passed to this function are a multiple of
            // Permuter_::step_ plus some offset
            intra_bucket = (intra_bucket >> Permuter_::shift_);
        }
        return intra_bucket == 0 ? 0 : intra_bucket - 1;
    }

    void SerializeIntern(std::ostream &os) const {
        (void)os;
    }

    void DeserializeIntern(std::istream &is, bool switchendian, Index num_buckets) {
        (void)is;
        (void)switchendian;
        (void)num_buckets;
    }
};

template <typename Config>
class TwoBitThreshold : public Permuter<Config> {
public:
    using Permuter_ = Permuter<Config>;
    using Permuter_::Permuter;

    IMPORT_RIBBON_CONFIG(Config);

    static constexpr Index NoBumpThresh() {
        return 3;
    }

    void Prepare(Index num_slots, double slots_per_item) {
        if (num_slots == 0) {
            return;
        }
        const double eps = slots_per_item - 1.0;
        // TODO so many magic values
        if (kCoeffBits == 16) {
            lower_ = static_cast<Index>((0.6 + eps / 2) * kBucketSize);
            upper_ = static_cast<Index>((0.82 + eps / 2) * kBucketSize);
        } else if (kCoeffBits == 32) {
            lower_ = static_cast<Index>((0.7 + eps / 2) * kBucketSize);
            upper_ = static_cast<Index>((0.87 + eps / 2) * kBucketSize);
        } else {
            lower_ = static_cast<Index>((0.78 + 1.30 * eps) * kBucketSize);
            upper_ = static_cast<Index>((0.91 + 0.75 * eps) * kBucketSize);
        }
        sLOGC(Config::log) << "Using" << lower_ << "and" << upper_
                           << "as thresholds";
    }

    inline Index Compress(Index intrabucket) const {
        if (intrabucket >= kBucketSize) // no bumping
            return 3;
        else if (intrabucket > upper_) // some bumping
            return 2;
        else if (intrabucket > lower_) // lots of bumping
            return 1;
        else // everything bumped
            return 0;
    }

    void SerializeIntern(std::ostream &os) const {
        os.write(reinterpret_cast<const char *>(&lower_), sizeof(Index));
        os.write(reinterpret_cast<const char *>(&upper_), sizeof(Index));
    }

    void DeserializeIntern(std::istream &is, bool switchendian, Index num_buckets) {
        (void)num_buckets;
        is.read(reinterpret_cast<char *>(&lower_), sizeof(Index));
        if (switchendian && !bswap_generic(lower_))
            throw parse_error("error converting endianness");
        is.read(reinterpret_cast<char *>(&upper_), sizeof(Index));
        if (switchendian && !bswap_generic(upper_))
            throw parse_error("error converting endianness");
    }

    /*
    inline Index GetCompressedIntraBucket(Index sortpos, Index gap) const {
        return Compress(GetIntraBucket(sortpos, gap));
    }
    */

protected:
    Index lower_, upper_;
};

namespace {
// xxhash3 for cuckoo table
struct dysect_xxh3 {
    static constexpr std::string_view name = "xxhash3";
    static constexpr size_t significant_digits = 64;

    dysect_xxh3() = default;
    dysect_xxh3(size_t) {}

    inline uint64_t operator()(const uint64_t k) const {
        auto local = k;
        return XXH3_64bits(&local, sizeof(local));
    }
};

template <typename Index, bool log>
class OnePlusBase {
public:
    using table_t =
        dysect::cuckoo_standard<Index, Index, dysect_xxh3, dysect::cuckoo_config<4, 2>>;

    // 32 results in approximately 20% set bits in filter
    // 64 -> 35%
    // 128 -> 60%
    // 16 -> 10%
    static constexpr Index granularity = 32, bin_thresh = 32;
    // threshold before switching to the cuckoo table
    // minimum table size is 1024, 5% slack -> 975 as threshold
    static constexpr Index ht_thresh = 975;

    static constexpr Index NoBumpThresh() {
        return 1;
    }

    inline void Set(Index bucket, Index val) {
        assert(val < thresh_);
        buffer_.emplace_back(bucket, val);
        // plus_.insert(bucket + 1, val);
        // assert(plus_.find(bucket + 1) != plus_.end());
    }

    void Finalise(Index num_buckets) {
        if (buffer_.size() > ht_thresh) {
            plus_ = std::make_unique<table_t>(buffer_.size(), 1.03);
            for (const auto &[bucket, val] : buffer_) {
                // Key 0 not allowed -> offset by 1
                plus_->insert(bucket + 1, val);
            }
        } else {
            std::sort(
                buffer_.begin(), buffer_.end(),
                [](const auto &x, const auto &y) { return x.first < y.first; });
            buffer_.shrink_to_fit();
        }

        // Construct filter to probe before hitting hash table or binary search
        // TODO threshold tuning?
        if (buffer_.size() > bin_thresh) {
            const Index filter_size = (num_buckets + granularity - 1) / granularity;
            filter_.resize(filter_size);
            for (const auto &[bucket, _] : buffer_) {
                filter_[bucket / granularity] = true;
            }
            size_t c = 0;
            for (const bool set : filter_) {
                c += set;
            }
            sLOGC(log) << "1+ Bit Threshold:" << c << "of" << filter_size
                       << "bits set =" << c * 100.0 / filter_size << "% vs"
                       << buffer_.size() << "of" << num_buckets
                       << "buckets with thresholds ="
                       << buffer_.size() * 100.0 / num_buckets
                       << "%, buckets per filter bit:" << granularity;
        }

        // If we're using a hash table, deallocate the vector
        if (plus_ != nullptr) {
            buffer_ = std::vector<std::pair<Index, Index>>();
        }
        // hits = 0;
        // count = 0;
    }

    size_t NumEntries() const {
        return (plus_ == nullptr) ? buffer_.size() : plus_->size();
    }

    size_t Size() const {
        size_t capacity = (plus_ == nullptr) ? buffer_.size() : plus_->capacity;
        return capacity * sizeof(Index) * 2 + (filter_.size() + 7) / 8;
    }

    // unfortunately, this can't be const because there appears to be a bug in the
    // implementation of const iterators in DySECT
    void SerializeIntern(std::ostream &os) {
        os.write(reinterpret_cast<const char *>(&thresh_), sizeof(Index));
        std::size_t buffer_sz = NumEntries();
        os.write(reinterpret_cast<const char *>(&buffer_sz), sizeof(std::size_t));
        if (plus_ == nullptr) {
            for (const auto& e : buffer_) {
                os.write(reinterpret_cast<const char *>(&e.first), sizeof(Index));
                os.write(reinterpret_cast<const char *>(&e.second), sizeof(Index));
            }
        } else {
            for (auto it = plus_->begin(); it != plus_->end(); it++) {
                // the keys are offset by one in Finalise, so one needs to be
                // subtracted again here
                Index key = it->first - 1;
                os.write(reinterpret_cast<const char *>(&key), sizeof(Index));
                os.write(reinterpret_cast<const char *>(&it->second), sizeof(Index));
            }
        }
    }

    void DeserializeIntern(std::istream &is, bool switchendian, Index num_buckets) {
        is.read(reinterpret_cast<char *>(&thresh_), sizeof(Index));
        if (switchendian && !bswap_generic(thresh_))
            throw parse_error("error converting endianness");
        std::size_t buffer_sz = 0;
        is.read(reinterpret_cast<char *>(&buffer_sz), sizeof(std::size_t));
        if (switchendian && !bswap_generic(buffer_sz))
            throw parse_error("error converting endianness");
        buffer_.reserve(buffer_sz);
        Index key, value;
        /* NOTE: This could technically be improved a bit by storing whether the hash table
           or buffer was saved - if it was the original buffer, it doesn't need to be sorted
           again */
        for (std::size_t i = 0; i < buffer_sz; ++i) {
            is.read(reinterpret_cast<char *>(&key), sizeof(Index));
            // a call to bswap_type_supported is unnecessary because previous
            // calls to bswap_generic already check that Index is supported
            if (switchendian)
                bswap_generic(key);
            is.read(reinterpret_cast<char *>(&value), sizeof(Index));
            if (switchendian)
                bswap_generic(value);
            buffer_.emplace_back(key, value);
        }
        Finalise(num_buckets);
    }

protected:
    inline Index Get(Index bucket, const Index kBucketSize) const {
        // count++;
        if (!filter_.empty() && !filter_[bucket / granularity]) {
            // hits++;
            return kBucketSize;
        }
        if (plus_ != nullptr) {
            auto it = plus_->find(bucket + 1);
            if (it != plus_->end())
                return it->second;
        } else if (buffer_.size() > bin_thresh) {
            // binary search on buffer
            auto it = std::lower_bound(
                buffer_.begin(), buffer_.end(), bucket,
                [](const auto &x, const auto &b) { return x.first < b; });
            if (it != buffer_.end() && it->first == bucket) {
                return it->second;
            }
        } else {
            // linear search
            auto it = buffer_.begin();
            while (it != buffer_.end() && it->first < bucket)
                it++;
            if (it != buffer_.end() && it->first == bucket)
                return it->second;
        }
        return kBucketSize;
    }

    void Prepare(Index kBucketSize, Index /* kCoeffBits */, double slots_per_item) {
        // t = (1 + 2eps) B - c * stdev(bucket load)
        //   = (1 + 2eps) B - c' * sqrt(B/(1+eps))

        // cap epsilon at 0 to prevent useless thresholds in base ribbon
        const double eps = std::min(0.0, slots_per_item - 1.0);
        slots_per_item = std::min(1.0, slots_per_item);

        thresh_ = static_cast<Index>((1 + 2 * eps) * kBucketSize -
                                     0.5 * sqrt(kBucketSize / slots_per_item));
        sLOGC(log) << "1+ Bit Threshold: thresh =" << thresh_ << "="
                   << 1 + 2 * eps << "*" << kBucketSize << "- 0.5 * sqrt("
                   << kBucketSize / slots_per_item << "); B =" << kBucketSize;
    }

    Index thresh_;
    std::vector<bool> filter_;
    std::vector<std::pair<Index, Index>> buffer_;
    std::unique_ptr<table_t> plus_ = nullptr;
};

} // namespace

template <typename Config>
class OnePlusBitThreshold
    : public Permuter<Config>,
      public OnePlusBase<typename Config::Index, Config::log> {
public:
    using Permuter_ = Permuter<Config>;
    using Base = OnePlusBase<typename Config::Index, Config::log>;
    using Permuter_::Permuter;

    IMPORT_RIBBON_CONFIG(Config);

    void Prepare(Index /* num_slots */, double slots_per_item) {
        Base::Prepare(kBucketSize, kCoeffBits, slots_per_item);
    }

    inline Index Compress(Index intrabucket) const {
        if (intrabucket >= kBucketSize)
            // no bumping
            return 1;
        else if (intrabucket >= Base::thresh_)
            // some bumping
            return 0;
        else
            // the "plus" case, store threshold externally
            return 2;
    }

    inline Index Get(Index bucket) const {
        return Base::Get(bucket, kBucketSize);
    }
};

namespace {
template <ThreshMode mode, typename Config>
struct ThreshChooser;

template <typename Config>
struct ThreshChooser<ThreshMode::normal, Config> {
    static_assert(Config::kThreshMode == ThreshMode::normal, "wtf");
    using type = NormalThreshold<Config>;
};

template <typename Config>
struct ThreshChooser<ThreshMode::onebit, Config> {
    static_assert(Config::kThreshMode == ThreshMode::onebit, "wtf");
    using type = OnePlusBitThreshold<Config>;
};

template <typename Config>
struct ThreshChooser<ThreshMode::twobit, Config> {
    static_assert(Config::kThreshMode == ThreshMode::twobit, "wtf");
    using type = TwoBitThreshold<Config>;
};

} // namespace

template <typename Config>
using ChooseThreshold = typename ThreshChooser<Config::kThreshMode, Config>::type;

} // namespace ribbon
