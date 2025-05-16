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

    void Prepare(Index, double, Index = 1) {
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

    void SerializeIntern(std::ostream &) const {
    }

    void DeserializeIntern(std::istream &, bool, Index, Index = 1) {
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

    void Prepare(Index num_slots, double slots_per_item, Index = 1) {
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

    void DeserializeIntern(std::istream &is, bool switchendian, Index, Index = 1) {
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

template <typename Index, bool log, bool kMultipleRibbons = false>
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

    template <bool uv = kMultipleRibbons>
    std::enable_if_t<!uv, void>
    Finalise(Index num_buckets) {
        static_assert(!kMultipleRibbons);
        FinaliseIntern(num_buckets, filter_, buffer_, plus_);
    }

    template <bool uv = kMultipleRibbons>
    std::enable_if_t<uv, void>
    Finalise(Index num_buckets) {
        static_assert(kMultipleRibbons);
        for (size_t i = 0; i < buffer_.size(); ++i) {
            FinaliseIntern(num_buckets, filter_[i], buffer_[i], plus_[i]);
        }
    }

    template <bool uv = kMultipleRibbons>
    inline std::enable_if_t<!uv, void>
    Set(Index bucket, Index val) {
        static_assert(!kMultipleRibbons);
        assert(val < thresh_);
        buffer_.emplace_back(bucket, val);
        // plus_.insert(bucket + 1, val);
        // assert(plus_.find(bucket + 1) != plus_.end());
    }

    template <bool uv = kMultipleRibbons>
    inline std::enable_if_t<uv, void>
    Set(Index bucket, Index val, size_t ribbon_idx) {
        static_assert(kMultipleRibbons);
        assert(val < thresh_);
        buffer_[ribbon_idx].emplace_back(bucket, val);
    }

    size_t NumEntries() const {
        if constexpr (kMultipleRibbons) {
            size_t sz = 0;
            // this is only used to output debug info on the number of total
            // entries, so just add up everything in all ribbons
            for (size_t i = 0; i < buffer_.size(); ++i) {
                sz += (plus_[i] == nullptr) ? buffer_[i].size() : plus_[i]->size();
            }
            return sz;
        } else {
            return (plus_ == nullptr) ? buffer_.size() : plus_->size();
        }
    }

    size_t Size() const {
        // FIXME: should the overhead from the vectors maybe be counted as well?
        if constexpr (kMultipleRibbons) {
            size_t sz = 0;
            for (size_t i = 0; i < buffer_.size(); ++i) {
                size_t capacity = (plus_[i] == nullptr) ? buffer_[i].size() : plus_[i]->capacity;
                sz += capacity * sizeof(Index) * 2 + (filter_[i].size() + 7) / 8;
            }
            return sz;
        } else {
            size_t capacity = (plus_ == nullptr) ? buffer_.size() : plus_->capacity;
            return capacity * sizeof(Index) * 2 + (filter_.size() + 7) / 8;
        }
    }

    // unfortunately, this can't be const because there appears to be a bug in the
    // implementation of const iterators in DySECT
    void SerializeIntern(std::ostream &os) {
        os.write(reinterpret_cast<const char *>(&thresh_), sizeof(Index));
        if constexpr (kMultipleRibbons) {
            for (size_t i = 0; i < buffer_.size(); ++i) {
                SerializeSingleBuffer(os, buffer_[i], plus_[i]);
            }
        } else {
            SerializeSingleBuffer(os, buffer_, plus_);
        }
    }

    template <bool uv = kMultipleRibbons>
    std::enable_if_t<!uv, void>
    DeserializeIntern(std::istream &is, bool switchendian, Index num_buckets) {
        static_assert(!kMultipleRibbons);
        is.read(reinterpret_cast<char *>(&thresh_), sizeof(Index));
        if (switchendian && !bswap_generic(thresh_))
            throw parse_error("error converting endianness");
        DeserializeSingleBuffer(is, switchendian, buffer_);
        Finalise(num_buckets);
    }

    template <bool uv = kMultipleRibbons>
    std::enable_if_t<uv, void>
    DeserializeIntern(std::istream &is, bool switchendian, Index num_buckets, Index num_ribbons) {
        static_assert(kMultipleRibbons);
        is.read(reinterpret_cast<char *>(&thresh_), sizeof(Index));
        if (switchendian && !bswap_generic(thresh_))
            throw parse_error("error converting endianness");
        PrepareVLR(num_ribbons);
        for (Index i = 0; i < num_ribbons; ++i) {
            DeserializeSingleBuffer(is, switchendian, buffer_[i]);
        }
        Finalise(num_buckets);
    }

protected:
    // TODO: check if these are inlined (otherwise, split up GetIntern completely
    // so no unnecessary parameters have to get passed)
    template <bool uv = kMultipleRibbons>
    inline std::enable_if_t<!uv, Index>
    Get(Index bucket, const Index kBucketSize) const {
        static_assert(!kMultipleRibbons);
        return GetIntern(bucket, kBucketSize, filter_, buffer_, plus_);
    }

    template <bool uv = kMultipleRibbons>
    inline std::enable_if_t<uv, Index>
    Get(Index bucket, const Index kBucketSize, size_t ribbon_idx) const {
        static_assert(kMultipleRibbons);
        return GetIntern(bucket, kBucketSize, filter_[ribbon_idx], buffer_[ribbon_idx], plus_[ribbon_idx]);
    }

    template <bool uv = kMultipleRibbons>
    std::enable_if_t<uv, void>
    PrepareVLR(Index num_ribbons) {
        static_assert(kMultipleRibbons);
        assert(num_ribbons > 0);
        filter_.resize(num_ribbons);
        filter_.shrink_to_fit();
        buffer_.resize(num_ribbons);
        buffer_.shrink_to_fit();
        plus_ = std::make_unique<std::unique_ptr<table_t>[]>(num_ribbons);
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

private:
    void SerializeSingleBuffer(std::ostream &os,
                               const std::vector<std::pair<Index, Index>> &buffer,
                               const std::unique_ptr<table_t> &plus) {
        std::size_t buffer_sz = (plus == nullptr) ? buffer.size() : plus->size();
        os.write(reinterpret_cast<const char *>(&buffer_sz), sizeof(std::size_t));
        if (plus == nullptr) {
            for (const auto& e : buffer) {
                os.write(reinterpret_cast<const char *>(&e.first), sizeof(Index));
                os.write(reinterpret_cast<const char *>(&e.second), sizeof(Index));
            }
        } else {
            for (auto it = plus->begin(); it != plus->end(); it++) {
                // the keys are offset by one in Finalise, so one needs to be
                // subtracted again here
                Index key = it->first - 1;
                os.write(reinterpret_cast<const char *>(&key), sizeof(Index));
                os.write(reinterpret_cast<const char *>(&it->second), sizeof(Index));
            }
        }
    }

    void DeserializeSingleBuffer(std::istream &is, bool switchendian,
                                 std::vector<std::pair<Index, Index>> &buffer) {
        std::size_t buffer_sz = 0;
        is.read(reinterpret_cast<char *>(&buffer_sz), sizeof(std::size_t));
        if (switchendian && !bswap_generic(buffer_sz))
            throw parse_error("error converting endianness");
        buffer.reserve(buffer_sz);
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
            buffer.emplace_back(key, value);
        }
    }

    void FinaliseIntern(Index num_buckets, std::vector<bool> &filter,
                        std::vector<std::pair<Index, Index>> &buffer, std::unique_ptr<table_t> &plus) {
        plus = nullptr;
        if (buffer.size() > ht_thresh) {
            plus = std::make_unique<table_t>(buffer.size(), 1.03);
            for (const auto &[bucket, val] : buffer) {
                // Key 0 not allowed -> offset by 1
                plus->insert(bucket + 1, val);
            }
        } else {
            std::sort(
                buffer.begin(), buffer.end(),
                [](const auto &x, const auto &y) { return x.first < y.first; });
            buffer.shrink_to_fit();
        }

        // Construct filter to probe before hitting hash table or binary search
        // TODO threshold tuning?
        if (buffer.size() > bin_thresh) {
            const Index filter_size = (num_buckets + granularity - 1) / granularity;
            filter.resize(filter_size);
            for (const auto &[bucket, _] : buffer) {
                filter[bucket / granularity] = true;
            }
            size_t c = 0;
            for (const bool set : filter) {
                c += set;
            }
            sLOGC(log) << "1+ Bit Threshold:" << c << "of" << filter_size
                       << "bits set =" << c * 100.0 / filter_size << "% vs"
                       << buffer.size() << "of" << num_buckets
                       << "buckets with thresholds ="
                       << buffer.size() * 100.0 / num_buckets
                       << "%, buckets per filter bit:" << granularity;
        }

        // If we're using a hash table, deallocate the vector
        if (plus != nullptr) {
            buffer = std::vector<std::pair<Index, Index>>();
        }
        // hits = 0;
        // count = 0;
    }

    inline Index GetIntern(Index bucket, const Index kBucketSize, const std::vector<bool> &filter,
                           const std::vector<std::pair<Index, Index>> &buffer, const std::unique_ptr<table_t> &plus) const {
        // count++;
        if (!filter.empty() && !filter[bucket / granularity]) {
            // hits++;
            return kBucketSize;
        }
        if (plus != nullptr) {
            auto it = plus->find(bucket + 1);
            if (it != plus->end())
                return it->second;
        } else if (buffer.size() > bin_thresh) {
            // binary search on buffer
            auto it = std::lower_bound(
                buffer.begin(), buffer.end(), bucket,
                [](const auto &x, const auto &b) { return x.first < b; });
            if (it != buffer.end() && it->first == bucket) {
                return it->second;
            }
        } else {
            // linear search
            auto it = buffer.begin();
            while (it != buffer.end() && it->first < bucket)
                it++;
            if (it != buffer.end() && it->first == bucket)
                return it->second;
        }
        return kBucketSize;
    }

    std::conditional_t<kMultipleRibbons, std::vector<std::vector<bool>>, std::vector<bool>> filter_;
    std::conditional_t<kMultipleRibbons, std::vector<std::vector<std::pair<Index, Index>>>, std::vector<std::pair<Index, Index>>> buffer_;
    std::conditional_t<kMultipleRibbons, std::unique_ptr<std::unique_ptr<table_t>[]>, std::unique_ptr<table_t>> plus_;
protected:
    Index thresh_;
};

} // namespace

template <typename Config>
class OnePlusBitThreshold
    : public Permuter<Config>,
      public OnePlusBase<typename Config::Index, Config::log, Config::kUseVLR && !Config::kVLRShareMeta> {
public:
    using Permuter_ = Permuter<Config>;
    using Base = OnePlusBase<typename Config::Index, Config::log, Config::kUseVLR && !Config::kVLRShareMeta>;
    using Permuter_::Permuter;

    IMPORT_RIBBON_CONFIG(Config);

    template <bool uv = Config::kUseVLR && !Config::kVLRShareMeta>
    std::enable_if_t<!uv, void>
    Prepare(Index /* num_slots */, double slots_per_item) {
        static_assert(!Config::kUseVLR || Config::kVLRShareMeta);
        Base::Prepare(kBucketSize, kCoeffBits, slots_per_item);
    }

    template <bool uv = Config::kUseVLR && !Config::kVLRShareMeta>
    std::enable_if_t<uv, void>
    Prepare(Index /* num_slots */, double slots_per_item, Index num_ribbons) {
        static_assert(Config::kUseVLR && !Config::kVLRShareMeta);
        Base::PrepareVLR(num_ribbons); // initialize vectors needed for vlr version
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

    template <bool uv = Config::kUseVLR && !Config::kVLRShareMeta>
    inline std::enable_if_t<!uv, Index>
    Get(Index bucket) const {
        static_assert(!Config::kUseVLR || Config::kVLRShareMeta);
        return Base::Get(bucket, kBucketSize);
    }

    template <bool uv = Config::kUseVLR && !Config::kVLRShareMeta>
    inline std::enable_if_t<uv, Index>
    Get(Index bucket, size_t ribbon_idx) const {
        static_assert(Config::kUseVLR && !Config::kVLRShareMeta);
        return Base::Get(bucket, kBucketSize, ribbon_idx);
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
