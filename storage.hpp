//  Copyright (c) Lorenz Hübschle-Schneider
//  Copyright (c) Facebook, Inc. and its affiliates.
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "config.hpp"
#include "serialization.hpp"

#include <cassert>
#include <memory>
#include <ostream>
#include <istream>

namespace ribbon {

namespace {
template <typename Config>
class MetaStorage {
public:
    IMPORT_RIBBON_CONFIG(Config);
    static constexpr bool debug = true;

    // TODO: if Config::kSparseCoeffs, we don't need to store the least
    // significant bits of the threshold because the start positions (and ergo
    // thresholds) are aligned to log(#set bits) or something like that
    static constexpr unsigned meta_bits = thresh_meta_bits<Config>;
    using meta_t = at_least_t<meta_bits>;
    static constexpr unsigned meta_t_bits = 8u * sizeof(meta_t);
    // whether meta_bits divides meta_t_bits cleanly, i.e., we always only need
    // one meta_t to get a bucket's metadata
    static constexpr bool div_clean = (meta_t_bits % meta_bits) == 0;
    using fetch_t =
        std::conditional_t<div_clean, meta_t, at_least_t<2 * meta_t_bits>>;
    static constexpr unsigned fetch_bits = 8u * sizeof(fetch_t),
                              items_per_fetch = fetch_bits / meta_bits;
    static constexpr fetch_t extractor_mask = (fetch_t{1} << meta_bits) - 1;
    static constexpr Index shift_mask = items_per_fetch - 1;
    /* sanity check to make sure the concurrent usage is safe */
    static_assert(sizeof(fetch_t) <= 2 * sizeof(meta_t));


    void Prepare(size_t num_slots) {
        assert(num_slots >= kCoeffBits);
        num_slots_ = num_slots;
        Index num_starts = num_slots - kCoeffBits + 1;
        num_buckets_ = (num_starts + kBucketSize - 1) / kBucketSize;

        // +!div_clean at the end so we don't fetch beyond the bounds, even if
        // we don't use it
        size_t size = GetMetaSize() + !div_clean;
        sLOGC(Config::log) << "Meta: allocating" << size << "entries of"
                           << sizeof(meta_t) << "Bytes each";
        meta_ = std::make_unique<meta_t[]>(size);
        if constexpr (kThreshMode == ThreshMode::onebit) {
            assert(size == (num_buckets_ + 7) / 8);
        } else if constexpr (kThreshMode == ThreshMode::twobit) {
            assert(size == (num_buckets_ + 3) / 4);
        }
    }
    void Reset() {
        meta_.reset();
        thread_borders_.reset();
    }

    /* FIXME: document that this is not clamped, i.e. should not be used by first/last thread */
       /* (or maybe just change that?) */
    /* Return the first bucket after 'start' that can be safely read/written
       without the possibility of a read/write of a bucket before 'start'
       interfering with it. */
    Index GetNextSafeStart(Index start) {
        if constexpr (div_clean) {
            if (start % items_per_fetch != 0) {
                start += items_per_fetch - (start % items_per_fetch);
            }
        } else {
            Index start_bit = start * meta_bits;
            /* GetMeta/SetMeta always read/write two entries in meta_,
               so if another bucket metadata starts before 'start' in
               the same meta_ entry, we need to skip over this meta_
               entry *and* the next one. If no other bucket metadata
               can start before this one in the same meta_ entry, we
               only need to skip over one meta_ entry. */
            if (start_bit % meta_t_bits >= meta_bits) {
                start_bit += 2 * meta_t_bits;
            } else {
                start_bit += meta_t_bits;
            }
            /* find first bucket that starts in this meta_ entry */
            start_bit -= (start_bit % meta_t_bits);
            start = (start_bit + meta_bits - 1) / meta_bits;
        }
        return start;
    }

    /* Return the last bucket before 'end' that can be safely read/written
       without the possibility of a read/write of a bucket after 'end'
       interfering with it. */
    Index GetPrevSafeEnd(Index end) {
        assert(num_buckets_ > 0);
        if (end >= num_buckets_ - 1)
            return num_buckets_ - 1;
        if constexpr (div_clean) {
            if (end % items_per_fetch != items_per_fetch - 1) {
                end -= (end % items_per_fetch) + 1;
            }
        } else {
            Index end_bit = end * meta_bits + meta_bits - 1;
            /* If the metadata is at the very end of a meta_ entry,
               we just need to jump back one entry, otherwise two. */
            if (end_bit % meta_t_bits == meta_t_bits - 1) {
                end_bit -= meta_t_bits;
            } else {
                end_bit -= 2 * meta_t_bits;
            }
            /* find last bucket that starts in this meta_ entry */
            end_bit += meta_t_bits - (end_bit % meta_t_bits) - 1;
            end = end_bit / meta_bits;
        }
        return end;
    }

    inline void PrefetchMeta(Index bucket) const {
        const Index fetch_bucket = div_clean ? bucket / items_per_fetch
                                             : bucket * meta_bits / meta_t_bits;
        __builtin_prefetch(meta_.get() + fetch_bucket,
                           /* rw */ 0, /* locality */ 1);
    }
    inline meta_t GetMeta(Index bucket) const {
        assert(bucket < num_buckets_);
        if constexpr (div_clean) {
            const meta_t fetch = meta_[bucket / items_per_fetch];
            const unsigned shift = meta_bits * (bucket & shift_mask);
            assert(shift < fetch_bits);
            return (fetch >> shift) & extractor_mask;
        } else {
            // find the fetch position first
            Index start_bit = bucket * meta_bits;
            Index fetch_bucket = start_bit / meta_t_bits;
            fetch_t fetch;
            memcpy(&fetch, meta_.get() + fetch_bucket, sizeof(fetch_t));
            // start_bit now indicates which bits of 'fetch' we need
            start_bit -= fetch_bucket * meta_t_bits;
            return (fetch >> start_bit) & extractor_mask;
        }
    }
    inline void SetMeta(Index bucket, meta_t val) {
        assert(bucket < num_buckets_);
        assert(val <= extractor_mask);
        if constexpr (div_clean) {
            const Index pos = bucket / items_per_fetch;
            const unsigned shift = meta_bits * (bucket & shift_mask);
            meta_[pos] &= ~static_cast<meta_t>(extractor_mask << shift);
            meta_[pos] |= (val << shift);
        } else {
            // find the fetch position first
            Index start_bit = bucket * meta_bits;
            Index fetch_bucket = start_bit / meta_t_bits;
            // start_bit now indicates which bits of 'fetch' we need
            start_bit -= fetch_bucket * meta_t_bits;
            fetch_t* fetch =
                reinterpret_cast<fetch_t*>(meta_.get() + fetch_bucket);
            *fetch &= ~(extractor_mask << start_bit);
            *fetch |= (val << start_bit);
        }
        assert(GetMeta(bucket) == val);
    }

    // invalidates other->meta_!
    template <typename Other>
    void MoveMetadata(Other* other) {
        assert(num_buckets_ == other->num_buckets_);
        meta_.swap(other->meta_);
    }

    // clang-format off
    inline Index GetNumSlots() const { return num_slots_; }
    inline Index GetNumStarts() const { return num_slots_ - kCoeffBits + 1; }
    inline Index GetNumBuckets() const { return num_buckets_; }
    // clang-format on

    inline void SetNumThreadBorders(std::size_t num_borders) {
        thread_borders_ = std::make_unique<Index[]>(num_borders);
        num_thread_borders_ = num_borders;
    }

    inline Index GetThreadBorderBucket(std::size_t t) const {
        /* this is only called by the back substitution */
        if (t >= num_thread_borders_) {
            std::cerr << "Wrong number of threads given to back substitution.\n";
            abort();
        }
        return thread_borders_[t];
    }

    inline void SetThreadBorderBucket(std::size_t t, Index bucket) {
        thread_borders_[t] = bucket;
    }

    size_t Size() const {
        const size_t meta_bytes = GetMetaSize() * sizeof(meta_t);
        sLOGC(Config::log) << "\tmeta size: " << num_buckets_ << "*"
                           << meta_bits << "bits ->" << meta_bytes << "Bytes";
        return meta_bytes + 2 * sizeof(Index) /* don't count num_buckets */;
    }

    void SerializeIntern(std::ostream &os) const {
        os.write(reinterpret_cast<const char *>(&num_slots_), sizeof(Index));
        size_t size = GetMetaSize() + !div_clean;
        os.write(reinterpret_cast<const char *>(meta_.get()), sizeof(meta_t) * size);
    }

    void DeserializeIntern(std::istream &is, bool switchendian) {
        is.read(reinterpret_cast<char *>(&num_slots_), sizeof(Index));
        if (switchendian && !bswap_generic(num_slots_))
            throw parse_error("error converting endianness");
        Index num_starts = num_slots_ - kCoeffBits + 1;
        num_buckets_ = (num_starts + kBucketSize - 1) / kBucketSize;
        size_t size = GetMetaSize() + !div_clean;
        meta_ = std::make_unique<meta_t[]>(size);
        is.read(reinterpret_cast<char *>(meta_.get()), sizeof(meta_t) * size);
        if (switchendian && sizeof(meta_t) > 1) {
            if (!bswap_type_supported<meta_t>())
                throw parse_error("error converting endianness");
            for (size_t i = 0; i < size; ++i) {
                bswap_generic(meta_[i]);
            }
        }
    }

protected:
    size_t GetMetaSize() const {
        return (num_buckets_ * meta_bits + meta_t_bits - 1) / meta_t_bits;
    }
    // num_buckets_ is for debugging only & can be recomputed easily
    Index num_slots_ = 0, num_buckets_ = 0;
    std::unique_ptr<meta_t[]> meta_;
    // this is only here so the back substitution can
    // access the thread borders used for the insertion
    std::unique_ptr<Index[]> thread_borders_;
    std::size_t num_thread_borders_ = 0;
};
} // namespace

template <typename Config>
class BasicStorage : public MetaStorage<Config> {
public:
    IMPORT_RIBBON_CONFIG(Config);
    using Super = MetaStorage<Config>;

    BasicStorage() = default;
    explicit BasicStorage(Index num_slots) {
        if (num_slots > 0)
            Prepare(num_slots);
    }

    void Prepare(size_t num_slots) {
        Super::Prepare(num_slots);

        coeffs_ = std::make_unique<CoeffRow[]>(num_slots);
        results_ = std::make_unique<ResultRow[]>(num_slots);
    }

    void Reset() {
        coeffs_.reset();
        results_.reset();
        Super::Reset();
    }

    inline void PrefetchQuery(Index i) const {
        __builtin_prefetch(&results_[i], /* rw */ 0, /* locality */ 1);
    }

    inline CoeffRow GetCoeffs(Index row) const {
        return coeffs_[row];
    }
    inline void SetCoeffs(Index row, CoeffRow val) {
        coeffs_[row] = val;
    }
    inline ResultRow GetResult(Index row) const {
        return results_[row];
    }
    inline void SetResult(Index row, ResultRow val) {
        results_[row] = val;
    }

    // dummy interface
    using State = Index;
    inline State PrepareGetResult(Index row) const {
        return row;
    }
    inline ResultRow GetFromState(const State& state) const {
        return GetResult(state);
    }
    inline State AdvanceState(State state) const {
        return state + 1;
    }

    template <typename Iterator, typename Hasher, typename Callback>
    void AddRange(Iterator begin, Iterator end, const Hasher& hasher,
                  Callback bump_callback) {
        BandingAddRange(this, hasher, begin, end, bump_callback);
    }

    size_t Size() const {
        return Super::num_slots_ * sizeof(ResultRow) + Super::Size();
    }

    void SerializeIntern(std::ostream &os) const {
        Super::SerializeIntern(os);
        os.write(reinterpret_cast<const char *>(results_.get()), sizeof(ResultRow) * Super::num_slots_);
    }

    void DeserializeIntern(std::istream &is, bool switchendian) {
        Super::DeserializeIntern(is, switchendian);
        results_ = std::make_unique<ResultRow[]>(Super::num_slots_);
        is.read(reinterpret_cast<char *>(results_.get()), sizeof(ResultRow) * Super::num_slots_);
        if (switchendian && sizeof(ResultRow) > 1) {
            if (!bswap_type_supported<ResultRow>())
                throw parse_error("error converting endianness");
            for (Index i = 0; i < Super::num_slots_; ++i) {
                bswap_generic(results_[i]);
            }
        }
    }

protected:
    std::unique_ptr<CoeffRow[]> coeffs_;
    std::unique_ptr<ResultRow[]> results_;
};

// only for backsubstition, can't be used for AddRange
template <typename Config>
class InterleavedSolutionStorage : public MetaStorage<Config> {
public:
    IMPORT_RIBBON_CONFIG(Config);
    using Super = MetaStorage<Config>;

    InterleavedSolutionStorage() = default;

    explicit InterleavedSolutionStorage(Index num_slots) {
        if (num_slots > 0)
            Prepare(num_slots);
    }

    void Prepare(size_t num_slots) {
        Super::Prepare(num_slots);
        size_t size = GetNumSegments() * sizeof(CoeffRow);
        data_ = std::make_unique<unsigned char[]>(size);
    }

    void PrefetchQuery(Index segment_num) const {
        __builtin_prefetch(data_.get() + segment_num * sizeof(CoeffRow),
                           /* rw */ 0, /* locality */ 1);
    }

    inline CoeffRow GetSegment(Index segment_num) const {
        CoeffRow result;
        memcpy(&result, data_.get() + segment_num * sizeof(CoeffRow),
               sizeof(CoeffRow));
        return result;
        // return *reinterpret_cast<CoeffRow *>(data_.get() +
        //                                    segment_num * sizeof(CoeffRow));
    }
    inline void SetSegment(Index segment_num, CoeffRow val) {
        memcpy(data_.get() + segment_num * sizeof(CoeffRow), &val,
               sizeof(CoeffRow));
        // *reinterpret_cast<CoeffRow *>(data_.get() +
        //                              segment_num * sizeof(CoeffRow)) = val;
    }

    // clang-format off
    inline Index GetNumBlocks() const { return Super::num_slots_ / kCoeffBits; }
    inline Index GetNumSegments() const { return kResultBits * GetNumBlocks(); }
    // clang-format on

    size_t Size() const {
        return GetNumSegments() * sizeof(CoeffRow) + Super::Size();
    };

    void SerializeIntern(std::ostream &os) const {
        Super::SerializeIntern(os);
        size_t size = GetNumSegments() * sizeof(CoeffRow);
        os.write(reinterpret_cast<const char*>(data_.get()), size);
    }

    void DeserializeIntern(std::istream &is, bool switchendian) {
        Super::DeserializeIntern(is, switchendian);
        size_t size = GetNumSegments() * sizeof(CoeffRow);
        data_ = std::make_unique<unsigned char[]>(size);
        is.read(reinterpret_cast<char*>(data_.get()), size);
        if (switchendian && sizeof(CoeffRow) > 1) {
            if (!bswap_type_supported<CoeffRow>())
                throw parse_error("error converting endianness");
            for (Index i = 0; i < GetNumSegments(); ++i) {
                // this could probably be made a bit more efficient
                CoeffRow seg = GetSegment(i);
                bswap_generic(seg);
                SetSegment(i, seg);
            }
        }
    }

protected:
    std::unique_ptr<unsigned char[]> data_;
};


// For now, two-bit thresholds only
template <typename Config>
class CacheLineStorage {
public:
    IMPORT_RIBBON_CONFIG(Config);

    static constexpr Index
        // bits to store an entire bucket, incl metadata (-> adjust bucket size)
        bucketbits = kBucketSize * kResultBits,
        // yikes, use fake larger CL if a bucket wouldn't fit
        clbits = (bucketbits > 512) ? bucketbits : 512, clsize = clbits / 8u,
        buckets_per_cl = clbits / bucketbits,
        items_per_row = 8u * sizeof(ResultRow) / kResultBits,
        meta_bits_per_bucket = thresh_meta_bits<Config>,
        // TODO this could be refined to pack the items, currently we round to
        // bytes (also might fail for meta_bits_per_bucket > 8 if buckets_per_cl
        // > 1 but when is that ever the case?)
        metabytes_per_cl = buckets_per_cl * (meta_bits_per_bucket + 7) / 8,
        metarows_per_cl =
            (metabytes_per_cl + sizeof(ResultRow) - 1) / sizeof(ResultRow),
        items_per_cl = (clbits - 8u * metabytes_per_cl) / kResultBits;
    static constexpr bool should_use_compression =
        tlx::integer_log2_ceil(kBucketSize) * buckets_per_cl > kResultBits;
    // static_assert(should_use_compression == (kThreshMode != ThreshMode::normal),
    //              "bad config: check kThreshMode");
    static constexpr ResultRow maxval = (1ul << kResultBits) - 1;
    using meta_t = at_least_t<8 * metabytes_per_cl>;
    using meta_item_t = at_least_t<meta_bits_per_bucket>;

    // haven't implemented more than 128 meta-bits for lack of a larger type,
    // would need to do indexing
    static_assert(metabytes_per_cl <= 16, "not implemented");
    static_assert(!(buckets_per_cl > 1 && meta_bits_per_bucket > 8),
                  "not implemented");

    static constexpr bool debug = false;

    CacheLineStorage() = default;

    explicit CacheLineStorage(Index num_slots) {
        // loudly warn about suboptimal config choice
        if constexpr (should_use_compression !=
                      (kThreshMode != ThreshMode::normal)) {
            sLOG1 << "WARNING: CacheLineStorage disagrees about your choice of "
                     "threshold compressor:"
                  << (should_use_compression
                          ? "SHOULD use compression but isn't"
                          : "SHOULD NOT use compression but is")
                  << "kThreshMode =" << (int)kThreshMode
                  << "uncompressed thresholds would need"
                  << (tlx::integer_log2_ceil(kBucketSize) * buckets_per_cl)
                  << "bits, have" << kResultBits << "bits for thresholds";
        }
        if (num_slots > 0)
            Prepare(num_slots);
    }

    void Prepare(size_t num_slots) {
        num_slots_ = num_slots;

        size_t cls = ((num_slots + items_per_cl - 1) / items_per_cl);
        size_ = cls * clsize;
        sLOGC(Config::log) << "Preparing for" << num_slots << "slots @"
                           << items_per_cl
                           << "items per cl, kResultBits =" << kResultBits
                           << "with" << metabytes_per_cl << "B/CL metainf ->"
                           << cls << "CLs," << size_ << "rows,"
                           << GetNumBuckets() << "buckets; efficiency:"
                           << items_per_cl * 1.0 / (clbits / kResultBits)
                           << "bucketbits =" << bucketbits << "clbits =" << clbits
                           << "thresh mode" << (int)kThreshMode;
        data_ = std::make_unique<unsigned char[]>(size_);
    }

    inline meta_item_t GetMeta(Index bucket) const {
        assert(bucket < GetNumBuckets());
        // clbits / kResultBits = #items that fit into a cacheline,
        // but there's also metadata to account for so divide by
        // items_per_cl again to get the same cache line as the bucket's items
        const Index mapped_bucket =
            (bucket * (clbits / kResultBits)) / items_per_cl;
        // now account for potentially more than one bucket per CL
        const Index cl = mapped_bucket / buckets_per_cl;
        meta_t meta_block;
        memcpy(&meta_block, data_.get() + cl * clsize, sizeof(meta_t));

        if constexpr (kThreshMode == ThreshMode::normal && buckets_per_cl == 1) {
            return meta_block;
        } else {
            const auto shift =
                meta_bits_per_bucket * (mapped_bucket & (buckets_per_cl - 1));
            const auto mask = (meta_item_t{1} << meta_bits_per_bucket) - 1;
            sLOG << "GetMeta" << bucket << "-> cl" << cl << " idx"
                 << cl * clsize << "shift" << shift << "mask" << mask;
            return static_cast<meta_item_t>(meta_block >> shift) & mask;
        }
    }

    inline void SetMeta(Index bucket, meta_item_t val) const {
        assert(bucket < GetNumBuckets());
        const Index mapped_bucket =
            (bucket * (clbits / kResultBits)) / items_per_cl;
        const Index cl = mapped_bucket / buckets_per_cl, idx = cl * clsize;
        meta_t* ptr = reinterpret_cast<meta_t*>(data_.get() + idx);

        if constexpr (kThreshMode == ThreshMode::normal && buckets_per_cl == 1) {
            *ptr = val;
        } else {
            const auto shift =
                meta_bits_per_bucket * (mapped_bucket & (buckets_per_cl - 1));
            const auto mask = (meta_item_t{1} << meta_bits_per_bucket) - 1;
            assert(val <= mask);
            sLOG << "SetMeta" << bucket << "val" << (int)val << "-> cl" << cl
                 << "idx" << idx << "shift" << shift << "mask" << mask;
            // first clear, then write
            *ptr &= ~static_cast<meta_t>(mask << shift);
            *ptr |= (static_cast<meta_t>(val) << shift);
        }
        assert(GetMeta(bucket) == val);
    }

    void PrefetchQuery(Index row) const {
        const Index cl = row / items_per_cl;
        const Index cl_start = cl * clsize;
        __builtin_prefetch(data_.get() + cl_start, /* rw */ 0, /* locality */ 1);
    }
    void PrefetchMeta(Index) const {
        // nothing to do, PrefetchQuery already does everything we need
    }

    inline ResultRow GetResult(Index row) const {
        const Index cl = row / items_per_cl;
        const Index row_in_cl = row - cl * items_per_cl;
        const Index offset = row_in_cl / items_per_row + metarows_per_cl;
        const Index data_row = cl * clsize + offset * sizeof(ResultRow);
        sLOG << "Get" << row << "cl" << cl << "offset" << offset << "in row"
             << data_row;

        const auto shift = (row_in_cl % items_per_row) * kResultBits;
        ResultRow result;
        memcpy(&result, data_.get() + data_row, sizeof(ResultRow));
        return (result >> shift) & maxval;
    }

    using State =
        std::conditional_t<items_per_row == 1, Index, std::pair<Index, Index>>;
    inline State PrepareGetResult(Index row) const {
        const Index cl = row / items_per_cl;
        const Index row_in_cl = row - cl * items_per_cl;
        const Index offset = row_in_cl / items_per_row + metarows_per_cl;
        const Index data_row = cl * clsize + offset * sizeof(ResultRow);
        if constexpr (items_per_row == 1) {
            sLOG << "Prep" << row << "cl" << cl << "offset" << offset
                 << "in row" << data_row;
            return data_row;
        } else {
            const auto shift = (row_in_cl % items_per_row) * kResultBits;
            sLOG << "Prep" << row << "cl" << cl << "offset" << offset
                 << "in row" << data_row << "/" << shift;
            return std::make_pair(data_row, shift);
        }
    }

    inline ResultRow GetFromState(const State& state) const {
        ResultRow result;
        if constexpr (items_per_row == 1) {
            memcpy(&result, data_.get() + state, sizeof(ResultRow));
            return result;
        } else {
            auto [data_row, shift] = state;
            memcpy(&result, data_.get() + data_row, sizeof(ResultRow));
            return (result >> shift) & maxval;
        }
    }

    inline State AdvanceState(State state) const {
        Index data_row;
        if constexpr (items_per_row != 1) {
            Index shift;
            std::tie(data_row, shift) = state;
            if (shift + kResultBits < 8u * sizeof(ResultRow)) {
                // same row, just shift more
                sLOG << "Adv" << state << "staying in row, new shift"
                     << shift + kResultBits;
                return std::make_pair(data_row, shift + kResultBits);
            }
        } else {
            data_row = state;
        }
        Index new_row = data_row + sizeof(ResultRow);
        // Skip metadata of next cache line
        if (new_row % clsize == 0) {
            new_row += metarows_per_cl * sizeof(ResultRow);
        }
        // new_row += metarows_per_cl * sizeof(ResultRow) * (new_row % clsize == 0);
        if constexpr (items_per_row == 1)
            return new_row;
        else
            return std::make_pair(new_row, 0);
    }

    /* FIXME: I'm not sure if this is correct yet, and it could
       be improved in some cases when a row has one of its ends
       exactly at a byte boundary */
    inline Index GetPrevSafeRowEnd(Index row) {
        Index offset = 8u / kResultBits + 1;
        return row > offset ? row - offset : 0;
    }

    inline Index GetNextSafeRowStart(Index row) {
        Index offset = 8u / kResultBits + 1;
        return row + offset;
    }

    inline void SetResult(Index row, ResultRow val) {
        const Index cl = row / items_per_cl;
        const Index row_in_cl = row - cl * items_per_cl;
        const Index offset = row_in_cl / items_per_row + metarows_per_cl;
        const Index data_row = cl * clsize + offset * sizeof(ResultRow);
        sLOG << "Set" << row << "to" << (int)val << "cl" << cl << "offset"
             << offset << "in row" << data_row;

        const auto shift = (row_in_cl % items_per_row) * kResultBits;
        ResultRow* ptr = reinterpret_cast<ResultRow*>(data_.get() + data_row);
        *ptr &= ~static_cast<ResultRow>(maxval << shift);
        *ptr |= (val << shift);
    }

    // clang-format off
    inline Index GetNumSlots() const { return num_slots_; }
    inline Index GetNumStarts() const { return num_slots_ - kCoeffBits + 1; }
    inline Index GetNumBuckets() const { return (GetNumStarts() + kBucketSize - 1) / kBucketSize; }
    // clang-format on


    size_t Size() const {
        assert(size_ == ((num_slots_ + items_per_cl - 1) / items_per_cl) * clsize);
        return size_ + 2 * sizeof(Index);
    }

protected:
    size_t size_;
    Index num_slots_;
    std::unique_ptr<unsigned char[]> data_;
};

} // namespace ribbon
