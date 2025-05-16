//  Copyright (c) Lorenz Hübschle-Schneider
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "backsubst.hpp"
#include "config.hpp"
#include "construction.hpp"
#include "hasher.hpp"
#include "permute.hpp"
#include "query.hpp"
#include "rocksdb/stop_watch.h"
#include "storage.hpp"
#include "thresh_compress.hpp"
#include "serialization.hpp"

#include <tlx/logger.hpp>
#include <tlx/define/likely.hpp>

#include <fstream>
#include <iostream>
#include <type_traits>

namespace ribbon {

namespace {

template <typename Config>
// Base level filter is always *at least* 64-bit
struct BaseConfig : Config {
    using CoeffRow = std::conditional_t<sizeof(typename Config::CoeffRow) < 8,
                                        uint64_t, typename Config::CoeffRow>;
    static constexpr typename Config::Index kBucketSize =
        recommended_bucket_size<8u * sizeof(CoeffRow), Config::kThreshMode>;
};

// Ribbon base class
template <typename Config>
class ribbon_base {
public:
    IMPORT_RIBBON_CONFIG(Config);
    using Hasher = ChooseThreshold<Config>;
    using mhc_or_key_t = typename HashTraits<Hasher>::mhc_or_key_t;

    ribbon_base() {}
    template <typename C = Config>
    ribbon_base(size_t num_slots, double slots_per_item, uint64_t seed,
                std::enable_if_t<!C::kUseMHC && !C::kUseVLR> * = 0) {
        static_assert(!C::kUseMHC && !C::kUseVLR);
        Init(num_slots, slots_per_item, seed);
    }

    template <typename C = Config>
    ribbon_base(size_t num_slots, double slots_per_item, uint64_t seed, uint32_t idx,
                std::enable_if_t<C::kUseMHC && !C::kUseVLR> * = 0) {
        static_assert(C::kUseMHC && !C::kUseVLR);
        Init(num_slots, slots_per_item, seed, idx);
    }

    template <typename C = Config>
    ribbon_base(double slots_per_item, uint64_t seed, Index num_ribbons,
                std::enable_if_t<!C::kUseMHC && C::kUseVLR> * = 0) {
        static_assert(!C::kUseMHC && C::kUseVLR);
        Init(0, slots_per_item, seed, num_ribbons);
    }

    template <typename C = Config>
    ribbon_base(double slots_per_item, uint64_t seed, uint32_t idx, Index num_ribbons,
                std::enable_if_t<C::kUseMHC && C::kUseVLR> * = 0) {
        static_assert(C::kUseMHC && C::kUseVLR);
        Init(0, slots_per_item, seed, idx, num_ribbons);
    }

    // Specialisation for Master Hash Code Hasher
    template <typename C = Config>
    std::enable_if_t<C::kUseMHC, void>
    Init(size_t num_slots, double slots_per_item, uint64_t seed, uint32_t idx, Index num_ribbons = 1) {
        sLOGC(Config::log) << "ribbon_base (MHC) called with" << num_slots
                           << "slots and" << slots_per_item << "slots per item";
        num_ribbons_ = num_ribbons;
        slots_per_item_ = slots_per_item;
        hasher_.Seed(seed, idx);
        if (num_slots > 0)
            Prepare(num_slots);
    }

    template <typename C = Config>
    std::enable_if_t<!C::kUseMHC, void>
    Init(size_t num_slots, double slots_per_item, uint64_t seed, Index num_ribbons = 1) {
        sLOGC(Config::log) << "ribbon_base called with" << num_slots
                           << "slots and" << slots_per_item << "slots per item";
        static_assert(!Config::kUseMHC, "wat");
        num_ribbons_ = num_ribbons;
        slots_per_item_ = slots_per_item;
        hasher_.Seed(seed);
        if (num_slots > 0)
            Prepare(num_slots);
    }

    void Realloc(size_t num_slots, double slots_per_item) {
        sLOGC(Config::log) << "Reallocating ribbon with a bigger size:" << num_slots
                           << "slots," << slots_per_item << "slots per item";
        slots_per_item_ = slots_per_item;
        Prepare(num_slots);
    }

    // filter queries don't make sense with variable length retrieval
    template <typename C = Config>
    inline std::enable_if_t<!C::kUseVLR, std::pair<bool, bool>>
    QueryFilter(const Key &key) const {
        if constexpr (!kIsFilter) {
            assert(false);
            return std::pair(true, false);
        }
        if constexpr (kUseInterleavedSol) {
            // it can happen that a filter is empty but still gets queries
            // because the prior filter would have bumped that key if it had
            // been inserted
            if (TLX_UNLIKELY(sol_.GetNumSlots() == 0))
                return std::make_pair(false, false);
            return InterleavedFilterQuery(key, hasher_, sol_);
        } else if constexpr (kUseCacheLineStorage) {
            if (TLX_UNLIKELY(sol_.GetNumSlots() == 0))
                return std::make_pair(false, false);
            return SimpleFilterQuery(key, hasher_, sol_);
        } else {
            if (TLX_UNLIKELY(storage_.GetNumSlots() == 0))
                return std::make_pair(false, false);
            return SimpleFilterQuery(key, hasher_, storage_);
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC && !C::kUseVLR, std::pair<bool, bool>>
    QueryFilterMHC(const mhc_or_key_t mhc) const {
        if constexpr (!kIsFilter) {
            assert(false);
            return std::pair(true, false);
        }
        if constexpr (kUseInterleavedSol) {
            // it can happen that a filter is empty but still gets queries
            // because the prior filter would have bumped that key if it had
            // been inserted
            if (TLX_UNLIKELY(sol_.GetNumSlots() == 0))
                return std::make_pair(false, false);
            return InterleavedFilterQuery(mhc, hasher_, sol_);
        } else if constexpr (kUseCacheLineStorage) {
            if (TLX_UNLIKELY(sol_.GetNumSlots() == 0))
                return std::make_pair(false, false);
            return SimpleFilterQuery(mhc, hasher_, sol_);
        } else {
            if (TLX_UNLIKELY(storage_.GetNumSlots() == 0))
                return std::make_pair(false, false);
            return SimpleFilterQuery(mhc, hasher_, storage_);
        }
    }

    template <typename Iterator, typename C = Config>
    std::enable_if_t<!C::kUseVLR || C::kVLRShareMeta, bool>
    AddRange(
        Iterator begin, Iterator end,
        std::vector<std::conditional_t<
            kUseMHC, std::conditional_t<kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, std::conditional_t<kUseVLR, ResultRowVLR, ResultRow>>>,
            typename std::iterator_traits<Iterator>::value_type>> *bump_vec) {
        const auto input_size = end - begin;

        if constexpr (kUseVLR)
            CalcRibbonSizesVLR(begin, end);

        LOGC(Config::log) << "Constructing for " << storage_.GetNumSlots()
                          << " slots, " << storage_.GetNumStarts() << " starts, "
                          << storage_.GetNumBuckets() << " buckets";

        rocksdb::StopWatchNano timer(true);
        bool success;
        if constexpr (kUseMHC) {
            if constexpr (kUseVLR && kVLRShareMeta)
                success = BandingAddRangeMHC(&storage_, hasher_, begin, end, bump_vec, num_ribbons_);
            else
                success = BandingAddRangeMHC(&storage_, hasher_, begin, end, bump_vec);
        } else {
            if constexpr (kUseVLR && kVLRShareMeta)
                success = BandingAddRange(&storage_, hasher_, begin, end, bump_vec, num_ribbons_);
            else
                success = BandingAddRange(&storage_, hasher_, begin, end, bump_vec);
        }
        LOGC(Config::log) << "Insertion of " << input_size << " items took "
                          << timer.ElapsedNanos(true) / 1e6 << "ms";

        if (bump_vec == nullptr)
            num_bumped = success ? 0 : input_size;
        else
            num_bumped = bump_vec->size();
        empty_slots = static_cast<ssize_t>(storage_.GetNumSlots()) -
                      input_size + num_bumped;
        sLOGC(Config::log) << "Bumped" << num_bumped << "out of" << input_size
                           << "=" << (num_bumped * 100.0 / input_size)
                           << "% with" << slots_per_item_ << "slots per item =>"
                           << empty_slots << "empty slots ="
                           << empty_slots * 100.0 / storage_.GetNumSlots() << "%";
        return success;
    }

    template <typename Iterator, typename C = Config>
    std::enable_if_t<C::kUseVLR && !C::kVLRShareMeta, bool>
    AddRange(
        Iterator begin, Iterator end,
        std::vector<std::conditional_t<
            kUseMHC, std::tuple<mhc_or_key_t, ResultRowVLR, ResultRowVLR>,
            std::tuple<Key, ResultRowVLR, ResultRowVLR>>> *bump_vec) {

        const size_t input_size = end - begin;
        CalcRibbonSizesVLR(begin, end);
        LOGC(Config::log) << "Constructing for " << storage_.GetNumSlots()
                          << " slots, " << storage_.GetNumStarts() << " starts, "
                          << storage_.GetNumBuckets() << " buckets";
        rocksdb::StopWatchNano timer(true);
        bool success;
        if constexpr (kUseMHC) {
            success = BandingAddRangeMHCVLR(&storage_, hasher_, begin, end, bump_vec, num_ribbons_);
        } else {
            success = BandingAddRangeVLR(&storage_, hasher_, begin, end, bump_vec, num_ribbons_);
        }
        LOGC(Config::log) << "Insertion of " << input_size << " items took "
                          << timer.ElapsedNanos(true) / 1e6 << "ms";

        if (bump_vec == nullptr)
            num_bumped = success ? 0 : input_size;
        else
            num_bumped = bump_vec->size();
        empty_slots = static_cast<ssize_t>(storage_.GetNumSlots()) -
                      input_size + num_bumped;
        // FIXME: There isn't really any good way to calculate empty_slots in the VLR version
        // (the only way is to count the ones in all bumped items)
        sLOGC(Config::log) << "Bumped" << num_bumped << "out of" << input_size
                           << "=" << (num_bumped * 100.0 / input_size)
                           << "% with" << slots_per_item_ << "slots per item =>"
                           << empty_slots << "empty slots ="
                           << empty_slots * 100.0 / storage_.GetNumSlots() << "%";
        return success;
    }

    template <typename C = Config>
    std::enable_if_t<!C::kUseVLR, void>
    BackSubst() {
        if (storage_.GetNumSlots() == 0)
            return;
        rocksdb::StopWatchNano timer(true);
        if constexpr (kUseInterleavedSol) {
            InterleavedBackSubst(storage_, &sol_);
            // move metadata by swapping pointers
            sol_.MoveMetadata(&storage_);
            storage_.Reset();
        } else if constexpr (kUseCacheLineStorage) {
            sol_.Prepare(storage_.GetNumSlots());
            SimpleBackSubst(storage_, &sol_);
            // copy metadata one-by-one
            for (Index bucket = 0; bucket < storage_.GetNumBuckets(); ++bucket) {
                sol_.SetMeta(bucket, storage_.GetMeta(bucket));
            }
            storage_.Reset();
        } else {
            SimpleBackSubst(storage_, &storage_);
        }

        LOGC(Config::log) << "Backsubstitution for " << storage_.GetNumSlots()
                          << " slots took " << timer.ElapsedNanos(true) / 1e6
                          << "ms";
    }

    template <typename C = Config>
    std::enable_if_t<C::kUseVLR, void>
    BackSubst() {
        if (storage_.GetNumSlots() == 0)
            return;
        rocksdb::StopWatchNano timer(true);
        if constexpr (kUseInterleavedSol) {
            InterleavedBackSubst(storage_, &sol_, num_ribbons_);
            // move metadata by swapping pointers
            sol_.MoveMetadata(&storage_);
            storage_.Reset();
        } else if constexpr (kUseCacheLineStorage) {
            static_assert(!kUseCacheLineStorage); // not supported
            sol_.Prepare(storage_.GetNumSlots(), num_ribbons_);
            SimpleBackSubst(storage_, &sol_, num_ribbons_);
            // copy metadata one-by-one
            for (Index bucket = 0; bucket < storage_.GetNumBuckets(); ++bucket) {
                if constexpr (kVLRShareMeta) {
                    sol_.SetMeta(bucket, storage_.GetMeta(bucket));
                } else {
                    for (Index i = 0; i < num_ribbons_; ++i) {
                        sol_.SetMeta(bucket, storage_.GetMeta(bucket, i), i);
                    }
                }
            }
            storage_.Reset();
        } else {
            SimpleBackSubst(storage_, &storage_, num_ribbons_);
        }

        LOGC(Config::log) << "Backsubstitution for " << storage_.GetNumSlots()
                          << " slots, " << num_ribbons_ << " ribbons took "
                          << timer.ElapsedNanos(true) / 1e6 << "ms";
    }

    // start_idx and num_bits only supported in VLR mode
    template <typename C = Config>
    inline std::enable_if_t<(!C::kUseVLR || C::kVLRShareMeta) && !C::kUseMHC, std::pair<bool, std::conditional_t<kUseVLR, ResultRowVLR, ResultRow>>>
    QueryRetrieval(const Key &key, [[maybe_unused]] Index start_idx = 0, [[maybe_unused]] Index num_bits = 0) const {
        if constexpr (kIsFilter || kUseMHC) {
            assert(false);
            return std::make_pair(true, 0);
        }
        if constexpr (kUseInterleavedSol) {
            if constexpr (kUseVLR)
                return InterleavedRetrievalQuery(key, hasher_, sol_, start_idx, num_bits);
            else
                return InterleavedRetrievalQuery(key, hasher_, sol_);
        } else if constexpr (kUseCacheLineStorage) {
            if constexpr (kUseVLR)
                assert(false);
            return SimpleRetrievalQuery(key, hasher_, sol_);
        } else {
            if constexpr (kUseVLR)
                return SimpleRetrievalQuery(key, hasher_, storage_, start_idx, num_bits);
            else
                return SimpleRetrievalQuery(key, hasher_, storage_);
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<(!C::kUseVLR || C::kVLRShareMeta) && C::kUseMHC, std::pair<bool, std::conditional_t<kUseVLR, ResultRowVLR, ResultRow>>>
    QueryRetrievalMHC(const mhc_or_key_t mhc, [[maybe_unused]] Index start_idx = 0, [[maybe_unused]] Index num_bits = 0) const {
        if constexpr (kIsFilter || !kUseMHC) {
            assert(false);
            return std::make_pair(true, 0);
        }
        if constexpr (kUseInterleavedSol) {
            if constexpr (kUseVLR)
                return InterleavedRetrievalQuery(mhc, hasher_, sol_, start_idx, num_bits);
            else
                return InterleavedRetrievalQuery(mhc, hasher_, sol_);
        } else if constexpr (kUseCacheLineStorage) {
            if constexpr (kUseVLR)
                assert(false);
            return SimpleRetrievalQuery(mhc, hasher_, sol_);
        } else {
            if constexpr (kUseVLR)
                return SimpleRetrievalQuery(mhc, hasher_, storage_, start_idx, num_bits);
            else
                return SimpleRetrievalQuery(mhc, hasher_, storage_);
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseVLR && !kVLRShareMeta && !C::kUseMHC, std::pair<ResultRowVLR, ResultRowVLR>>
    QueryRetrieval(const Key &key, ResultRowVLR bump_mask) const {
        static_assert(!kIsFilter && !kUseMHC);
        if constexpr (kUseInterleavedSol) {
            return InterleavedRetrievalQueryVLR(key, hasher_, sol_, bump_mask);
        } else if constexpr (kUseCacheLineStorage) {
            assert(false);
        } else {
            return SimpleRetrievalQueryVLR(key, hasher_, storage_, bump_mask);
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseVLR && !kVLRShareMeta && C::kUseMHC, std::pair<ResultRowVLR, ResultRowVLR>>
    QueryRetrievalMHC(const mhc_or_key_t mhc, ResultRowVLR bump_mask) const {
        static_assert(!kIsFilter && kUseMHC);
        if constexpr (kUseInterleavedSol) {
            return InterleavedRetrievalQueryVLR(mhc, hasher_, sol_, bump_mask);
        } else if constexpr (kUseCacheLineStorage) {
            assert(false);
        } else {
            return SimpleRetrievalQueryVLR(mhc, hasher_, storage_, bump_mask);
        }
    }

    std::tuple<ssize_t, ssize_t, double, ssize_t> GetStats() const {
        double frac_empty;
        if constexpr (kUseVLR) {
            frac_empty = static_cast<double>(empty_slots) / (storage_.GetNumSlots() * num_ribbons_);
        } else {
            frac_empty = static_cast<double>(empty_slots) / storage_.GetNumSlots();
        }
        ssize_t thresh_bytes = 0;
        if constexpr (kThreshMode == ThreshMode::onebit) {
            thresh_bytes = hasher_.Size();
        }
        return std::make_tuple(num_bumped, empty_slots, frac_empty, thresh_bytes);
    }

    size_t Size() const {
        size_t sol_size, num_slots;
        if constexpr (kUseInterleavedSol || kUseCacheLineStorage) {
            sol_size = sol_.Size();
            if constexpr (kUseVLR)
                num_slots = sol_.GetNumSlots() * num_ribbons_;
            else
                num_slots = sol_.GetNumSlots();
        } else {
            sol_size = storage_.Size();
            if constexpr (kUseVLR)
                num_slots = storage_.GetNumSlots() * num_ribbons_;
            else
                num_slots = storage_.GetNumSlots();
        }

        size_t extra_bytes = 0, extra_count = 0;
        if constexpr (kThreshMode == ThreshMode::onebit) {
            extra_bytes = hasher_.Size();
            extra_count = hasher_.NumEntries();
        }
        sLOGC(Config::log) << "Size:" << sol_size << "Bytes for" << num_slots
                           << "slots +" << extra_bytes << "Bytes for"
                           << extra_count << "thresholds";
        return sol_size + extra_bytes;
    }

    /*
    void PrintStats() const {
        if constexpr (Config::kThreshMode == ThreshMode::onebit) {
            sLOG1 << hasher_.hits << "of" << hasher_.count
                  << "filter hits =" << hasher_.hits * 100.0 / hasher_.count
                  << "%, size" << hasher_.filter_.size() << "with"
                  << hasher_.buffer_.size() << "potential 1s ="
                  << hasher_.buffer_.size() * 100.0 / hasher_.filter_.size()
                  << "%";
            hasher_.hits = 0;
            hasher_.count = 0;
        }
    }
    */

protected:
    template <typename C = Config>
    static constexpr inline std::enable_if_t<C::kUseVLR && !C::kVLRShareMeta, ResultRowVLR>
    GetVLRMask(Index start_idx, Index num_bits) {
        if (num_bits == 0) {
            return ~ResultRowVLR(0);
        } else {
            if constexpr (kVLRFlipOutputBits)
                return ((~ResultRowVLR(0)) >> (sizeof(ResultRowVLR) * 8U - num_bits)) << start_idx;
            else
                return ((~ResultRowVLR(0)) >> (sizeof(ResultRowVLR) * 8U - num_bits)) << (sizeof(ResultRowVLR) * 8U - num_bits - start_idx);
        }
    }

    template <typename Iterator, typename C = Config>
    std::enable_if_t<C::kUseVLR, void>
    CalcRibbonSizesVLR(Iterator begin, Iterator end) {
        rocksdb::StopWatchNano timer(true);
        const size_t input_size = end - begin;
        std::vector<size_t> ribbon_sizes(num_ribbons_);
        // this part is mainly copied from construction.hpp
        for (size_t i = 0; i < input_size; ++i) {
            Index ribbon_start;
            if constexpr (kUseMHC) {
                const auto mhc = *(begin + i);
                const auto hash = hasher_.GetHash(mhc);
                ribbon_start = hasher_.GetVLRIndex(hash, num_ribbons_);
            } else {
                const Hash h = hasher_.GetHash(*(begin + i));
#ifdef RIBBON_PASS_HASH
                constexpr bool sparse = Hasher::kSparseCoeffs && Hasher::kCoeffBits < 128;
#else
                constexpr bool sparse = false;
#endif
                if constexpr (sparse) {
                    uint32_t compact_hash = hasher_.GetCompactHash(h);
                    ribbon_start = hasher_.GetVLRIndex(compact_hash, num_ribbons_);
                } else {
                    ribbon_start = hasher_.GetVLRIndex(h, num_ribbons_);
                }
            }
            const ResultRowVLR rr = hasher_.GetResultRowVLRFromInput(*(begin + i));
            // there must be at least a 1 to mark the beginning of the actual value
            assert(rr != 0);
            // the first 1 is not part of the actual value
            const int num_zeroes = rocksdb::CountLeadingZeroBits(rr) + 1;
            const int num_bits = (sizeof(ResultRowVLR) * 8) - num_zeroes;
            assert(num_bits != 0);
            // In toplevel (or if kVLRShareMeta is set), there is no bump mask.
            constexpr bool toplevel = kVLRShareMeta ||
                                      std::is_same_v<typename std::iterator_traits<Iterator>::value_type, std::pair<mhc_or_key_t, ResultRowVLR>>;
            ResultRowVLR prev_bumped;
            if constexpr (toplevel)
                prev_bumped = ~ResultRowVLR(0);
            else
                prev_bumped = std::get<2>(*(begin + i));

            for (int bit = 0; bit < num_bits; ++bit) {
                const int shift = num_bits - bit - 1;
                // if the element wasn't bumped previously, it doesn't need to be inserted
                if (((prev_bumped >> shift) & 0x1))
                    ribbon_sizes[(ribbon_start + bit) % num_ribbons_]++;
            }
        }
        size_t num_slots = *std::max_element(ribbon_sizes.begin(), ribbon_sizes.end());
        num_slots = std::max(size_t{1}, static_cast<size_t>(slots_per_item_ * num_slots));
        LOGC(Config::log) << "Calculating max ribbon size for " << input_size << " items took "
                          << timer.ElapsedNanos(true) / 1e6 << "ms";
        LOGC(Config::log) << "Total number of bits: " << std::accumulate(ribbon_sizes.begin(), ribbon_sizes.end(), 0) << "\n";
        Prepare(num_slots);
        LOGC(Config::log) << "Allocation for " << input_size << " items took "
                          << timer.ElapsedNanos(true) / 1e6 << "ms";
    }

    void Prepare(size_t num_slots) {
        if (num_slots == 0)
            return;
        // round up to next multiple of kCoeffBits for interleaved storage
        if constexpr (kUseInterleavedSol)
            num_slots = ((num_slots + kCoeffBits - 1) / kCoeffBits) * kCoeffBits;
        else
            num_slots = std::max(num_slots, static_cast<size_t>(kCoeffBits));
        if constexpr (kUseVLR) {
            storage_.Prepare(num_slots, num_ribbons_);
            if constexpr (kVLRShareMeta)
                hasher_.Prepare(num_slots, slots_per_item_);
            else
                hasher_.Prepare(num_slots, slots_per_item_, num_ribbons_);
        } else {
            storage_.Prepare(num_slots);
            hasher_.Prepare(num_slots, slots_per_item_);
        }
    }

    template <typename Iterator, typename C = Config>
    std::enable_if_t<C::kUseMHC,
                     std::unique_ptr<std::conditional_t<kUseVLR,
                         std::pair<mhc_or_key_t, ResultRowVLR>, std::conditional_t<
                         kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, ResultRow>>>[]>>
    PrepareAddRangeMHC(Iterator begin, Iterator end) {
        // this is the top-level filter, transform from vector<Key> to vector<mhc_t>
        rocksdb::StopWatchNano timer(true);
        const auto input_size = end - begin;
        using value_type = std::conditional_t<kUseVLR, std::pair<mhc_or_key_t, ResultRowVLR>,
                 std::conditional_t<kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, ResultRow>>>;
        auto input = std::make_unique<value_type[]>(input_size);
        auto it = begin;
        for (ssize_t i = 0; i < input_size; i++, ++it) {
            if constexpr (kIsFilter)
                input[i] = hasher_.GetMHC(*it);
            else
                input[i] = std::make_pair(hasher_.GetMHC(it->first), it->second);
        }
        LOGC(Config::log) << "MHC input transformation took "
                          << timer.ElapsedNanos() / 1e6 << "ms";
        return input;
    }

    // these (De)SerializeIntern methods are called on each level, while the
    // (De)Serialize methods below are only called on the top level
    virtual void SerializeIntern(std::ostream &os) {
        if constexpr (kUseCacheLineStorage) {
            throw config_error("kUseCacheLineStorage not supported");
        } else if constexpr (kUseInterleavedSol) {
            sol_.SerializeIntern(os);
        } else {
            storage_.SerializeIntern(os);
        }
        hasher_.SerializeIntern(os);
    }

    virtual void DeserializeIntern(std::istream &is, bool switchendian, uint64_t seed, uint32_t idx, Index num_ribbons = 1) {
        num_ribbons_ = num_ribbons;
        Index num_buckets = 0;
        if constexpr (kUseCacheLineStorage) {
            throw config_error("kUseCacheLineStorage not supported");
        } else if constexpr (kUseInterleavedSol) {
            sol_.DeserializeIntern(is, switchendian, num_ribbons);
            num_buckets = sol_.GetNumBuckets();
        } else {
            storage_.DeserializeIntern(is, switchendian, num_ribbons);
            num_buckets = storage_.GetNumBuckets();
        }
        if constexpr (kUseVLR && !kVLRShareMeta)
            hasher_.DeserializeIntern(is, switchendian, num_buckets, num_ribbons_);
        else
            hasher_.DeserializeIntern(is, switchendian, num_buckets);
        if constexpr (Config::kUseMHC)
            hasher_.Seed(seed, idx);
        else
            hasher_.Seed(seed);
    }

    // these methods are in ribbon_base so they can still be called by
    // the base case ribbon
    void Serialize(std::ostream &os, uint8_t depth) {
        auto mask = os.exceptions();
        os.exceptions(~std::ios::goodbit);

        if constexpr (kUseCacheLineStorage)
            throw config_error("kUseCacheLineStorage not supported");

        os.write("BuRR", 4);
        uint16_t bom = 0xFEFF;
        os.write(reinterpret_cast<const char *>(&bom), sizeof(uint16_t));
        uint16_t version = 0;
        os.write(reinterpret_cast<const char *>(&version), sizeof(uint16_t));

        char tmp = sizeof(CoeffRow);
        os.write(&tmp, 1);
        tmp = sizeof(ResultRow);
        os.write(&tmp, 1);
        tmp = kResultBits;
        os.write(&tmp, 1);
        tmp = sizeof(Index);
        os.write(&tmp, 1);
        os.write(reinterpret_cast<const char *>(&kBucketSize), sizeof(Index));
        tmp = sizeof(Hash);
        os.write(&tmp, 1);
        tmp = static_cast<int>(kThreshMode);
        os.write(&tmp, 1);
        // this needs to be unsigned so we don't mess with the sign bit
        unsigned char bits = kUseMultiplyShiftHash | (kIsFilter << 1) |
              (kFirstCoeffAlwaysOne << 2) | (kSparseCoeffs << 3) |
              (kUseInterleavedSol << 4) | (kUseMHC << 5) |
              (kUseVLR << 6) | (kVLRShareMeta << 7);
        os.write(reinterpret_cast<const char *>(&bits), 1);
        if constexpr (kUseVLR) {
            tmp = sizeof(ResultRowVLR);
            os.write(&tmp, 1);
            os.write(reinterpret_cast<const char *>(&num_ribbons_), sizeof(Index));
        }

        uint8_t d = depth;
        os.write(reinterpret_cast<const char *>(&d), sizeof(uint8_t));
        uint64_t seed = this->hasher_.GetSeed();
        os.write(reinterpret_cast<const char *>(&seed), sizeof(uint64_t));

        // this has to call the overridden method in the child class so
        // all levels of the data structure are serialized
        SerializeIntern(os);

        // set exception mask to original mask again
        os.exceptions(mask);
    }

    // FIXME: should all internal structures be reset on deserialize?
    void Deserialize(std::istream &is, uint8_t depth) {
        auto mask = is.exceptions();
        is.exceptions(~std::ios::goodbit);

        if constexpr (kUseCacheLineStorage)
            throw config_error("kUseCacheLineStorage not supported");

        char magic[4];
        is.read(magic, 4);
        if (strncmp(magic, "BuRR", 4))
            throw parse_error("wrong magic number");
        uint16_t bom;
        is.read(reinterpret_cast<char *>(&bom), sizeof(uint16_t));
        bool switchendian = false;
        if (bom == 0xFFFE)
            switchendian = true;
        else if (bom != 0xFEFF)
            throw parse_error("invalid endianness specification");
        uint16_t version = 0;
        is.read(reinterpret_cast<char *>(&version), sizeof(uint16_t));
        if (version != 0)
            throw parse_error("invalid version number");

        char tmp;
        is.read(&tmp, 1);
        if (tmp != sizeof(CoeffRow))
            throw config_error("sizeof(CoeffRow) mismatch");
        is.read(&tmp, 1);
        if (tmp != sizeof(ResultRow))
            throw config_error("sizeof(ResultRow) mismatch");
        is.read(&tmp, 1);
        if (tmp != kResultBits)
            throw config_error("kResultBits mismatch");
        is.read(&tmp, 1);
        if (tmp != sizeof(Index))
            throw config_error("sizeof(Index) mismatch");
        Index bucketsz;
        is.read(reinterpret_cast<char *>(&bucketsz), sizeof(Index));
        if (switchendian && !bswap_generic(bucketsz))
            throw parse_error("error converting endianness of kBucketSize");
        else if (bucketsz != kBucketSize)
            throw config_error("kBucketSize mismatch");
        is.read(&tmp, 1);
        if (tmp != sizeof(Hash))
            throw config_error("sizeof(Hash) mismatch");
        is.read(&tmp, 1);
        if (tmp != static_cast<int>(kThreshMode))
            throw config_error("kThreshMode mismatch");
        // this needs to be unsigned since we use all bits, including the
        // sign bit if it was signed
        unsigned char bits;
        is.read(reinterpret_cast<char *>(&bits), 1);
        if ((bits & 0x1) != kUseMultiplyShiftHash)
            throw config_error("kUseMultiplyShiftHash mismatch");
        else if (((bits >> 1) & 0x1) != kIsFilter)
            throw config_error("kIsFilter mismatch");
        else if (((bits >> 2) & 0x1) != kFirstCoeffAlwaysOne)
            throw config_error("kFirstCoeffAlwaysOne mismatch");
        else if (((bits >> 3) & 0x1) != kSparseCoeffs)
            throw config_error("kSparseCoeffs mismatch");
        else if (((bits >> 4) & 0x1) != kUseInterleavedSol)
            throw config_error("kUseInterleavedSol mismatch");
        else if (((bits >> 5) & 0x1) != kUseMHC)
            throw config_error("kUseMHC mismatch");
        else if (((bits >> 6) & 0x1) != kUseVLR)
            throw config_error("kUseVLR mismatch");
        else if (((bits >> 6) & 0x1) != kVLRShareMeta)
            throw config_error("kVLRShareMeta mismatch");
        if constexpr (kUseVLR) {
            is.read(&tmp, 1);
            if (tmp != sizeof(ResultRowVLR))
                throw config_error("sizeof(ResultRowVLR) mismatch");
            is.read(reinterpret_cast<char *>(&num_ribbons_), sizeof(Index));
            if (switchendian && !bswap_generic(num_ribbons_))
                throw parse_error("error converting endianness of num_ribbons");
        }

        uint8_t d;
        is.read(reinterpret_cast<char *>(&d), sizeof(uint8_t));
        if (d != depth)
            throw config_error("depth mismatch");
        uint64_t seed;
        is.read(reinterpret_cast<char *>(&seed), sizeof(uint64_t));
        if (switchendian && !bswap_generic(seed))
            throw parse_error("error converting endianness");

        // this has to call the overridden method in the child class so
        // all levels of the data structure are deserialized
        DeserializeIntern(is, switchendian, seed, 0, num_ribbons_);

        // reset exception mask
        is.exceptions(mask);
    }

    // convenience methods
    void Serialize(const std::string &filename, uint8_t depth) {
        std::ofstream os(filename, std::ios::binary|std::ios::out|std::ios::trunc);
        if (!os.is_open())
            throw file_open_error("unable to open file " + filename + " for writing");
        Serialize(os, depth);
    }

    void Deserialize(const std::string &filename, uint8_t depth) {
        std::ifstream is(filename, std::ios::binary|std::ios::in);
        if (!is.is_open())
            throw file_open_error("unable to open file " + filename + " for reading");
        Deserialize(is, depth);
    }

    // statistics
    ssize_t num_bumped = 0;
    ssize_t empty_slots = 0;

    // actual data
    double slots_per_item_;
    Index num_ribbons_ = 1;
    BasicStorage<Config> storage_;

    template <bool /* cls */, bool /* int */, typename C>
    struct sol_t {
        // dummy
        using type = sol_t<false, false, C>;
    };

    template <typename C>
    struct sol_t<true, false, C> {
        using type = CacheLineStorage<C>;
    };
    template <typename C>
    struct sol_t<false, true, C> {
        using type = InterleavedSolutionStorage<C>;
    };

    typename sol_t<kUseCacheLineStorage, kUseInterleavedSol, Config>::type sol_;
    Hasher hasher_;
};

} // namespace

template <uint8_t depth, typename Config>
class ribbon_filter : public ribbon_base<Config> {
public:
    IMPORT_RIBBON_CONFIG(Config);
    using Super = ribbon_base<Config>;
    using Super::slots_per_item_;
    using mhc_or_key_t = typename Super::mhc_or_key_t;

    ribbon_filter() = default;

    // TODO: maybe only make versions with num_ribbons available when !kUseVLR?
    ribbon_filter(std::istream &is) {
        Deserialize(is);
    }

    ribbon_filter(const std::string &filename) {
        Deserialize(filename);
    }

    // MHC top-level constructor
    template <typename C = Config>
    ribbon_filter(size_t num_slots, double slots_per_item, uint64_t seed,
                  typename std::enable_if<C::kUseMHC && !C::kUseVLR>::type * = 0)
        : Super(num_slots, slots_per_item, seed, 0),
          child_ribbon_(0, slots_per_item, seed, 1) {}

    // MHC VLR top-level constructor
    template <typename C = Config>
    ribbon_filter(double slots_per_item, uint64_t seed, Index num_ribbons,
                  typename std::enable_if<C::kUseMHC && C::kUseVLR>::type * = 0)
        : Super(slots_per_item, seed, 0, num_ribbons),
          child_ribbon_(slots_per_item, seed, 1, num_ribbons) {}

protected:
    // MHC child constructor
    template <typename C = Config>
    ribbon_filter(size_t num_slots, double slots_per_item, uint64_t seed, uint32_t idx,
                  typename std::enable_if<C::kUseMHC && !C::kUseVLR>::type * = 0)
        : Super(num_slots, slots_per_item, seed, idx),
          child_ribbon_(0, slots_per_item, seed, idx + 1) {}

    // MHC VLR child constructor
    template <typename C = Config>
    ribbon_filter(double slots_per_item, uint64_t seed, uint32_t idx, Index num_ribbons,
                  typename std::enable_if<C::kUseMHC && C::kUseVLR>::type * = 0)
        : Super(slots_per_item, seed, idx, num_ribbons),
          child_ribbon_(slots_per_item, seed, idx + 1, num_ribbons) {}

public:
    // non-MHC top-level constructor
    template <typename C = Config>
    ribbon_filter(size_t num_slots, double slots_per_item, uint64_t seed,
                  typename std::enable_if<!C::kUseMHC && !C::kUseVLR>::type * = 0)
        : Super(num_slots, slots_per_item, seed),
          child_ribbon_(0, slots_per_item, seed + 1) {}

    // non-MHC VLR top-level constructor
    template <typename C = Config>
    ribbon_filter(double slots_per_item, uint64_t seed, Index num_ribbons,
                  typename std::enable_if<!C::kUseMHC && C::kUseVLR>::type * = 0)
        : Super(slots_per_item, seed, num_ribbons),
          child_ribbon_(slots_per_item, seed + 1, num_ribbons) {}

    // non-MHC init
    template <typename C = Config>
    std::enable_if_t<!C::kUseMHC && !C::kUseVLR, void>
    Init(size_t num_slots, double slots_per_item, uint64_t seed) {
        Super::Init(num_slots, slots_per_item, seed);
        child_ribbon_.Init(0, slots_per_item, seed + 1);
    }

    // non-MHC VLR init
    template <typename C = Config>
    std::enable_if_t<!C::kUseMHC && C::kUseVLR, void>
    Init(double slots_per_item, uint64_t seed, Index num_ribbons) {
        Super::Init(slots_per_item, seed, num_ribbons);
        child_ribbon_.Init(slots_per_item, seed + 1, num_ribbons);
    }

    // MHC init
    template <typename C = Config>
    std::enable_if_t<C::kUseMHC && !C::kUseVLR, void>
    Init(size_t num_slots, double slots_per_item, uint64_t seed, uint32_t idx) {
        Super::Init(num_slots, slots_per_item, seed, idx);
        child_ribbon_.Init(0, slots_per_item, seed, idx + 1);
    }

    // MHC VLR init
    template <typename C = Config>
    std::enable_if_t<C::kUseMHC && C::kUseVLR, void>
    Init(double slots_per_item, uint64_t seed, uint32_t idx, Index num_ribbons) {
        Super::Init(slots_per_item, seed, idx, num_ribbons);
        child_ribbon_.Init(slots_per_item, seed, idx + 1, num_ribbons);
    }

    // Just a safety measure so the function can't be called with an illegal third argument
    template <typename Iterator, typename C = Config>
    std::enable_if_t<!C::kUseMHC, bool> AddRange(Iterator begin, Iterator end) {
        return AddRangeInternal(begin, end);
    }

    // MHC version
    template <typename Iterator, typename C = Config>
    std::enable_if_t<C::kUseMHC, bool> AddRange(Iterator begin, Iterator end) {
        auto input = Super::PrepareAddRangeMHC(begin, end);
        return AddRangeMHCInternal(input.get(), input.get() + (end - begin));
    }

    void BackSubst() {
        Super::BackSubst();
        child_ribbon_.BackSubst();
    }

    template <typename C = Config>
    inline std::enable_if_t<!C::kUseVLR, bool>
    QueryFilter(const Key &key) const {
        if constexpr (kUseMHC) {
            const auto mhc = this->hasher_.GetMHC(key);
            return QueryFilterMHC(mhc);
        } else {
            auto [was_bumped, found] = Super::QueryFilter(key);
            if (was_bumped) {
                return child_ribbon_.QueryFilter(key);
            } else {
                return found;
            }
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC && !C::kUseVLR, bool>
    QueryFilterMHC(const mhc_or_key_t mhc) const {
        auto [was_bumped, found] = Super::QueryFilterMHC(mhc);
        if (was_bumped)
            return child_ribbon_.QueryFilterMHC(mhc);
        else
            return found;
    }

    inline std::conditional_t<kUseVLR, ResultRowVLR, ResultRow> QueryRetrieval(const Key &key, [[maybe_unused]] Index start_idx = 0, [[maybe_unused]] Index num_bits = 0) const {
        if constexpr (kUseMHC) {
            const auto mhc = this->hasher_.GetMHC(key);
            return QueryRetrievalMHC(mhc, start_idx, num_bits);
        } else if constexpr (kUseVLR && !kVLRShareMeta) {
            return QueryRetrievalVLR(key, Super::GetVLRMask(start_idx, num_bits));
        } else {
            auto [was_bumped, result] = Super::QueryRetrieval(key, start_idx, num_bits);
            if (was_bumped) {
                return child_ribbon_.QueryRetrieval(key, start_idx, num_bits);
            } else {
                return result;
            }
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseVLR && !C::kVLRShareMeta, ResultRowVLR>
    QueryRetrieval(const Key &key, ResultRowVLR mask) const {
        if constexpr (kUseMHC) {
            const auto mhc = this->hasher_.GetMHC(key);
            return QueryRetrievalMHCVLR(mhc, mask);
        } else {
            return QueryRetrievalVLR(key, mask);
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC && C::kUseVLR, ResultRowVLR>
    QueryRetrievalMHC(const mhc_or_key_t mhc, ResultRowVLR mask) const {
        return QueryRetrievalMHCVLR(mhc, mask);
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC, std::conditional_t<kUseVLR, ResultRowVLR, ResultRow>>
    QueryRetrievalMHC(const mhc_or_key_t mhc, [[maybe_unused]] Index start_idx = 0, [[maybe_unused]] Index num_bits = 0) const {
        if constexpr (kUseVLR && !kVLRShareMeta) {
            return QueryRetrievalMHCVLR(mhc, Super::GetVLRMask(start_idx, num_bits));
        } else {
            auto [was_bumped, result] = Super::QueryRetrievalMHC(mhc, start_idx, num_bits);
            if (was_bumped)
                return child_ribbon_.QueryRetrievalMHC(mhc, start_idx, num_bits);
            else
                return result;
        }
    }

    size_t Size() const {
        return Super::Size() + child_ribbon_.Size();
    }

    void Serialize(const std::string &filename) {
        Super::Serialize(filename, depth);
    }

    void Deserialize(const std::string &filename) {
        Super::Deserialize(filename, depth);
    }

    void Serialize(std::ostream &os) {
        Super::Serialize(os, depth);
    }

    void Deserialize(std::istream &is) {
        Super::Deserialize(is, depth);
    }

    /*
    void PrintStats() const {
        Super::PrintStats();
        child_ribbon_.PrintStats();
    }
    */

protected:
    // the VLR versions need to be separate from the non-VLR versions so
    // the bump mask can be passed down
    template <typename C = Config>
    inline std::enable_if_t<!C::kUseMHC && C::kUseVLR, ResultRowVLR>
    QueryRetrievalVLR(const Key &key, ResultRowVLR mask) const {
        auto [was_bumped, result] = Super::QueryRetrieval(key, mask);
        if (was_bumped) {
            return result | child_ribbon_.QueryRetrievalVLR(key, was_bumped);
        } else {
            return result;
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC && C::kUseVLR, ResultRowVLR>
    QueryRetrievalMHCVLR(const mhc_or_key_t mhc, ResultRowVLR mask) const {
        auto [was_bumped, result] = Super::QueryRetrievalMHC(mhc, mask);
        if (was_bumped)
            return result | child_ribbon_.QueryRetrievalMHCVLR(mhc, was_bumped);
        else
            return result;
    }

    template <typename Iterator, typename C = Config>
    std::enable_if_t<!C::kUseMHC, bool> AddRangeInternal(Iterator begin, Iterator end) {
        const auto input_size = end - begin;
        std::vector<std::conditional_t<kUseVLR,
                                       std::conditional_t<kVLRShareMeta,
                                                          std::pair<Key, ResultRowVLR>,
                                                          std::tuple<Key, ResultRowVLR, ResultRowVLR>>,
                                       std::conditional_t<kIsFilter,
                                                          Key,
                                                          std::pair<Key, ResultRow>>
                                      >> bumped_items;
        bumped_items.reserve(std::max(
            0l, static_cast<ssize_t>((1 - slots_per_item_) * input_size)));
        // this calls either the vlr or non-vlr version automatically because of enable_if
        if (!Super::AddRange(begin, end, &bumped_items))
            return false;

        if (bumped_items.size() == 0)
            return true;
        // VLR num_slots is calculated in ribbon_base::AddRange
        if constexpr (!kUseVLR) {
            // child ribbon will round this up as needed
            const size_t child_slots =
                std::max(size_t{1},
                         static_cast<size_t>(slots_per_item_ * bumped_items.size()));
            child_ribbon_.Prepare(child_slots);
        }
        return child_ribbon_.AddRangeInternal(
            bumped_items.data(), bumped_items.data() + bumped_items.size()
        );
    }

    template <typename C = Config, typename Iterator>
    std::enable_if_t<C::kUseMHC, bool> AddRangeMHCInternal(Iterator begin, Iterator end) {
        const auto input_size = end - begin;
        std::vector<std::conditional_t<kUseVLR,
                                       std::conditional_t<kVLRShareMeta,
                                                          std::pair<mhc_or_key_t, ResultRowVLR>,
                                                          std::tuple<mhc_or_key_t, ResultRowVLR, ResultRowVLR>>,
                                       std::conditional_t<kIsFilter,
                                                          mhc_or_key_t,
                                                          std::pair<mhc_or_key_t, ResultRow>>
                                      >> bumped_items;
        bumped_items.reserve(std::max(
            0l, static_cast<ssize_t>((1 - slots_per_item_) * input_size)));

        // this calls either the vlr or non-vlr version automatically because of enable_if
        if (!Super::AddRange(begin, end, &bumped_items))
            return false;

        if (bumped_items.size() == 0)
            return true;
        if constexpr (!kUseVLR) {
            // child ribbon will round this up as needed
            const size_t child_slots =
                std::max(size_t{1},
                         static_cast<size_t>(slots_per_item_ * bumped_items.size()));
            child_ribbon_.Prepare(child_slots);
        }
        return child_ribbon_.AddRangeMHCInternal(
            bumped_items.data(), bumped_items.data() + bumped_items.size()
        );
    }

    void SerializeIntern(std::ostream &os) override {
        Super::SerializeIntern(os);
        child_ribbon_.SerializeIntern(os);
    }

    void DeserializeIntern(std::istream &is, bool switchendian, uint64_t seed, uint32_t idx, Index num_ribbons = 1) override {
        if constexpr (!kUseVLR)
            assert(num_ribbons == 1);
        Super::DeserializeIntern(is, switchendian, seed, idx, num_ribbons);
        child_ribbon_.DeserializeIntern(is, switchendian, seed + 1, idx + 1, num_ribbons);
    }

    ribbon_filter<depth - 1, Config> child_ribbon_;

    friend ribbon_filter<depth + 1, Config>;
    friend Super;
};

// base case ribbon
template <typename Config>
class ribbon_filter<0u, Config> : public ribbon_base<BaseConfig<Config>> {
public:
    IMPORT_RIBBON_CONFIG(BaseConfig<Config>);
    using Super = ribbon_base<BaseConfig<Config>>;

    double base_slots_per_item_ = 1.0;

    ribbon_filter() = default;

    ribbon_filter(std::istream &is) {
        Deserialize(is);
    }

    ribbon_filter(const std::string &filename) {
        Deserialize(filename);
    }

    template <typename C = Config>
    ribbon_filter(size_t num_slots, double parent_slots_per_item, uint64_t seed,
                  std::enable_if_t<!C::kUseVLR> * = 0) {
        if constexpr (!Config::kUseMHC) {
            Init(num_slots, parent_slots_per_item, seed);
        } else {
            // assume top level -> hasher index 0
            Init(num_slots, parent_slots_per_item, seed, 0);
        }
    }

    template <typename C = Config>
    ribbon_filter(size_t num_slots, double parent_slots_per_item, uint64_t seed,
                  uint32_t idx, std::enable_if_t<!C::kUseVLR> * = 0) {
        if constexpr (Config::kUseMHC) {
            Init(num_slots, parent_slots_per_item, seed, idx);
        } else {
            // you called the wrong constructor but we can just ignore it
            LOG1 << "Warning: called MHC base ribbon constructor in non-MHC "
                    "configuration, check the number of arguments! Should be "
                    "3, not 4. Ignoring last argument which is only relevant "
                    "in MHC configurations";
            Init(num_slots, parent_slots_per_item, seed);
        }
    }

    template <typename C = Config>
    ribbon_filter(double parent_slots_per_item, uint64_t seed, Index num_ribbons,
                  std::enable_if_t<C::kUseVLR> * = 0) {
        if constexpr (!Config::kUseMHC) {
            Init(parent_slots_per_item, seed, num_ribbons);
        } else {
            // assume top level -> hasher index 0
            Init(parent_slots_per_item, seed, 0, num_ribbons);
        }
    }

    template <typename C = Config>
    ribbon_filter(double parent_slots_per_item, uint64_t seed, uint32_t idx,
                  Index num_ribbons, std::enable_if_t<C::kUseVLR> * = 0) {
        if constexpr (Config::kUseMHC) {
            Init(parent_slots_per_item, seed, idx, num_ribbons);
        } else {
            // you called the wrong constructor but we can just ignore it
            LOG1 << "Warning: called MHC base ribbon constructor in non-MHC "
                    "configuration, check the number of arguments! Should be "
                    "3, not 4. Ignoring last argument which is only relevant "
                    "in MHC configurations";
            Init(parent_slots_per_item, seed, num_ribbons);
        }
    }

    template <typename C = Config>
    std::enable_if_t<!C::kUseMHC && !C::kUseVLR, void>
    Init(size_t num_slots, double parent_slots_per_item, uint64_t seed) {
        parent_slots_per_item_ = parent_slots_per_item;
        orig_num_slots_ = num_slots / parent_slots_per_item * base_slots_per_item_;
        Super::Init(orig_num_slots_, base_slots_per_item_, seed);
    }

    template <typename C = Config>
    std::enable_if_t<C::kUseMHC && !C::kUseVLR, void> Init(size_t num_slots,
                                            double parent_slots_per_item,
                                            uint64_t seed, uint32_t idx) {
        parent_slots_per_item_ = parent_slots_per_item;
        orig_num_slots_ = num_slots / parent_slots_per_item * base_slots_per_item_;
        Super::Init(orig_num_slots_, base_slots_per_item_, seed, idx);
    }

    template <typename C = Config>
    std::enable_if_t<!C::kUseMHC && C::kUseVLR, void>
    Init(double parent_slots_per_item, uint64_t seed, Index num_ribbons) {
        parent_slots_per_item_ = parent_slots_per_item;
        Super::Init(0, base_slots_per_item_, seed, num_ribbons);
    }

    template <typename C = Config>
    std::enable_if_t<C::kUseMHC && C::kUseVLR, void>
    Init(double parent_slots_per_item, uint64_t seed, uint32_t idx, Index num_ribbons) {
        parent_slots_per_item_ = parent_slots_per_item;
        Super::Init(0, base_slots_per_item_, seed, idx, num_ribbons);
    }

    template <typename Iterator, typename C = Config>
    std::enable_if_t<!C::kUseMHC, bool> AddRange(Iterator begin, Iterator end) {
        // there's really no distinction in the base case
        return AddRangeMHCInternal<true>(begin, end);
    }

    // MHC version
    template <typename Iterator, typename C = Config>
    std::enable_if_t<C::kUseMHC, bool> AddRange(Iterator begin, Iterator end) {
        auto input = Super::PrepareAddRangeMHC(begin, end);
        return AddRangeMHCInternal<true>(input.get(), input.get() + (end - begin));
    }

    template <typename C = Config>
    inline std::enable_if_t<!C::kUseVLR, bool>
    QueryFilter(const Key &key) const {
        if constexpr (kUseMHC) {
            const auto mhc = this->hasher_.GetMHC(key);
            return QueryFilterMHC(mhc);
        } else {
            auto [was_bumped, found] = Super::QueryFilter(key);
            assert(!was_bumped);
            return found;
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC && !C::kUseVLR, bool>
    QueryFilterMHC(const typename Super::mhc_or_key_t mhc) const {
        auto [was_bumped, found] = Super::QueryFilterMHC(mhc);
        assert(!was_bumped);
        return found;
    }

    inline std::conditional_t<kUseVLR, ResultRowVLR, ResultRow> QueryRetrieval(const Key &key, [[maybe_unused]] Index start_idx = 0, [[maybe_unused]] Index num_bits = 0) const {
        if constexpr (kUseMHC) {
            const auto mhc = this->hasher_.GetMHC(key);
            return QueryRetrievalMHC(mhc);
        } else if constexpr (kUseVLR && !kVLRShareMeta) {
            return QueryRetrievalVLR(key, Super::GetVLRMask(start_idx, num_bits));
        } else {
            auto [was_bumped, result] = Super::QueryRetrieval(key, start_idx, num_bits);
            // FIXME: maybe set metadata of all buckets in base case ribbon so was_bumped cannot occur
            //assert(!was_bumped);
            return result;
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC, std::conditional_t<kUseVLR, ResultRowVLR, ResultRow>>
    QueryRetrievalMHC(const typename Super::mhc_or_key_t mhc, [[maybe_unused]] Index start_idx = 0, [[maybe_unused]] Index num_bits = 0) const {
        if constexpr (kUseVLR && !kVLRShareMeta) {
            return QueryRetrievalMHCVLR(mhc, Super::GetVLRMask(start_idx, num_bits));
        } else {
            auto [was_bumped, result] = Super::QueryRetrievalMHC(mhc, start_idx, num_bits);
            //assert(!was_bumped);
            return result;
        }
    }

    void Serialize(const std::string &filename) {
        Super::Serialize(filename, 0);
    }

    void Deserialize(const std::string &filename) {
        Super::Deserialize(filename, 0);
    }

    void Serialize(std::ostream &os) {
        Super::Serialize(os, 0);
    }

    void Deserialize(std::istream &is) {
        Super::Deserialize(is, 0);
    }

protected:
    template <typename C = Config>
    inline std::enable_if_t<!C::kUseMHC && C::kUseVLR, ResultRowVLR>
    QueryRetrievalVLR(const Key &key, ResultRowVLR mask) const {
        auto [was_bumped, result] = Super::QueryRetrieval(key, mask);
        return result;
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC && C::kUseVLR, ResultRowVLR>
    QueryRetrievalMHCVLR(const typename Super::mhc_or_key_t mhc, ResultRowVLR mask) const {
        auto [was_bumped, result] = Super::QueryRetrievalMHC(mhc, mask);
        return result;
    }

    void Prepare(size_t num_slots) {
        orig_num_slots_ = num_slots / parent_slots_per_item_ * base_slots_per_item_;
        Super::Prepare(orig_num_slots_);
    }

    // same as AddRangeMHCInternal in base case
    template <typename Iterator>
    bool AddRangeInternal(Iterator begin, Iterator end) {
        return AddRangeMHCInternal(begin, end);
    }

    // misnomer but needed for recursive construction
    template <typename Iterator>
    bool AddRangeMHCInternal(Iterator begin, Iterator end) {
        bool success;
        // In the VLR version, we only know the number of slots after AddRange has been called,
        // so we have to wait before we can set orig_num_slots.
        [[maybe_unused]] bool num_slots_set = false;
        do {
            // if VLR is used, we only know the number of slots after it is calculated by Super::AddRange
            if constexpr (kUseVLR) {
                // This automatically calls different functions depending on the value of kVLRShareMeta.
                success = Super::AddRange(begin, end, nullptr);
                if (!num_slots_set) {
                    orig_num_slots_ = Super::storage_.GetNumSlots();
                    num_slots_set = true;
                }
            } else {
                success = Super::AddRange(begin, end, nullptr);
            }
            if (!success) {
                // increase epsilon and try again
                base_slots_per_item_ += 0.05;
                size_t new_num_slots = orig_num_slots_ * base_slots_per_item_;
                Super::Realloc(new_num_slots, base_slots_per_item_);
            }
        } while (!success);
        return success;
    }

    size_t orig_num_slots_ = 0;
    double parent_slots_per_item_;
    friend ribbon_filter<1, Config>;
    friend Super;
};

} // namespace ribbon
