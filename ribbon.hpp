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

#include <tlx/logger.hpp>

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
    ribbon_base(size_t num_slots, double slots_per_item, uint64_t seed) {
        if constexpr (!Config::kUseMHC)
            Init(num_slots, slots_per_item, seed);
        else
            assert(false);
    }

    // TODO make this invalid if !kUseMHC
    ribbon_base(size_t num_slots, double slots_per_item, uint64_t seed,
                uint32_t idx) {
        if constexpr (Config::kUseMHC)
            Init(num_slots, slots_per_item, seed, idx);
        else
            assert(false);
    }

    // Specialisation for Master Hash Code Hasher
    template <typename C = Config>
    std::enable_if_t<C::kUseMHC, void>
    Init(size_t num_slots, double slots_per_item, uint64_t seed, uint32_t idx) {
        sLOGC(Config::log) << "ribbon_base (MHC) called with" << num_slots
                           << "slots and" << slots_per_item << "slots per item";
        slots_per_item_ = slots_per_item;
        hasher_.Seed(seed, idx);
        if (num_slots > 0)
            Prepare(num_slots);
    }

    template <typename C = Config>
    std::enable_if_t<!C::kUseMHC, void>
    Init(size_t num_slots, double slots_per_item, uint64_t seed) {
        sLOGC(Config::log) << "ribbon_base called with" << num_slots
                           << "slots and" << slots_per_item << "slots per item";
        static_assert(!Config::kUseMHC, "wat");
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

    inline std::pair<bool, bool> QueryFilter(const Key &key) const {
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
    inline std::enable_if_t<C::kUseMHC, std::pair<bool, bool>>
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

    template <typename Iterator>
    bool AddRange(
        Iterator begin, Iterator end,
        std::vector<std::conditional_t<
            kUseMHC, std::conditional_t<kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, ResultRow>>,
            typename std::iterator_traits<Iterator>::value_type>> *bump_vec) {
        const auto input_size = end - begin;
        LOGC(Config::log) << "Constructing for " << storage_.GetNumSlots()
                          << " slots, " << storage_.GetNumStarts() << " starts, "
                          << storage_.GetNumBuckets() << " buckets";

        rocksdb::StopWatchNano timer(true);
        bool success;
        if constexpr (kUseMHC) {
            success = BandingAddRangeMHC(&storage_, hasher_, begin, end, bump_vec);
        } else {
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

    void BackSubst() {
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

    inline std::pair<bool, ResultRow> QueryRetrieval(const Key &key) const {
        if constexpr (kIsFilter || kUseMHC) {
            assert(false);
            return std::make_pair(true, 0);
        }
        if constexpr (kUseInterleavedSol) {
            return InterleavedRetrievalQuery(key, hasher_, sol_);
        } else if constexpr (kUseCacheLineStorage) {
            return SimpleRetrievalQuery(key, hasher_, sol_);
        } else {
            return SimpleRetrievalQuery(key, hasher_, storage_);
        }
    }

    inline std::pair<bool, ResultRow> QueryRetrievalMHC(const mhc_or_key_t mhc) const {
        if constexpr (kIsFilter || !kUseMHC) {
            assert(false);
            return std::make_pair(true, 0);
        }
        if constexpr (kUseInterleavedSol) {
            return InterleavedRetrievalQuery(mhc, hasher_, sol_);
        } else if constexpr (kUseCacheLineStorage) {
            return SimpleRetrievalQuery(mhc, hasher_, sol_);
        } else {
            return SimpleRetrievalQuery(mhc, hasher_, storage_);
        }
    }

    std::tuple<ssize_t, ssize_t, double, ssize_t> GetStats() const {
        double frac_empty =
            static_cast<double>(empty_slots) / storage_.GetNumSlots();
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
            num_slots = sol_.GetNumSlots();
        } else {
            sol_size = storage_.Size();
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
    void Prepare(size_t num_slots) {
        if (num_slots == 0)
            return;
        // round up to next multiple of kCoeffBits for interleaved storage
        if constexpr (kUseInterleavedSol)
            num_slots = ((num_slots + kCoeffBits - 1) / kCoeffBits) * kCoeffBits;
        else
            num_slots = std::max(num_slots, static_cast<size_t>(kCoeffBits));
        storage_.Prepare(num_slots);
        hasher_.Prepare(num_slots, slots_per_item_);
    }

    template <typename Iterator, typename C = Config>
    std::enable_if_t<C::kUseMHC,
                     std::unique_ptr<std::conditional_t<
                         kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, ResultRow>>[]>>
    PrepareAddRangeMHC(Iterator begin, Iterator end) {
        // this is the top-level filter, transform from vector<Key> to vector<mhc_t>
        rocksdb::StopWatchNano timer(true);
        const auto input_size = end - begin;
        using value_type =
            std::conditional_t<kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, ResultRow>>;
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

    // statistics
    ssize_t num_bumped;
    ssize_t empty_slots;

    // actual data
    double slots_per_item_;
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

    // MHC top-level constructor
    template <typename C = Config>
    ribbon_filter(size_t num_slots, double slots_per_item, uint64_t seed,
                  typename std::enable_if<C::kUseMHC>::type * = 0)
        : Super(num_slots, slots_per_item, seed, 0),
          child_ribbon_(0, slots_per_item, seed, 1) {}

protected:
    // MHC child constructor
    template <typename C = Config>
    ribbon_filter(size_t num_slots, double slots_per_item, uint64_t seed,
                  uint32_t idx, typename std::enable_if<C::kUseMHC>::type * = 0)
        : Super(num_slots, slots_per_item, seed, idx),
          child_ribbon_(0, slots_per_item, seed, idx + 1) {}

public:
    // non-MHC top-level constructor
    template <typename C = Config>
    ribbon_filter(size_t num_slots, double slots_per_item, uint64_t seed,
                  typename std::enable_if<!C::kUseMHC>::type * = 0)
        : Super(num_slots, slots_per_item, seed),
          child_ribbon_(0, slots_per_item, seed + 1) {}

    // non-MHC init
    template <typename C = Config>
    std::enable_if_t<!C::kUseMHC, void>
    Init(size_t num_slots, double slots_per_item, uint64_t seed) {
        Super::Init(num_slots, slots_per_item, seed);
        child_ribbon_.Init(0, slots_per_item, seed + 1);
    }

    // MHC init
    template <typename C = Config>
    std::enable_if_t<C::kUseMHC, void>
    Init(size_t num_slots, double slots_per_item, uint64_t seed, uint32_t idx) {
        Super::Init(num_slots, slots_per_item, seed, idx);
        child_ribbon_.Init(0, slots_per_item, seed, idx + 1);
    }

    template <typename Iterator, typename C = Config>
    std::enable_if_t<!C::kUseMHC, bool> AddRange(Iterator begin, Iterator end) {
        const auto input_size = end - begin;
        std::vector<std::conditional_t<kIsFilter, Key, std::pair<Key, ResultRow>>> bumped_items;
        bumped_items.reserve(std::max(
            0l, static_cast<ssize_t>((1 - slots_per_item_) * input_size)));

        if (!Super::AddRange(begin, end, &bumped_items))
            return false;

        if (bumped_items.size() == 0)
            return true;
        // child ribbon will round this up as needed
        const size_t child_slots =
            std::max(size_t{1},
                     static_cast<size_t>(slots_per_item_ * bumped_items.size()));
        child_ribbon_.Prepare(child_slots);
        return child_ribbon_.AddRange(bumped_items.data(),
                                      bumped_items.data() + bumped_items.size());
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

    inline bool QueryFilter(const Key &key) const {
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
    inline std::enable_if_t<C::kUseMHC, bool>
    QueryFilterMHC(const mhc_or_key_t mhc) const {
        auto [was_bumped, found] = Super::QueryFilterMHC(mhc);
        if (was_bumped)
            return child_ribbon_.QueryFilterMHC(mhc);
        else
            return found;
    }

    inline ResultRow QueryRetrieval(const Key &key) const {
        if constexpr (kUseMHC) {
            const auto mhc = this->hasher_.GetMHC(key);
            return QueryRetrievalMHC(mhc);
        } else {
            auto [was_bumped, result] = Super::QueryRetrieval(key);
            if (was_bumped) {
                return child_ribbon_.QueryRetrieval(key);
            } else {
                return result;
            }
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC, ResultRow>
    QueryRetrievalMHC(const mhc_or_key_t mhc) const {
        auto [was_bumped, result] = Super::QueryRetrievalMHC(mhc);
        if (was_bumped)
            return child_ribbon_.QueryRetrievalMHC(mhc);
        else
            return result;
    }

    size_t Size() const {
        return Super::Size() + child_ribbon_.Size();
    }

    /*
    void PrintStats() const {
        Super::PrintStats();
        child_ribbon_.PrintStats();
    }
    */

protected:
    template <typename C = Config, typename Iterator>
    std::enable_if_t<C::kUseMHC, bool> AddRangeMHCInternal(Iterator begin,
                                                           Iterator end) {
        const auto input_size = end - begin;
        using value_type =
            std::conditional_t<kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, ResultRow>>;
        std::vector<value_type> bumped_items;
        bumped_items.reserve(std::max(
            0l, static_cast<ssize_t>((1 - slots_per_item_) * input_size)));

        if (!Super::AddRange(begin, end, &bumped_items))
            return false;

        if (bumped_items.size() == 0)
            return true;
        // child ribbon will round this up as needed
        const size_t child_slots =
            std::max(size_t{1},
                     static_cast<size_t>(slots_per_item_ * bumped_items.size()));
        child_ribbon_.Prepare(child_slots);
        return child_ribbon_.AddRangeMHCInternal(
            bumped_items.data(), bumped_items.data() + bumped_items.size());
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
    ribbon_filter(size_t num_slots, double parent_slots_per_item, uint64_t seed) {
        if constexpr (!Config::kUseMHC) {
            Init(num_slots, parent_slots_per_item, seed);
        } else {
            // assume top level -> hasher index 0
            Init(num_slots, parent_slots_per_item, seed, 0);
        }
    }
    ribbon_filter(size_t num_slots, double parent_slots_per_item, uint64_t seed,
                  uint32_t idx) {
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
    std::enable_if_t<!C::kUseMHC, void>
    Init(size_t num_slots, double parent_slots_per_item, uint64_t seed) {
        parent_slots_per_item_ = parent_slots_per_item;
        orig_num_slots_ = num_slots / parent_slots_per_item * base_slots_per_item_;
        Super::Init(orig_num_slots_, base_slots_per_item_, seed);
    }

    template <typename C = Config>
    std::enable_if_t<C::kUseMHC, void> Init(size_t num_slots,
                                            double parent_slots_per_item,
                                            uint64_t seed, uint32_t idx) {
        parent_slots_per_item_ = parent_slots_per_item;
        orig_num_slots_ = num_slots / parent_slots_per_item * base_slots_per_item_;
        Super::Init(orig_num_slots_, base_slots_per_item_, seed, idx);
    }

    template <typename Iterator, typename C = Config>
    std::enable_if_t<!C::kUseMHC, bool> AddRange(Iterator begin, Iterator end) {
        // there's really no distinction in the base case
        return AddRangeMHCInternal(begin, end);
    }

    // MHC version
    template <typename Iterator, typename C = Config>
    std::enable_if_t<C::kUseMHC, bool> AddRange(Iterator begin, Iterator end) {
        auto input = Super::PrepareAddRangeMHC(begin, end);
        return AddRangeMHCInternal(input.get(), input.get() + (end - begin));
    }

    inline bool QueryFilter(const Key &key) const {
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
    inline std::enable_if_t<C::kUseMHC, bool>
    QueryFilterMHC(const typename Super::mhc_or_key_t mhc) const {
        auto [was_bumped, found] = Super::QueryFilterMHC(mhc);
        assert(!was_bumped);
        return found;
    }

    ResultRow QueryRetrieval(const Key &key) const {
        auto [was_bumped, result] = Super::QueryRetrieval(key);
        assert(!was_bumped);
        return result;
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC, ResultRow>
    QueryRetrievalMHC(const typename Super::mhc_or_key_t mhc) const {
        auto [was_bumped, result] = Super::QueryRetrievalMHC(mhc);
        assert(!was_bumped);
        return result;
    }


protected:
    void Prepare(size_t num_slots) {
        orig_num_slots_ = num_slots / parent_slots_per_item_ * base_slots_per_item_;
        Super::Prepare(orig_num_slots_);
    }

    // misnomer but needed for recursive construction
    template <typename Iterator>
    bool AddRangeMHCInternal(Iterator begin, Iterator end) {
        bool success;
        do {
            success = Super::AddRange(begin, end, nullptr);
            if (!success) {
                // increase epsilon and try again
                base_slots_per_item_ += 0.05;
                size_t new_num_slots = orig_num_slots_ * base_slots_per_item_;
                Super::Realloc(new_num_slots, base_slots_per_item_);
            }
        } while (!success);
        return success;
    }

    size_t orig_num_slots_;
    double parent_slots_per_item_;
    friend ribbon_filter<1, Config>;
    friend Super;
};

} // namespace ribbon
