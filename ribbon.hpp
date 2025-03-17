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

struct RibbonLevelStats {
    ssize_t num_bumped;
    ssize_t empty_slots;
    size_t num_threads;
    uint64_t sort_time;
    size_t size;
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

    RibbonLevelStats GetSingleLevelStats() const {
        return RibbonLevelStats{num_bumped, empty_slots, num_threads_final, sort_time, Size()};
    }

protected:
    /* NOTE: The internal functions also call the sequential version if num_threads == 0, just in case.
       However, these functions should never be called with num_threads == 0 since the public functions
       already handle that case. */
    template <typename Iterator>
    bool AddRangeInternal(
        Iterator begin, Iterator end,
        std::vector<std::conditional_t<
            kUseMHC, std::conditional_t<kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, ResultRow>>,
            typename std::iterator_traits<Iterator>::value_type>> *bump_vec, std::size_t num_threads = 1) {
        const auto input_size = end - begin;
        LOGC(Config::log) << "Constructing for " << storage_.GetNumSlots()
                          << " slots, " << storage_.GetNumStarts() << " starts, "
                          << storage_.GetNumBuckets() << " buckets";

        rocksdb::StopWatchNano timer(true);
        bool success;
        if constexpr (kUseMHC) {
            if (num_threads <= 1)
                std::tie(success, num_threads_final, sort_time) = BandingAddRangeMHC(&storage_, hasher_, begin, end, bump_vec);
            #ifdef _REENTRANT
            else
                std::tie(success, num_threads_final, sort_time) = BandingAddRangeParallelMHC(&storage_, hasher_, begin, end, bump_vec, num_threads);
            #else
            else {
                std::cerr << "Parallel version called but not compiled in. This should be impossible.\n";
                abort(); /* should never happen */
            }
            #endif
        } else {
            if (num_threads <= 1)
                std::tie(success, num_threads_final, sort_time) = BandingAddRange(&storage_, hasher_, begin, end, bump_vec);
            #ifdef _REENTRANT
            else
                std::tie(success, num_threads_final, sort_time) = BandingAddRangeParallel(&storage_, hasher_, begin, end, bump_vec, num_threads);
            #else
            else {
                std::cerr << "Parallel version called but not compiled in. This should be impossible.\n";
                abort(); /* should never happen */
            }
            #endif
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

    void BackSubstInternal(std::size_t num_threads = 1) {
        if (storage_.GetNumSlots() == 0)
            return;
        rocksdb::StopWatchNano timer(true);
        if constexpr (kUseInterleavedSol) {
            if (num_threads <= 1)
                InterleavedBackSubst(storage_, &sol_);
            #ifdef _REENTRANT
            else
                InterleavedBackSubstParallel(storage_, &sol_, num_threads);
            #else
            else {
                std::cerr << "Parallel version called but not compiled in. This should be impossible.\n";
                abort(); /* should never happen */
            }
            #endif
            // move metadata by swapping pointers
            sol_.MoveMetadata(&storage_);
            storage_.Reset();
        } else if constexpr (kUseCacheLineStorage) {
            sol_.Prepare(storage_.GetNumSlots());
            if (num_threads <= 1)
                SimpleBackSubst(storage_, &sol_);
            #ifdef _REENTRANT
            else
                SimpleBackSubstParallel(storage_, &sol_, num_threads);
            #else
            else {
                std::cerr << "Parallel version called but not compiled in. This should be impossible.\n";
                abort(); /* should never happen */
            }
            #endif
            /* FIXME: parallize setting the metadata */
            // copy metadata one-by-one
            for (Index bucket = 0; bucket < storage_.GetNumBuckets(); ++bucket) {
                sol_.SetMeta(bucket, storage_.GetMeta(bucket));
            }
            storage_.Reset();
        } else {
            if (num_threads <= 1)
                SimpleBackSubst(storage_, &storage_);
            #ifdef _REENTRANT
            else
                SimpleBackSubstParallel(storage_, &storage_, num_threads);
            #else
            else {
                std::cerr << "Parallel version called but not compiled in. This should be impossible.\n";
                abort(); /* should never happen */
            }
            #endif
        }

        LOGC(Config::log) << "Backsubstitution for " << storage_.GetNumSlots()
                          << " slots took " << timer.ElapsedNanos(true) / 1e6
                          << "ms";
    }

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
    PrepareAddRangeMHC(Iterator begin, Iterator end, std::size_t num_threads = 1) {
        // this is the top-level filter, transform from vector<Key> to vector<mhc_t>
        rocksdb::StopWatchNano timer(true);
        const auto input_size = end - begin;
        using value_type =
            std::conditional_t<kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, ResultRow>>;
        auto input = std::make_unique<value_type[]>(input_size);
        if (num_threads <= 1) {
            auto it = begin;
            for (ssize_t i = 0; i < input_size; i++, ++it) {
                if constexpr (kIsFilter)
                    input[i] = hasher_.GetMHC(*it);
                else
                    input[i] = std::make_pair(hasher_.GetMHC(it->first), it->second);
            }
        #ifdef _REENTRANT
        } else {
            std::size_t items_per_thread = (input_size + num_threads - 1) / num_threads;
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            for (std::size_t ti = 0; ti < num_threads; ++ti) {
                threads.emplace_back([&, ti]() {
                    ssize_t start_idx = static_cast<ssize_t>(ti * items_per_thread);
                    ssize_t end_idx = static_cast<ssize_t>(std::min((ti + 1) * items_per_thread, static_cast<std::size_t>(input_size)));
                    auto it = begin + start_idx;
                    for (ssize_t i = start_idx; i < end_idx; i++, ++it) {
                        if constexpr (kIsFilter)
                            input[i] = hasher_.GetMHC(*it);
                        else
                            input[i] = std::make_pair(hasher_.GetMHC(it->first), it->second);
                    }
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }
        #else
        } else {
            std::cerr << "Parallel version called but not compiled in. This should be impossible.\n";
            abort(); /* should never happen */
        }
        #endif
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

    virtual void DeserializeIntern(std::istream &is, bool switchendian, uint64_t seed, uint32_t idx) {
        Index num_buckets = 0;
        if constexpr (kUseCacheLineStorage) {
            throw config_error("kUseCacheLineStorage not supported");
        } else if constexpr (kUseInterleavedSol) {
            sol_.DeserializeIntern(is, switchendian);
            num_buckets = sol_.GetNumBuckets();
        } else {
            storage_.DeserializeIntern(is, switchendian);
            num_buckets = storage_.GetNumBuckets();
        }
        hasher_.DeserializeIntern(is, switchendian, num_buckets);
        if constexpr (Config::kUseMHC) {
            hasher_.Seed(seed, idx);
        } else {
            hasher_.Seed(seed);
        }
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
        unsigned char bits = kUseMultiplyShiftHash | (kIsFilter << 1) |
              (kFirstCoeffAlwaysOne << 2) | (kSparseCoeffs << 3) |
              (kUseInterleavedSol << 4) | (kUseMHC << 5);
        os.write(reinterpret_cast<const char *>(&bits), 1);

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
        DeserializeIntern(is, switchendian, seed, 0);

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
    size_t num_threads_final = 1;
    uint64_t sort_time = 0;

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

    ribbon_filter(std::istream &is) {
        Deserialize(is);
    }

    ribbon_filter(const std::string &filename) {
        Deserialize(filename);
    }

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

    template <typename Iterator>
    bool AddRange(Iterator begin, Iterator end) {
        return AddRangeInternal(begin, end);
    }

    void BackSubst() {
        BackSubstInternal();
    }

    #ifdef _REENTRANT
    template <typename Iterator>
    bool AddRange(Iterator begin, Iterator end, std::size_t num_threads) {
        if (num_threads == 0)
            num_threads = std::thread::hardware_concurrency();
        return AddRangeInternal(begin, end, num_threads);
    }
    void BackSubst(std::size_t num_threads) {
        if (num_threads == 0)
            num_threads = std::thread::hardware_concurrency();
        BackSubstInternal(num_threads);
    }
    #endif

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

    std::vector<RibbonLevelStats> GetLevelStats() const {
        auto stats = child_ribbon_.GetLevelStats();
        stats.push_back(Super::GetSingleLevelStats());
        return stats;
    }

protected:
    template <typename Iterator, typename C = Config>
    std::enable_if_t<!C::kUseMHC, bool> AddRangeInternal(Iterator begin, Iterator end, std::size_t num_threads = 1) {
        const auto input_size = end - begin;
        std::vector<std::conditional_t<kIsFilter, Key, std::pair<Key, ResultRow>>> bumped_items;
        bumped_items.reserve(std::max(
            0l, static_cast<ssize_t>((1 - slots_per_item_) * input_size)));

        if (!Super::AddRangeInternal(begin, end, &bumped_items, num_threads))
            return false;

        if (bumped_items.size() == 0)
            return true;
        // child ribbon will round this up as needed
        const size_t child_slots =
            std::max(size_t{1},
                     static_cast<size_t>(slots_per_item_ * bumped_items.size()));
        child_ribbon_.Prepare(child_slots);
        return child_ribbon_.AddRangeInternal(bumped_items.data(),
                                      bumped_items.data() + bumped_items.size(), num_threads);
    }

    // MHC version
    template <typename Iterator, typename C = Config>
    std::enable_if_t<C::kUseMHC, bool> AddRangeInternal(Iterator begin, Iterator end, std::size_t num_threads = 1) {
        auto input = Super::PrepareAddRangeMHC(begin, end, num_threads);
        return AddRangeMHCInternal(input.get(), input.get() + (end - begin), num_threads);
    }

    void BackSubstInternal(std::size_t num_threads = 1) {
        Super::BackSubstInternal(num_threads);
        child_ribbon_.BackSubstInternal(num_threads);
    }

    template <typename C = Config, typename Iterator>
    std::enable_if_t<C::kUseMHC, bool> AddRangeMHCInternal(Iterator begin,
                                                           Iterator end,
                                                           std::size_t num_threads = 1) {
        const auto input_size = end - begin;
        using value_type =
            std::conditional_t<kIsFilter, mhc_or_key_t, std::pair<mhc_or_key_t, ResultRow>>;
        std::vector<value_type> bumped_items;
        bumped_items.reserve(std::max(
            0l, static_cast<ssize_t>((1 - slots_per_item_) * input_size)));

        if (!Super::AddRangeInternal(begin, end, &bumped_items, num_threads))
            return false;

        if (bumped_items.size() == 0)
            return true;
        // child ribbon will round this up as needed
        const size_t child_slots =
            std::max(size_t{1},
                     static_cast<size_t>(slots_per_item_ * bumped_items.size()));
        child_ribbon_.Prepare(child_slots);
        return child_ribbon_.AddRangeMHCInternal(
            bumped_items.data(), bumped_items.data() + bumped_items.size(), num_threads);
    }

    void SerializeIntern(std::ostream &os) override {
        Super::SerializeIntern(os);
        child_ribbon_.SerializeIntern(os);
    }

    void DeserializeIntern(std::istream &is, bool switchendian, uint64_t seed, uint32_t idx) override {
        Super::DeserializeIntern(is, switchendian, seed, idx);
        child_ribbon_.DeserializeIntern(is, switchendian, seed + 1, idx + 1);
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

    template <typename Iterator>
    bool AddRange(Iterator begin, Iterator end) {
        return AddRangeInternal(begin, end);
    }

    void BackSubst() {
        BackSubstInternal();
    }

    #ifdef _REENTRANT
    template <typename Iterator>
    bool AddRange(Iterator begin, Iterator end, std::size_t num_threads) {
        if (num_threads == 0)
            num_threads = std::thread::hardware_concurrency();
        return AddRangeInternal(begin, end, num_threads);
    }
    void BackSubst(std::size_t num_threads) {
        if (num_threads == 0)
            num_threads = std::thread::hardware_concurrency();
        BackSubstInternal(num_threads);
    }
    #endif

    inline bool QueryFilter(const Key &key) const {
        if constexpr (kUseMHC) {
            const auto mhc = this->hasher_.GetMHC(key);
            return QueryFilterMHC(mhc);
        } else {
            auto [was_bumped, found] = Super::QueryFilter(key);
            // FIXME: is there any sensible assert that could still work?
            // This assert can fail in certain circumstances with
            // negative queries. There can be buckets that no inserted
            // item is mapped to, in which case the bucket is marked
            // as "all bumped", meaning that any negative queries
            // mapped to this bucket will return was_bumped.
            // An alternative option would be to set the metadata for
            // such a bucket to "nothing bumped".
            //assert(!was_bumped);
            return found;
        }
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC, bool>
    QueryFilterMHC(const typename Super::mhc_or_key_t mhc) const {
        auto [was_bumped, found] = Super::QueryFilterMHC(mhc);
        // see comment in QueryFilter
        //assert(!was_bumped);
        return found;
    }

    ResultRow QueryRetrieval(const Key &key) const {
        auto [was_bumped, result] = Super::QueryRetrieval(key);
        // see comment in QueryFilter
        //assert(!was_bumped);
        return result;
    }

    template <typename C = Config>
    inline std::enable_if_t<C::kUseMHC, ResultRow>
    QueryRetrievalMHC(const typename Super::mhc_or_key_t mhc) const {
        auto [was_bumped, result] = Super::QueryRetrievalMHC(mhc);
        // see comment in QueryFilter
        //assert(!was_bumped);
        return result;
    }

    std::vector<RibbonLevelStats> GetLevelStats() const {
        std::vector<RibbonLevelStats> stats;
        stats.push_back(Super::GetSingleLevelStats());
        return stats;
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
    template <typename Iterator, typename C = Config>
    std::enable_if_t<!C::kUseMHC, bool> AddRangeInternal(Iterator begin, Iterator end, std::size_t num_threads = 1) {
        // there's really no distinction in the base case
        return AddRangeMHCInternal(begin, end, num_threads);
    }

    // MHC version
    template <typename Iterator, typename C = Config>
    std::enable_if_t<C::kUseMHC, bool> AddRangeInternal(Iterator begin, Iterator end, std::size_t num_threads = 1) {
        auto input = Super::PrepareAddRangeMHC(begin, end);
        return AddRangeMHCInternal(input.get(), input.get() + (end - begin), num_threads);
    }

    void BackSubstInternal(std::size_t num_threads = 1) {
        (void)num_threads;
        /* the base case ribbon doesn't use any bumping, so the
           sequential version of back substitution must be used */
        Super::BackSubstInternal(1);
    }

    void Prepare(size_t num_slots) {
        orig_num_slots_ = num_slots / parent_slots_per_item_ * base_slots_per_item_;
        Super::Prepare(orig_num_slots_);
    }

    // misnomer but needed for recursive construction
    template <typename Iterator>
    bool AddRangeMHCInternal(Iterator begin, Iterator end, std::size_t num_threads = 1) {
        bool success;
        do {
            success = Super::AddRangeInternal(begin, end, nullptr, num_threads);
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
