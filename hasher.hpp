//  Copyright (c) Lorenz Hübschle-Schneider
//  Copyright (c) Facebook, Inc. and its affiliates.
//  All Rights Reserved.  This source code is licensed under the Apache 2.0
//  License (found in the LICENSE file in the root directory).

#pragma once

#include "config.hpp"
#include "pcg-cpp/include/pcg_random.hpp"
#include "rocksdb/coding.h"
#include "rocksdb/fastrange.h"

#include <tlx/math/integer_log2.hpp>

#include <utility>

namespace ribbon {

namespace {
// for std::conditional_t<cond, ActualThing, DummyData>
struct DummyData {};
} // namespace

// Based on Peter Dillinger / Facebook's rocksdb::ribbon::StandardHasher
template <typename Config>
class Hasher {
public:
    IMPORT_RIBBON_CONFIG(Config);
    static constexpr bool log = Config::log; // export for BandingAddRange...
    static_assert(sizeof(Hash) == 8 || sizeof(Hash) == 4,
                  "Unsupported hash size");
    static_assert(sizeof(CoeffRow) <= 16,
                  "More than 64 coefficient bits not supported atm");

    Hasher() = default;
    explicit Hasher(uint64_t seed) {
        Seed(seed);
    }

    void Seed(uint64_t seed) {
        seed_ = seed;
        if constexpr (kUseMultiplyShiftHash) {
            pcg32 random(seed);
            for (auto v : {&multiply_, &add_}) {
                *v = random();
                for (int i = 1; i <= 4; ++i) {
                    *v = *v << 32;
                    *v |= random();
                }
            }
        }
    }

    inline Hash GetHash(const Key& key) const {
        if constexpr (kUseMultiplyShiftHash && std::is_arithmetic_v<Key>) {
            return (add_ + multiply_ * static_cast<decltype(multiply_)>(key)) >> 64;
        } else {
            return Config::HashFn(key, seed_);
        }
    }

    inline Hash GetHash(const std::pair<Key, ResultRow>& in) const {
        return GetHash(in.first);
    }

    // this has to be fast
    inline Index GetStart(Hash h, Index num_starts) const {
        return rocksdb::FastRangeGeneric(h, num_starts);
    }

    inline CoeffRow GetCoeffs(Hash h) const {
        Hash a = h * kCoeffAndResultFactor + kCARFadd;
        CoeffRow cr;

        // expand hash to 64 bits if needed
        if constexpr (sizeof(CoeffRow) > sizeof(Hash)) {
            uint64_t b = a;
            // Almost-trivial hash expansion, favoring roughly
            // equal number of 1's and 0's in result
            b = (b << 32) ^ b ^ kCoeffXor32;

            if constexpr (sizeof(CoeffRow) == 16) {
                __uint128_t c = b;
                c = (c << 64) ^ c ^ kCoeffXor64;
                cr = static_cast<CoeffRow>(c);
            } else {
                static_assert(sizeof(Hash) < 8);
                cr = static_cast<CoeffRow>(b);
            }
        } else {
            cr = static_cast<CoeffRow>(a);
        }

        // Now ensure the value is non-zero
        if constexpr (kFirstCoeffAlwaysOne) {
            cr |= 1;
        } else if constexpr (sizeof(CoeffRow) <= sizeof(Hash)) {
            // Still have to ensure some bit is non-zero
            cr |= (cr == 0) ? 1 : 0;
        } else {
            // We did trivial expansion with constant xor, which ensures
            // some bits are non-zero.
            assert(cr != 0);
        }
        return cr;
    }

    inline ResultRow GetResultRowFromHash(Hash h) const {
        // it makes no sense to call this for a retrieval configuration
        assert(Config::kIsFilter);
        Hash a = h * kCoeffAndResultFactor + kCARFadd;
        // The bits here that are *most* independent of Start are the highest
        // order bits (as in Knuth multiplicative hash). To make those the
        // most preferred for use in the result row, we do a bswap here.
        auto rr = static_cast<ResultRow>(rocksdb::EndianSwapValue(a));
        return rr & kResultRowMask;
    }

    // makes no sense for filters
    inline ResultRow GetResultRowFromInput(const Key&) const {
        assert(false);
        return 0;
    }

    inline ResultRow GetResultRowFromInput(const std::pair<Key, ResultRow>& in) const {
        // simple extraction
        return in.second;
    }

    // needed for serialization
    inline uint64_t GetSeed() const {
        return seed_;
    }

protected:
    uint64_t seed_;
    std::conditional_t<kUseMultiplyShiftHash, __uint128_t, DummyData> multiply_,
        add_;
    // For expanding hash: large random prime, congruent 1 modulo 4
    static constexpr Hash kCoeffAndResultFactor =
        static_cast<Hash>(0xc28f82822b650bedULL);
    // additional odd constant to add to the product
    static constexpr Hash kCARFadd = static_cast<Hash>(0xe6165a994ca52647);
    // random-ish data
    static constexpr uint32_t kCoeffXor32 = 0xa6293635U;
    static constexpr uint64_t kCoeffXor64 = 0xc367844a6e52731dU;

    static constexpr ResultRow kResultRowMask = (1ul << kResultBits) - 1;
};


namespace {
// some large random primes for Knuth multiplicative hashing - more than should
// ever be needed (if you need 10 recursive ribbons, you're doing it wrong)
static constexpr std::array<uint64_t, 9> kHashFactors = {
    0x805A57654F4304C3, 0x8E54A33A76D524A9, 0x6F2CC7AB2F1EC959,
    0x9E34124D0877F827, 0x9BE5DFCD6933E6E1, 0xE000C949B34E31FD,
    0xB96A37021B8A394D, 0xFB9B6D45A37C6055, 0xEC63A51B4B89C7BF};
} // namespace

template <typename Config>
class MHCHasher /* not derived */ {
public:
    IMPORT_RIBBON_CONFIG(Config);
    static constexpr bool log = Config::log; // export for BandingAddRange...
    using mhc_t = uint64_t;
    using hash_t = uint64_t;
    // Hash should not be used but make sure is set to something sane
    static_assert(std::is_same_v<Hash, hash_t>, "Hash must be uint64_t");

    MHCHasher() = default;
    explicit MHCHasher(uint64_t seed, unsigned idx) {
        Seed(seed, idx);
    }

    void Seed(uint64_t seed, unsigned idx) {
        seed_ = seed;
        assert(idx < kHashFactors.size());
        fct_ = kHashFactors[idx];

        if constexpr (kUseMultiplyShiftHash) {
            pcg32 random(seed);
            for (auto v : {&multiply_, &add_}) {
                *v = random();
                for (int i = 1; i <= 4; ++i) {
                    *v <<= 32;
                    *v |= random();
                }
            }
        }
    }

    inline hash_t GetHash(mhc_t mhc) const {
        // return middle bits of the product (Knuth Multiplicative Hashing)
        // XXX why middle? why not most significant?
        return static_cast<hash_t>(
            (static_cast<__uint128_t>(mhc) * static_cast<__uint128_t>(fct_)) >> 32);
    }

    inline hash_t GetHash(const std::pair<mhc_t, ResultRow>& p) const {
        assert(!kIsFilter);
        return GetHash(p.first);
    }

    // Get MHC
    inline mhc_t GetMHC(const Key& key) const {
        if constexpr (kUseMultiplyShiftHash && std::is_arithmetic_v<Key>) {
            return (add_ + multiply_ * static_cast<decltype(multiply_)>(key)) >> 64;
        } else {
            return Config::HashFn(key, seed_);
        }
    }

    inline Index GetStart(hash_t h, Index num_starts) const {
        return rocksdb::FastRange64(h, num_starts);
    }

    inline CoeffRow GetCoeffs(hash_t h) const {
        // intentional cast to hash_t
        hash_t a = h * kCoeffAndResultFactor + kCARFadd;
        CoeffRow cr;

        // Expand hash to 64 bits. Almost-trivial hash expansion, favoring
        // roughly equal number of 1's and 0's in result
        uint64_t b = a;
        // b = (b << 32) ^ b ^ kCoeffXor32;

        if constexpr (sizeof(CoeffRow) == 16) {
            __uint128_t c = b;
            c = (c << 64) ^ c ^ kCoeffXor64;
            cr = static_cast<CoeffRow>(c);
        } else {
            cr = static_cast<CoeffRow>(b);
        }

        // Now ensure the value is non-zero
        if constexpr (kFirstCoeffAlwaysOne) {
            cr |= 1;
        } else if constexpr (sizeof(CoeffRow) <= sizeof(hash_t)) {
            // Still have to ensure some bit is non-zero
            cr |= (cr == 0) ? 1 : 0;
        } else {
            // We did trivial expansion with constant xor, which ensures
            // some bits are non-zero.
            assert(cr != 0);
        }
        return cr;
    }

    inline ResultRow GetResultRowFromHash(hash_t h) const {
        // it makes no sense to call this for a retrieval configuration
        assert(Config::kIsFilter);
        uint64_t a = static_cast<mhc_t>(h) * kCoeffAndResultFactor + kCARFadd;
        // The bits here that are *most* independent of Start are the highest
        // order bits (as in Knuth multiplicative hash). To make those the
        // most preferred for use in the result row, we do a bswap here.
        auto rr = static_cast<ResultRow>(rocksdb::EndianSwapValue(a));
        return rr & kResultRowMask;
    }

    // makes no sense for filters
    inline ResultRow GetResultRowFromInput(const Key&) const {
        assert(false);
        return 0;
    }

    inline ResultRow GetResultRowFromInput(const std::pair<Key, ResultRow>& in) const {
        // simple extraction
        return in.second;
    }

    uint64_t GetFactor() const {
        return fct_;
    }

    // needed for serialization
    inline uint64_t GetSeed() const {
        return seed_;
    }

protected:
    uint64_t seed_;
    uint64_t fct_;
    std::conditional_t<kUseMultiplyShiftHash, __uint128_t, DummyData> multiply_,
        add_;
    // For expanding hash: large random prime, congruent 1 modulo 4
    static constexpr Hash kCoeffAndResultFactor =
        static_cast<Hash>(0xc28f82822b650bedULL);
    // additional odd constant to add to the product
    static constexpr Hash kCARFadd = static_cast<Hash>(0xe6165a994ca52647);
    // random-ish data
    static constexpr uint32_t kCoeffXor32 = 0xa6293635U;
    static constexpr uint64_t kCoeffXor64 = 0xc367844a6e52731dU;

    static constexpr ResultRow kResultRowMask = (1ul << kResultBits) - 1;
};

namespace {
template <bool mhc, typename Config>
struct BaseHasherChooser;

template <typename Config>
struct BaseHasherChooser<true, Config> {
    using type = MHCHasher<Config>;
};

template <typename Config>
struct BaseHasherChooser<false, Config> {
    using type = Hasher<Config>;
};

template <typename Config>
using ChooseBaseHasher =
    typename BaseHasherChooser<Config::kUseMHC, Config>::type;
} // namespace

template <typename Config>
class SparseHasher : public ChooseBaseHasher<Config> {
public:
    IMPORT_RIBBON_CONFIG(Config);
    using Super = ChooseBaseHasher<Config>;
    static constexpr bool debug = false;

    // when changing set_bits_, also update thresh_meta_bits in config.hpp!
    static constexpr unsigned set_bits_ = (kCoeffBits == 128)
                                              ? 16
                                              : (kCoeffBits == 16 ? 4 : 8),
                              step_ = kCoeffBits / set_bits_, mask_ = step_ - 1,
                              shift_ = tlx::integer_log2_ceil(step_);

    SparseHasher() : Super() {
        sLOGC(Config::log) << "Sparse Hasher with" << set_bits_ << "set bits, step"
                           << step_ << "mask" << mask_ << "shift" << shift_;
    }
    explicit SparseHasher(uint64_t seed) : Super(seed) {
        sLOGC(Config::log) << "Sparse Hasher with" << set_bits_ << "set bits, step"
                           << step_ << "mask" << mask_ << "shift" << shift_;
    }

    // this has to be fast
    inline Index GetStart(Hash h, Index num_starts) const {
        // byte-aligned
        return rocksdb::FastRangeGeneric(h, num_starts >> shift_) << shift_;
    }

    uint32_t GetCompactHash(Hash h) const {
        Hash a = h * Super::kCoeffAndResultFactor;

        // coeff positions
        const unsigned cbits = set_bits_ * shift_;
        uint32_t result = a & ((1u << cbits) - 1);
        // result
        ResultRow r = GetResultRowFromHash(h);
        assert(r < (1u << (32 - cbits)));
        result |= (r << cbits);

        // check
        assert(GetResultRowFromHash(result) == r);
        assert(GetCoeffs(result) == GetCoeffs(h));

        return result;
    }

    using Super::GetResultRowFromHash;
    inline ResultRow GetResultRowFromHash(uint32_t compact_hash) const {
        LOG0 << "Getting result from compact" << compact_hash;
        return compact_hash >> (set_bits_ * shift_);
    }

    inline CoeffRow GetCoeffs(uint32_t compact_hash) const {
        LOG0 << "Unpacking compact" << compact_hash;
        CoeffRow cr = 0;
        for (unsigned i = 0; i < set_bits_; i++) {
            unsigned pos = (compact_hash & mask_) + step_ * i;
            cr |= static_cast<CoeffRow>(1) << pos;
            compact_hash >>= shift_;
        }
        // compact_hash now contains the ResultRow

        // Now ensure the value is non-zero
        if constexpr (kFirstCoeffAlwaysOne) {
            cr |= 1;
        } else {
            // Still have to ensure some bit is non-zero
            cr |= (cr == 0) ? 1 : 0;
        }
        return cr;
    }

    inline CoeffRow GetCoeffs(Hash h) const {
        Hash a = h * Super::kCoeffAndResultFactor;
        CoeffRow cr = 0;

        for (unsigned i = 0; i < set_bits_; i++) {
            unsigned pos = (a & mask_) + step_ * i;
            cr |= static_cast<CoeffRow>(1) << pos;
            a >>= shift_;
        }
        // LOG << "GetCoeffs(0x" << std::hex << h << ") = 0x" << (uint64_t)cr
        //     << std::dec;

        // Now ensure the value is non-zero
        if constexpr (kFirstCoeffAlwaysOne) {
            cr |= 1;
        } else {
            // Still have to ensure some bit is non-zero
            cr |= (cr == 0) ? 1 : 0;
        }
        return cr;
    }
};

namespace {
template <bool, typename>
struct HasherChooser;

template <typename Config>
struct HasherChooser<true, Config> {
    using type = SparseHasher<Config>;
};

template <typename Config>
struct HasherChooser<false, Config> {
    using type = ChooseBaseHasher<Config>;
};

} // namespace

template <typename Config>
using ChooseHasher = typename HasherChooser<Config::kSparseCoeffs, Config>::type;


// can't do std::conditional_t<kUseMHC, typename Hasher::mhc_t, xyz> because
// mhc_t isn't defined unless kUseMHC is set so need to work around that
template <typename Hasher, typename = void>
struct HashTraits {
    using mhc_or_key_t = typename Hasher::Key;
    using hash_t = typename Hasher::Hash;
};

template <typename Hasher>
struct HashTraits<Hasher, std::void_t<typename Hasher::mhc_t>> {
    using mhc_or_key_t = typename Hasher::mhc_t;
    using hash_t = typename Hasher::hash_t;
};

} // namespace ribbon
