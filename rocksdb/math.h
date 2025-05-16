//  Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <assert.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cstdint>
#include <type_traits>

namespace rocksdb {

// Number of low-order zero bits before the first 1 bit. Undefined for 0.
template <typename T>
inline int CountTrailingZeroBits(T v) {
    static_assert(std::is_integral<T>::value || std::is_same_v<T, __uint128_t>,
                  "non-integral type");
    assert(v != 0);
#ifdef _MSC_VER
    static_assert(sizeof(T) <= sizeof(uint64_t), "type too big");
    unsigned long tz = 0;
    if (sizeof(T) <= sizeof(uint32_t)) {
        _BitScanForward(&tz, static_cast<uint32_t>(v));
    } else {
#if defined(_M_X64) || defined(_M_ARM64)
        _BitScanForward64(&tz, static_cast<uint64_t>(v));
#else
        _BitScanForward(&tz, static_cast<uint32_t>(v));
        if (tz == 0) {
            _BitScanForward(
                &tz, static_cast<uint32_t>(static_cast<uint64_t>(v) >> 32));
            tz += 32;
        }
#endif
    }
    return static_cast<int>(tz);
#else
    if constexpr (sizeof(T) <= sizeof(unsigned int)) {
        return __builtin_ctz(static_cast<unsigned int>(v));
    } else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
        return __builtin_ctzl(static_cast<unsigned long>(v));
    } else if constexpr (sizeof(T) <= sizeof(unsigned long long)) {
        return __builtin_ctzll(static_cast<unsigned long long>(v));
    } else if constexpr (sizeof(T) == 16) {
        const int lower = CountTrailingZeroBits(static_cast<uint64_t>(v));
        return lower < 64
                   ? lower
                   : 64 + CountTrailingZeroBits(static_cast<uint64_t>(v >> 64));
    } else {
        static_assert(sizeof(T) <= 16, "type too big");
    }
#endif
}

// this is not taken from rocksdb, but based on CountTrailingZeroBits
// Note: this has only been tested on gcc and clang
// Number of high-order zero bits before the first 1 bit. Undefined for 0.
template <typename T>
inline int CountLeadingZeroBits(T v) {
    static_assert(std::is_integral<T>::value || std::is_same_v<T, __uint128_t>,
                  "non-integral type");
    assert(v != 0);
    constexpr unsigned int num_bits = sizeof(v) * 8U;
#ifdef _MSC_VER
    static_assert(sizeof(T) <= sizeof(uint64_t), "type too big");
    unsigned long tz = 0;
    if (sizeof(T) <= sizeof(uint32_t)) {
        _BitScanReverse(&tz, static_cast<uint32_t>(v));
        tz = num_bits - tz - 1;
    } else {
#if defined(_M_X64) || defined(_M_ARM64)
        _BitScanReverse64(&tz, static_cast<uint64_t>(v));
        tz = num_bits - tz - 1;
#else
        uint32_t upper = static_cast<uint32_t>(static_cast<uint64_t>(v) >> 32);
        if (upper) {
            _BitScanReverse(&tz, upper);
            tz = num_bits - tz - 33;
        } else {
            _BitScanReverse(&tz, static_cast<uint32_t>(v));
            tz = num_bits - tz - 1;
        }
#endif
    }
    return static_cast<int>(tz);
#else
    if constexpr (sizeof(T) <= sizeof(unsigned int)) {
        return __builtin_clz(static_cast<unsigned int>(v)) - (sizeof(unsigned int) * 8U - num_bits);
    } else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
        return __builtin_clzl(static_cast<unsigned long>(v)) - (sizeof(unsigned long) * 8U - num_bits);
    } else if constexpr (sizeof(T) <= sizeof(unsigned long long)) {
        return __builtin_clzll(static_cast<unsigned long long>(v)) - (sizeof(unsigned long long) * 8U - num_bits);
    } else if constexpr (sizeof(T) == 16) {
        const uint64_t upper = static_cast<uint64_t>(v >> 64);
        return upper ? CountLeadingZeroBits(upper) : CountLeadingZeroBits(static_cast<uint64_t>(v)) + 64;
    } else {
        static_assert(sizeof(T) <= 16, "type too big");
    }
#endif
}

template <typename T>
inline int BitParity(T v) {
    static_assert(std::is_integral<T>::value || std::is_same_v<T, __uint128_t>,
                  "non-integral type");
#ifdef _MSC_VER
    // bit parity == oddness of popcount
    return BitsSetToOne(v) & 1;
#else
    if constexpr (sizeof(T) <= sizeof(unsigned int)) {
        // On any sane systen, potential sign extension here won't change parity
        return __builtin_parity(static_cast<unsigned int>(v));
    } else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
        return __builtin_parityl(static_cast<unsigned long>(v));
    } else if constexpr (sizeof(T) <= sizeof(unsigned long long)) {
        return __builtin_parityll(static_cast<unsigned long long>(v));
    } else if constexpr (sizeof(T) == 16) {
        return __builtin_parityll(static_cast<uint64_t>(v)) ^
               __builtin_parityll(static_cast<uint64_t>(v >> 64));
    } else {
        static_assert(sizeof(T) <= sizeof(unsigned long long), "type too big");
    }
#endif
}

} // namespace rocksdb
