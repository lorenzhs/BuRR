#pragma once

#include <tlx/math/bswap.hpp>

#include <string>
#include <stdexcept>

namespace ribbon {

class config_error : std::runtime_error {
public:
    config_error(const std::string& msg) : std::runtime_error(msg) {}
};

class parse_error : std::runtime_error {
public:
    parse_error(const std::string& msg) : std::runtime_error(msg) {}
};

class file_open_error : std::runtime_error {
public:
    file_open_error(const std::string& msg) : std::runtime_error(msg) {}
};

static inline __uint128_t bswap128(const __uint128_t& v) {
    #if defined(__GNUC__) && defined(__has_builtin)
        // Cannot use "&&" directly: Macros do not use short-circuiting
        #if  __has_builtin(__builtin_bswap128)
            #define BURR_USE_BSWAP
        #endif
    #endif
    #ifdef BURR_USE_BSWAP
        return __builtin_bswap128(v);
    #else
        __uint128_t lo = tlx::bswap64(v & 0xFFFFFFFFFFFFFFFFULL);
        return (lo << 64) | tlx::bswap64(v >> 64);
    #endif
}

template <typename T>
bool bswap_generic(T& val) {
    if constexpr (sizeof(T) == 1) {
        return true;
    } else if constexpr (sizeof(T) == 2) {
        val = tlx::bswap16(val);
    } else if constexpr (sizeof(T) == 4) {
        val = tlx::bswap32(val);
    } else if constexpr (sizeof(T) == 8) {
        val = tlx::bswap64(val);
    } else if constexpr (sizeof(T) == 16) {
        val = bswap128(val);
    } else {
        return false;
    }
    return true;
}

// if this is not the case, something is very wrong anyways...
template <typename T>
bool bswap_type_supported() {
    return sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
           sizeof(T) == 8 || sizeof(T) == 16;
}

} // namespace ribbon
