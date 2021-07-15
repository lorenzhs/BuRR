//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// Encoding independent of machine byte order:
// * Fixed-length numbers are encoded with least-significant byte first
//   (little endian, native order on Intel and others)
// * In addition we support variable length "varint" encoding
// * Strings are encoded prefixed by their length in varint format

#pragma once
#include <string>

namespace rocksdb {

// Swaps between big and little endian. Can be used to in combination
// with the little-endian encoding/decoding functions to encode/decode
// big endian.
template <typename T>
inline T EndianSwapValue(T v) {
  static_assert(std::is_integral<T>::value, "non-integral type");

#ifdef _MSC_VER
  if (sizeof(T) == 2) {
    return static_cast<T>(_byteswap_ushort(static_cast<uint16_t>(v)));
  } else if (sizeof(T) == 4) {
    return static_cast<T>(_byteswap_ulong(static_cast<uint32_t>(v)));
  } else if (sizeof(T) == 8) {
    return static_cast<T>(_byteswap_uint64(static_cast<uint64_t>(v)));
  }
#else
  if (sizeof(T) == 2) {
    return static_cast<T>(__builtin_bswap16(static_cast<uint16_t>(v)));
  } else if (sizeof(T) == 4) {
    return static_cast<T>(__builtin_bswap32(static_cast<uint32_t>(v)));
  } else if (sizeof(T) == 8) {
    return static_cast<T>(__builtin_bswap64(static_cast<uint64_t>(v)));
  }
#endif
  // Recognized by clang as bswap, but not by gcc :(
  T ret_val = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    ret_val |= ((v >> (8 * i)) & 0xff) << (8 * (sizeof(T) - 1 - i));
  }
  return ret_val;
}

inline void PutFixed64(std::string* dst, uint64_t value) {
    dst->append(const_cast<const char*>(reinterpret_cast<char*>(&value)),
      sizeof(value));
}

/**************************************************
 * what follows was originally from coding_lean.h *
 **************************************************/

// Lower-level versions of Put... that write directly into a character buffer
// REQUIRES: dst has enough space for the value being written
// -- Implementation of the functions declared above
inline void EncodeFixed16(char* buf, uint16_t value) {
    memcpy(buf, &value, sizeof(value));
}

inline void EncodeFixed32(char* buf, uint32_t value) {
    memcpy(buf, &value, sizeof(value));
}

inline void EncodeFixed64(char* buf, uint64_t value) {
    memcpy(buf, &value, sizeof(value));
}

// Lower-level versions of Get... that read directly from a character buffer
// without any bounds checking.

inline uint16_t DecodeFixed16(const char* ptr) {
    // Load the raw bytes
    uint16_t result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
}

inline uint32_t DecodeFixed32(const char* ptr) {
    // Load the raw bytes
    uint32_t result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
}

inline uint64_t DecodeFixed64(const char* ptr) {
    // Load the raw bytes
    uint64_t result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
}

}  // namespace rocksdb
