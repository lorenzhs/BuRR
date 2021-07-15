//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
#pragma once

#include <time.h>

namespace rocksdb {

// a nano second precision stopwatch
class StopWatchNano {
public:
  explicit StopWatchNano(bool auto_start = false) : start_(0) {
    if (auto_start) {
      Start();
    }
  }

  uint64_t NowNanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
  }

  void Start() { start_ = NowNanos(); }

  uint64_t ElapsedNanos(bool reset = false) {
    auto now = NowNanos();
    auto elapsed = now - start_;
    if (reset) {
      start_ = now;
    }
    return elapsed;
  }

  uint64_t ElapsedNanosSafe(bool reset = false) {
    return ElapsedNanos(reset);
  }

private:
  uint64_t start_;
};

} // namespace rocksdb
