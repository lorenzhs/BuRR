## BuRR: Bumped Ribbon Retrieval (and Filters)

BuRR is a static retrieval and approximate membership query data structure with extremely low overhead and fast queries. Our paper introducing BuRR, ["Fast Succinct Retrieval and Approximate Membership Using Ribbon"](https://drops.dagstuhl.de/opus/volltexte/2022/16538/), won the best paper award at the 20th International Symposium on Experimental Algorithms 2022 and is available in full as an open-access publication. For additional details and measurements, please [refer to the preprint on arxiv.org](https://arxiv.org/abs/2109.01892).

"Retrieval" means that you have a set of key-value pairs that you want to represent very compactly.  The difference to a hash table is that the data structure may return garbage values when queried with keys not in the set.  It's also typically far more compact: a BuRR representation of the data is typically not more than 0.1-0.5% larger than the *values* it represents (no keys are stored).

"Approximate Membership Query" (AMQ) means that you have a set and want to check whether an element is likely in the set.  Some well-known examples are Bloom filters, Xor filters, Cuckoo filters, and Quotient filters.  A query for an item in the set will always return `true`, while a query for an item *not* in the set *usually* returns `false`, but may return `true` with some small probability known as the *false-positive probability f*.  The space required to represent this set depends on *f*, and there is a lower bound of *log2(1/f)* bits per item of the set.  Classic Bloom filters use *1.44 log2(1/f)* bits per key, meaning they use 44% more space than needed.  With Xor filters, overhead on the order of 20% can be achieved, and Xor+ filters reduce this to approximately 10% for very small values of *f*.  BuRR can achieve overheads as low as 0.1%, and even configurations that trade overhead for speed achieve overheads of below 0.5%.

## Building and running

Make sure to fetch all submodules with `git submodule update --init --recursive`, then type `make bench` to compile a benchmark runner that includes a wide range of configurations, or `make tests` to compile the test suite.  The scripts used in the evaluation are located in the `scripts` folder.  You may also want to refer to [the fastfilter_cpp repository](https://github.com/lorenzhs/fastfilter_cpp) for a comparison to other filter data structures and more benchmarks used in our paper.

The library can be used similar to the following example:

```cpp
#include "ribbon.hpp"

std::vector<std::pair<uint64_t, uint8_t>> data;
data.emplace_back(0xabc, 0); // Key has to be a hash value
data.emplace_back(0xdef, 1);

using namespace ribbon;
using Config = FastRetrievalConfig</* result bits */ 1, uint64_t>;
using RibbonT = ribbon_filter</* depth */ 2, Config>;
RibbonT retrievalDs(data.size(), /* overload factor */ 0.965, /* seed */ 42);

// Construction
retrievalDs.AddRange(data.begin(), data.end());
retrievalDs.BackSubst();

// Queries
std::cout << (int) retrievalDs.QueryRetrieval(0xabc) << std::endl; // 0
std::cout << (int) retrievalDs.QueryRetrieval(0xdef) << std::endl; // 1
```

## Enhancements

- You can find a parallel implementation on the [`parallel` branch](https://github.com/lorenzhs/BuRR/tree/parallel) and its brief announcement paper on [arXiv](https://arxiv.org/abs/2411.12365).
- [SimpleRibbon](https://github.com/ByteHamster/SimpleRibbon) is a wrapper around BuRR that offers cmake support for setting up dependencies, as well as a non-header library to reduce compile times.

## Citation

If you use BuRR in the context of an academic publication, we ask that you please cite our paper:

```bibtex
@inproceedings{BuRR2022,
    author={Peter C. Dillinger, Lorenz Hübschle-Schneider, Peter Sanders, and Stefan Walzer},
    title={Fast Succinct Retrieval and Approximate Membership using Ribbon},
    booktitle={20th International Symposium on Experimental Algorithms (SEA 2022)},
    pages={4:1--4:20},
    year={2022},
    doi={10.4230/LIPIcs.SEA.2022.4}
}
```

## License

BuRR is licensed under the Apache 2.0 license. Copyright is held by Lorenz Hübschle-Schneider and Facebook, Inc.  It is based on [Peter C. Dillinger's implementation of Standard Ribbon](https://github.com/pdillinger/fastfilter_cpp/tree/dev/src/ribbon), which is copyright Facebook, Inc. and also licensed under the Apache 2.0 license.
