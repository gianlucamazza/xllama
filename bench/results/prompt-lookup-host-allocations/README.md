# GGUF prompt-lookup allocation replay

Measured cumulative allocations in the normal Linux CLI while reusing the accepted token history. The comparison uses xLlama base `62a577e00859ec381463707f3baf1430edda5ddc`, candidate code commit `f8085fed642ec4bf900d1c3b0c5c82bdb656d282` and their shared llama.cpp gitlink `d81235049384534c167caea52b85a694f6103d14`. The candidate Git blobs for the two changed runtime files are `b0b995d50caa429c060c8b6fc2a7cfd3f95b1768` (decode_loop.h) and `e27f7a305037cfb965dd76d7cbe15ce28d0acaf2` (inference.cpp). It changes buffer ownership in the existing GGUF decode loop. The smoke test covers accounting when decode appends to the input buffer.

| Prompt and output                    | Lookup | Fewer ordinary C++ allocations per process | Fewer bytes cumulatively requested | Accepted / drafted in both arms |
| ------------------------------------ | ------ | -----------------------------------------: | ---------------------------------: | ------------------------------: |
| CPU fixture, 13 / 64 tokens          | Off    |                                          1 |                                 52 |                           0 / 0 |
| CPU fixture, 13 / 64 tokens          | On     |                                         49 |                              6,088 |                         18 / 34 |
| CPU fixture, 769 / 64 tokens         | Off    |                                          1 |                              3,076 |                           0 / 0 |
| CPU fixture, 769 / 64 tokens         | On     |                                         64 |                            108,688 |                         30 / 30 |
| LFM2.5-230M QAD Q4_0, 14 / 24 tokens | Off    |                                          1 |                                 56 |                           0 / 0 |
| LFM2.5-230M QAD Q4_0, 14 / 24 tokens | On     |                                         21 |                              1,796 |                           4 / 6 |

Each allocation count was reproduced on two fresh launches per arm and configuration. All 76 CLI runs succeeded. For each prompt, greedy stdout bytes were identical across revisions, repetitions and lookup settings. The longer synthetic prompt exceeds the fixture's nominal training window. It exercises long buffer ownership. Its generated text carries no quality evidence.

[Raw per-process measurements](runs.csv) include unprofiled timings. Runs took place on a shared Linux x86-64 VM, AMD EPYC 9V74, Debian 13, kernel 6.18.57, GCC 14.2.0, CMake Debug (`-O0`), portable AVX2 with OpenMP. All requests used one CPU thread, CPU-only GGUF, the default KV type and batch settings, and greedy decoding. The comparison measures the number and size of ordinary C++ allocation requests over the whole CLI lifetime. Freed bytes still count toward that total. C allocation functions, aligned allocation, mapped pages and residency require separate instruments. The shared host and limited uninstrumented repetitions, including one per real-model arm, give no reliable speed estimate. Board timing, memory and energy require a packaged console campaign.

## Reproduce

1. Build the base and candidate with the same compiler, CPU features and configuration. The recorded run rebuilt every translation unit including `decode_loop.h` against each revision and relinked the matching unchanged objects and pinned libraries. Verify the [CPU fixture and golden](../../../scripts/run-cpu-smoke.py) and the catalogue [LFM2.5-230M QAD Q4_0 entry](../../../uwp/models/manifest.json). The fixture SHA-256 is `ea5812a29a3260b9d489fadb272e1dddedadaf291d5fcf839b944ee440311881`. The catalogue GGUF SHA-256 is `e75f83268de11b2a1bcfab5f3b5c5c0c97569ddbbc0990aad88437e45b8ba292`.
2. Start a fresh process for each arm. For each workload below run both `--greedy` and `--greedy --prompt-lookup`. Alternate the base/candidate order each repetition. The two initial repetitions used the allocator counter below. The following 7, 5 and 1 repetitions respectively used the normal allocator. Hash stdout with its trailing newline. Read `SPEC_STATS` from stderr.
3. Use `-t 1 -c 128 -n 64 -p abcdabcdabcd` for the short fixture. For the long fixture use `-c 1024 -n 64 -t 1` and `-p` with exactly 192 repetitions of `abcd`. For LFM2.5-230M use `-c 512 -n 24 -t 1 -p 'Implement fibonacci in Python. def fibonacci(n):'`. Every command also needs `-m` with the corresponding model path.

The host counter used the executable's ordinary scalar throwing operator new. Save this Linux-only probe as `count.cpp` and build with `g++ -std=c++17 -O2 -shared -fPIC count.cpp -o count.so`. Prefix only the instrumented CLI processes with `LD_PRELOAD="$PWD/count.so"`.

```cpp
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>

static std::atomic<unsigned long long> calls{0}, bytes{0};

void* operator new(std::size_t n) {
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    calls.fetch_add(1, std::memory_order_relaxed);
    bytes.fetch_add(n, std::memory_order_relaxed);
    return p;
}

__attribute__((destructor)) static void counts() {
    std::fprintf(stderr, "ORDINARY_NEW calls=%llu bytes=%llu\n", calls.load(), bytes.load());
}
```
