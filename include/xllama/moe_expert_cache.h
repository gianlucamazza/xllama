// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "ggml-backend.h"
#include "xllama/moe_streaming.h"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace xllama {

// Bound to one immutable open file for the cache lifetime. read() is synchronous:
// it writes at most bytes, and even on exception no operation may access dst after return.
struct MoeExpertReader {
    virtual ~MoeExpertReader() = default;
    virtual size_t read(uint64_t offset, unsigned char* dst, size_t bytes) = 0;
};

// Only contiguous matrices, with experts packed consecutively. The table index
// is the tensor identity. The caller must verify packed GGUF strides and file_bytes
// against its held reader. The cache snapshots the table and never rebinds files.
struct MoeExpertTensor {
    MoeRole role;
    ggml_type type;
    uint64_t columns, rows, experts, offset;
};

struct MoeCacheCounters {
    uint64_t requests = 0, hits = 0, requested_bytes = 0, hit_bytes = 0;
    uint64_t logical_read_bytes = 0, evictions = 0;
};

// Threading contract: single coordinator thread. Pin and owner counts are plain
// integers, not atomics, so construction, acquire(), every Lease release (reset(),
// destructor, move-assignment over a held lease) and cache destruction must run on
// the thread that constructed the cache. Debug builds assert this; release builds
// do not check it. Workers may only call Lease::data()/size() on leases the
// coordinator holds, across a barrier that ends before the coordinator releases them.
// Native integration (#480) must keep acquisition and release on its coordinator.
// The reader itself must outlive the cache.
// Keep the memory reservation until the cache AND its last lease have been destroyed.
class MoeExpertCache {
    struct Tensor {
        MoeExpertTensor descriptor;
        size_t bytes;
    };
    struct Slot {
        size_t tensor = 0, pins = 0, rank = 0;
        uint64_t expert = 0;
        bool valid = false;
    };
    struct Storage {
        std::unique_ptr<Tensor[]> tensors;
        std::unique_ptr<Slot[]> slots;
        std::unique_ptr<unsigned char[]> allocation;
        unsigned char* arena = nullptr;
        size_t stride = 0, owners = 1; // cache plus every live lease
#ifndef NDEBUG
        std::thread::id coordinator = std::this_thread::get_id();
#endif
        void assert_coordinator() const noexcept {
#ifndef NDEBUG
            assert(coordinator == std::this_thread::get_id() &&
                   "MoeExpertCache: pins and owners are coordinator-thread only");
#endif
        }
        void release() noexcept {
            assert_coordinator();
            if (--owners == 0)
                delete this;
        }
    };

  public:
    class Lease {
      public:
        Lease() = default;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept
            : storage_(std::exchange(other.storage_, nullptr)), index_(other.index_) {}
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                reset();
                storage_ = std::exchange(other.storage_, nullptr);
                index_ = other.index_;
            }
            return *this;
        }
        ~Lease() {
            reset();
        }

        const unsigned char* data() const noexcept {
            return storage_ ? storage_->arena + index_ * storage_->stride : nullptr;
        }
        size_t size() const noexcept {
            return storage_ ? storage_->tensors[storage_->slots[index_].tensor].bytes : 0;
        }
        void reset() noexcept {
            if (storage_) {
                storage_->assert_coordinator();
                --storage_->slots[index_].pins;
                std::exchange(storage_, nullptr)->release();
            }
        }

      private:
        friend class MoeExpertCache;
        Lease(Storage* storage, size_t index) noexcept : storage_(storage), index_(index) {
            storage_->assert_coordinator();
            ++storage_->owners;
            ++storage_->slots[index_].pins;
        }
        Storage* storage_ = nullptr;
        size_t index_ = 0;
    };

    // Requested storage, including fixed objects, immutable descriptors, slots
    // and aligned payloads. Reads go directly into an unpublished slot (zero
    // extra staging). Allocator overhead and reader-owned buffers are external.
    // Allocation failures propagate as std::bad_alloc and release partial allocations.
    static size_t required_bytes(size_t tensor_count, size_t slots, size_t slot_bytes) {
        if (!tensor_count || !slots || !slot_bytes)
            throw std::invalid_argument("empty expert cache");
        size_t total = sizeof(MoeExpertCache) + sizeof(Storage);
        total = add(total, multiply(tensor_count, sizeof(Tensor)));
        total = add(total, multiply(slots, sizeof(Slot)));
        return add(total, arena_bytes(slots, aligned_stride(slot_bytes)));
    }

    MoeExpertCache(MoeExpertReader& reader, uint64_t file_bytes, const MoeExpertTensor* tensors,
                   size_t tensor_count, size_t slots, size_t slot_bytes, size_t budget)
        : reader_(reader), tensor_count_(tensor_count), slot_count_(slots) {
        if (!tensors || required_bytes(tensor_count, slots, slot_bytes) > budget)
            throw std::invalid_argument("invalid expert cache table or byte budget");
        auto storage = std::make_unique<Storage>();
        storage->tensors = std::make_unique<Tensor[]>(tensor_count);
        for (size_t i = 0; i < tensor_count; ++i) {
            const auto& desc = tensors[i];
            const auto bytes = slice_bytes(desc);
            if (!desc.experts || desc.offset > file_bytes ||
                desc.experts > (file_bytes - desc.offset) / bytes)
                throw std::out_of_range("expert tensor exceeds file");
            if (bytes > slot_bytes)
                throw std::length_error("expert slice exceeds slot capacity");
            storage->tensors[i] = {desc, static_cast<size_t>(bytes)};
        }
        storage->slots = std::make_unique<Slot[]>(slots);
        for (size_t i = 0; i < slots; ++i)
            storage->slots[i].rank = i;
        storage->stride = aligned_stride(slot_bytes);
        size_t space = arena_bytes(slots, storage->stride);
        storage->allocation.reset(new unsigned char[space]);
        void* base = storage->allocation.get();
        storage->arena = static_cast<unsigned char*>(
            std::align(cpu_alignment(), slots * storage->stride, base, space));
        assert(storage->arena != nullptr);
        storage_ = storage.release();
    }

    MoeExpertCache(const MoeExpertCache&) = delete;
    MoeExpertCache& operator=(const MoeExpertCache&) = delete;
    ~MoeExpertCache() {
        storage_->release();
    }

    const MoeCacheCounters& counters() const noexcept {
        return counters_;
    }

    Lease acquire(size_t tensor_id, uint64_t expert_id, const std::atomic_bool* stop = nullptr) {
        storage_->assert_coordinator();
        if (reading_)
            throw std::logic_error("reentrant expert cache acquisition");
        if (tensor_id >= tensor_count_)
            throw std::out_of_range("unknown routed tensor");
        const Tensor& tensor = storage_->tensors[tensor_id];
        if (expert_id >= tensor.descriptor.experts)
            throw std::out_of_range("unknown expert");
        check_stop(stop);
        if (storage_->owners == std::numeric_limits<size_t>::max())
            throw std::overflow_error("too many expert leases");
        count(counters_.requests, 1);
        count(counters_.requested_bytes, tensor.bytes);

        size_t victim = slot_count_;
        for (size_t i = 0; i < slot_count_; ++i) {
            auto& slot = storage_->slots[i];
            if (slot.valid && slot.tensor == tensor_id && slot.expert == expert_id) {
                count(counters_.hits, 1);
                count(counters_.hit_bytes, tensor.bytes);
                touch(i);
                return Lease(storage_, i);
            }
            if (!slot.pins && (victim == slot_count_ || precedes(slot, storage_->slots[victim])))
                victim = i;
        }
        if (victim == slot_count_)
            throw std::runtime_error("all expert slots are leased");

        auto& slot = storage_->slots[victim];
        if (slot.valid)
            count(counters_.evictions, 1);
        slot.valid = false;
        // Extent admission above proves both operations fit in uint64_t.
        const auto offset = tensor.descriptor.offset + expert_id * tensor.bytes;
        reading_ = true;
        size_t returned;
        try {
            returned =
                reader_.read(offset, storage_->arena + victim * storage_->stride, tensor.bytes);
        } catch (...) {
            reading_ = false;
            throw;
        }
        reading_ = false;
        if (returned <= tensor.bytes)
            count(counters_.logical_read_bytes, returned);
        if (returned != tensor.bytes)
            throw std::runtime_error("incomplete or invalid expert read");
        check_stop(stop);
        slot.tensor = tensor_id;
        slot.expert = expert_id;
        slot.valid = true;
        touch(victim);
        return Lease(storage_, victim);
    }

  private:
    static uint64_t slice_bytes(const MoeExpertTensor& t) {
        switch (t.role) {
        case MoeRole::gate_exps:
        case MoeRole::up_exps:
        case MoeRole::down_exps:
        case MoeRole::gate_up_exps:
            break;
        default:
            throw std::invalid_argument("invalid routed tensor role");
        }
        if (t.type != GGML_TYPE_F32 && t.type != GGML_TYPE_Q4_K && t.type != GGML_TYPE_Q6_K)
            throw std::invalid_argument("unsupported expert quantization");
        const auto block = static_cast<uint64_t>(ggml_blck_size(t.type));
        if (!t.rows || !t.columns || t.columns % block != 0)
            throw std::invalid_argument("invalid packed expert shape");
        return multiply(multiply(t.columns / block, static_cast<uint64_t>(ggml_type_size(t.type))),
                        t.rows);
    }
    template <class T> static T multiply(T a, T b) {
        if (a && b > std::numeric_limits<T>::max() / a)
            throw std::overflow_error("expert cache size overflow");
        return a * b;
    }
    static size_t add(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a)
            throw std::overflow_error("expert cache budget overflow");
        return a + b;
    }
    static size_t cpu_alignment() {
        return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    }
    static size_t aligned_stride(size_t bytes) {
        const size_t alignment = cpu_alignment();
        const auto remainder = bytes % alignment;
        return remainder ? add(bytes, alignment - remainder) : bytes;
    }
    static size_t arena_bytes(size_t slots, size_t stride) {
        // GGML checks the base of every mapped CPU buffer. Reserve the worst-case leading pad
        // as well as the stride padding, even when the allocator already aligns the first slot.
        return add(multiply(slots, stride), cpu_alignment() - 1);
    }
    static void count(uint64_t& value, uint64_t amount) noexcept {
        constexpr auto max = std::numeric_limits<uint64_t>::max();
        value = amount > max - value ? max : value + amount;
    }
    static void check_stop(const std::atomic_bool* stop) {
        if (stop && stop->load(std::memory_order_relaxed))
            throw std::runtime_error("expert read cancelled");
    }
    static bool precedes(const Slot& candidate, const Slot& selected) noexcept {
        if (candidate.valid != selected.valid)
            return !candidate.valid;
        return candidate.rank > selected.rank;
    }
    // Ranks stay a permutation of [0, slot_count). No timestamp wraparound.
    void touch(size_t index) noexcept {
        auto& slot = storage_->slots[index];
        for (size_t i = 0; i < slot_count_; ++i)
            if (storage_->slots[i].rank < slot.rank)
                ++storage_->slots[i].rank;
        slot.rank = 0;
    }

    MoeExpertReader& reader_;
    size_t tensor_count_, slot_count_;
    Storage* storage_;
    MoeCacheCounters counters_;
    bool reading_ = false;
};

} // namespace xllama
