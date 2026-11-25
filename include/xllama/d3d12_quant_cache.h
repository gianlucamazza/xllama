// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Identity-keyed cache of CPU-quantized activations shared by matmuls in one
// d3d12 graph submission. Looked up by tensor owner, data pointer, column
// stride, type and row size — not by address alone. The graph allocator reuses
// memory, so two different tensors can occupy the same address.
// Header-only, WinRT- and D3D12-free, host-testable.
#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace xllama {

struct D3d12QuantKey {
    const void* owner = nullptr;
    const void* data = nullptr;
    std::size_t stride = 0;
    ggml_type type = GGML_TYPE_F32;
    std::uint32_t row = 0;

    bool operator==(const D3d12QuantKey& o) const {
        return owner == o.owner && data == o.data && stride == o.stride && type == o.type &&
               row == o.row;
    }
};

struct D3d12QuantHit {
    std::size_t off = 0;
    std::uint32_t row = 0;
    int ncols = 0;
};

class D3d12QuantCache {
  public:
    std::optional<D3d12QuantHit> find(const D3d12QuantKey& key, int ncols) const {
        for (const Entry& e : entries_) {
            if (e.key == key && e.ncols >= ncols)
                return D3d12QuantHit{e.off, e.key.row, e.ncols};
        }
        return std::nullopt;
    }

    D3d12QuantHit insert(const D3d12QuantKey& key, std::size_t off, int ncols) {
        entries_.push_back({key, off, ncols});
        return {off, key.row, ncols};
    }

    void clear() {
        entries_.clear();
    }

  private:
    struct Entry {
        D3d12QuantKey key;
        std::size_t off = 0;
        int ncols = 0;
    };
    std::vector<Entry> entries_;
};

} // namespace xllama
