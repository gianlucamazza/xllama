// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "xllama/model_provision.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <utility>

namespace xllama {

// A coroutine may resume on another thread: ownership cannot be a mutex lock.
class ModelWriteGate {
  public:
    class Permit {
      public:
        Permit() = default;
        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;
        Permit(Permit&& other) noexcept : m_gate(std::exchange(other.m_gate, nullptr)) {}
        Permit& operator=(Permit&& other) noexcept {
            if (this != &other) {
                release();
                m_gate = std::exchange(other.m_gate, nullptr);
            }
            return *this;
        }
        ~Permit() {
            release();
        }
        bool owns_lock() const {
            return m_gate != nullptr;
        }
        void release() {
            if (m_gate) {
                m_gate->m_busy.store(false, std::memory_order_release);
                m_gate = nullptr;
            }
        }

      private:
        friend class ModelWriteGate;
        explicit Permit(ModelWriteGate* gate) : m_gate(gate) {}
        ModelWriteGate* m_gate = nullptr;
    };

    Permit try_acquire() {
        bool expected = false;
        return m_busy.compare_exchange_strong(expected, true, std::memory_order_acquire)
                   ? Permit(this)
                   : Permit();
    }

  private:
    std::atomic<bool> m_busy{false};
};

// Shared by network downloads, USB import, and explicit rollback. Rejected
// writers cannot invalidate metadata, create directories, or remove old files.
class ModelWriteOperation {
  public:
    ModelWriteOperation(ModelWriteGate& gate, std::filesystem::path root)
        : m_permit(gate.try_acquire()), m_root(std::move(root)) {}
    bool owns_lock() const {
        return m_permit.owns_lock();
    }
    void release() {
        m_permit.release();
    }

    bool begin(std::error_code& ec) {
        if (!owns_lock()) {
            ec = std::make_error_code(std::errc::device_or_resource_busy);
            return false;
        }
        std::filesystem::create_directories(m_root, ec);
        if (ec)
            return false;
        std::filesystem::remove(m_root / ".complete", ec);
        return !ec;
    }

    // Call only after every expected file has been verified/promoted. An empty
    // list is useful for rollback/USB: publish the marker without pruning.
    bool complete(const std::vector<std::wstring>& expected_files, std::error_code& ec) {
        if (!owns_lock()) {
            ec = std::make_error_code(std::errc::device_or_resource_busy);
            return false;
        }
        if (!expected_files.empty()) {
            std::vector<std::wstring> keep;
            for (const auto& file : expected_files)
                keep.push_back(normalize_model_path(file));
            std::filesystem::directory_iterator it(m_root, ec), end;
            while (!ec && it != end) {
                const auto name = normalize_model_path(it->path().filename().wstring());
                if (it->is_regular_file(ec) && name.size() > 5 &&
                    name.compare(name.size() - 5, 5, L".gguf") == 0 && name != L"adapter.gguf" &&
                    std::find(keep.begin(), keep.end(), name) == keep.end())
                    std::filesystem::remove(it->path(), ec);
                if (!ec)
                    it.increment(ec);
            }
            if (ec)
                return false;
        }
        std::ofstream marker(m_root / ".complete", std::ios::binary | std::ios::trunc);
        marker << "ok";
        marker.close();
        if (!marker) {
            ec = std::make_error_code(std::errc::io_error);
            return false;
        }
        ec.clear();
        return true;
    }

  private:
    ModelWriteGate::Permit m_permit;
    std::filesystem::path m_root;
};

} // namespace xllama
