// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include "xllama/model_file_plan.h"

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

    // Observer for routing: a held permit means some writer is mid-download
    // or import. Does not acquire.
    bool busy() const {
        return m_busy.load(std::memory_order_acquire);
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

    // A retry replaces the staging file instead of resuming its contents.
    bool discard_partial(const std::wstring& filename, std::error_code& ec) {
        if (!owns_lock()) {
            ec = std::make_error_code(std::errc::device_or_resource_busy);
            return false;
        }
        const std::filesystem::path relative(filename);
        if (relative.has_root_path() || !relative.has_filename() ||
            std::find(relative.begin(), relative.end(), L"..") != relative.end()) {
            ec = std::make_error_code(std::errc::invalid_argument);
            return false;
        }
        const auto part = m_root / (filename + L".part");
        const auto status = std::filesystem::symlink_status(part, ec);
        if (status.type() == std::filesystem::file_type::not_found) {
            ec.clear();
            return true;
        }
        if (ec)
            return false;
        if (!std::filesystem::is_regular_file(status)) {
            ec = std::make_error_code(std::errc::invalid_argument);
            return false;
        }
        std::filesystem::remove(part, ec);
        return !ec;
    }

    bool begin(std::error_code& ec) {
        DownloadSpaceCheck budget;
        return begin(ec, 0, budget);
    }

    bool begin(std::error_code& ec, uint64_t required_bytes, DownloadSpaceCheck& budget) {
        budget = {};
        if (!owns_lock()) {
            ec = std::make_error_code(std::errc::device_or_resource_busy);
            return false;
        }
        std::filesystem::create_directories(m_root, ec);
        if (ec)
            return false;
        // Query the real destination volume after it exists. A rejected plan
        // must leave the installed generation's completion marker readable.
        if (required_bytes) {
            const auto space = std::filesystem::space(m_root, ec);
            if (ec)
                return false;
            budget = check_download_space(required_bytes, space.available);
            if (!budget.ok) {
                ec = std::make_error_code(std::errc::no_space_on_device);
                return false;
            }
        }
        std::filesystem::remove(m_root / ".complete", ec);
        return !ec;
    }

    bool begin_usb_import(std::error_code& ec) {
        if (!begin(ec))
            return false;
        // Even an uncatalogued/manual layout can look ready after copying only
        // its config. Persist the exclusion across process restarts.
        std::filesystem::create_directory(m_root / kUsbImportPending, ec);
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
        std::filesystem::remove(m_root / kUsbImportPending, ec);
        return !ec;
    }

  private:
    ModelWriteGate::Permit m_permit;
    std::filesystem::path m_root;
};

} // namespace xllama
