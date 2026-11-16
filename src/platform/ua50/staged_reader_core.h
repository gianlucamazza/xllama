#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ua50::uwp {
using Clock = std::chrono::steady_clock;
enum class Cause { none, stopped, deadline, io, short_read, identity, protocol };
// Pending + error means completion is UNKNOWN: cancel and keep polling, never free.
struct IoResult { bool terminal; size_t bytes; uint32_t error; };
struct Ledger {
    uint64_t submissions=0, terminal_operations=0, terminal_bytes=0;
    uint64_t polls=0, cancellations=0, commits=0;
    uint32_t first_native_error=0;
};

// Backend start/poll/cancel/identity must be noexcept. A pending operation must
// eventually be reported terminal by the backend; there is NO hard drain deadline.
// This is the algorithm used by the Windows adapter, also exercised with fake I/O.
template<class Backend> class StagedReaderCore {
    static size_t checked_stage(size_t bytes) {
        if(!bytes || bytes>4u*1024u*1024u) throw std::invalid_argument("bounded stage");
        return bytes;
    }
    Backend backend_; // holds file/event longer than stage storage
    std::vector<unsigned char> stage_;
    const uint64_t file_bytes_;
    const Clock::time_point deadline_;
    std::atomic<bool> stopped_{false};
    std::mutex read_mutex_;
    Cause cause_=Cause::none;
    Ledger ledger_{};
    void latch(Cause cause) noexcept { if (cause_==Cause::none) cause_=cause; }
    void check_stop() noexcept {
        if (stopped_.load(std::memory_order_acquire)) latch(Cause::stopped);
        if (Clock::now()>=deadline_) latch(Cause::deadline);
    }
    [[noreturn]] void fail() const { throw std::runtime_error("UWP staged Reader failed; session invalid"); }
public:
    StagedReaderCore(Backend backend, uint64_t file_bytes, size_t stage_bytes,
                     Clock::time_point deadline)
        :backend_(std::move(backend)),stage_(checked_stage(stage_bytes)),file_bytes_(file_bytes),deadline_(deadline) {
        if (!file_bytes || !stage_bytes || stage_bytes>4u*1024u*1024u || deadline<=Clock::now())
            throw std::invalid_argument("bounded Reader size/deadline");
    }
    StagedReaderCore(const StagedReaderCore&)=delete;
    StagedReaderCore& operator=(const StagedReaderCore&)=delete;
    void request_stop() noexcept { stopped_.store(true,std::memory_order_release); }
    void drain_to_terminal() noexcept {
        request_stop();
        // A read owns this lock through actual completion and commit/refusal.
        // May block indefinitely on failed native cancellation. Never release early.
        std::lock_guard<std::mutex> guard(read_mutex_);
    }
    // Caller may inspect only after read has returned / graph workers drained.
    const Ledger& ledger_after_drain() const noexcept { return ledger_; }
    Cause cause_after_drain() const noexcept { return cause_; }
    size_t stage_bytes() const noexcept { return stage_.size(); }

    size_t read(uint64_t offset, unsigned char* destination, size_t bytes) {
        std::unique_lock<std::mutex> lock(read_mutex_,std::try_to_lock);
        if (!lock.owns_lock()) throw std::logic_error("Reader coordinator must be serialized");
        check_stop();
        if (cause_!=Cause::none) fail();
        if (!destination || bytes>stage_.size() || offset>file_bytes_ || bytes>file_bytes_-offset)
            { latch(Cause::protocol); fail(); }
        if (!backend_.identity()) { latch(Cause::identity); fail(); }
        if (!bytes) return 0;
        ++ledger_.submissions;
        IoResult result=backend_.start(offset,stage_.data(),bytes);
        bool cancelled=false;
        while (!result.terminal) {
            if (result.error) {
                if(!ledger_.first_native_error) ledger_.first_native_error=result.error;
                latch(Cause::protocol);
            }
            check_stop();
            if (cause_!=Cause::none && !cancelled) {
                ++ledger_.cancellations;
                backend_.cancel(); // request only; return code is NOT completion
                cancelled=true;
            }
            ++ledger_.polls;
            result=backend_.poll(50); // retain stage/OVERLAPPED/event/file on every timeout
        }
        ++ledger_.terminal_operations;
        ledger_.terminal_bytes+=result.bytes; // logical completed bytes, NOT physical SSD traffic
        if (result.error) {
            if(!ledger_.first_native_error) ledger_.first_native_error=result.error;
            latch(Cause::io);
        }
        if (result.bytes!=bytes) latch(Cause::short_read);
        check_stop();
        if (!backend_.identity()) latch(Cause::identity);
        if (cause_!=Cause::none) fail();
        // No partial, failed, cancelled, expired or identity-changed bytes reach caller.
        std::memcpy(destination,stage_.data(),bytes);
        ++ledger_.commits;
        return bytes;
    }
};
} // namespace ua50::uwp
