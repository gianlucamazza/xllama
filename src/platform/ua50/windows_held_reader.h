#pragma once
#include "expert_cache.h"
#include "staged_reader_core.h"
#include <cstdio>
#include <functional>
#include <memory>
#include <string>

namespace ua50::uwp {
struct ExpectedFile { uint64_t bytes; std::string sha256; };
struct ValidationResult { uint64_t bytes_checked; std::string sha256; };
// Mandatory whole-held-FILE validator supplied by trusted app composition.
// A callback's claim is not cryptographic implementation/acceptance by this adapter.
using ValidateHeldFile = std::function<ValidationResult(FILE*,uint64_t)>;
struct WindowsHeldVerifiedAdmission {
    uint64_t bytes;
    std::string sha256;
    std::string identity; // file ID + volume, not pathname identity
};
class WindowsReader : public Reader, public std::enable_shared_from_this<WindowsReader> {
public:
    virtual void request_stop() noexcept=0;
    virtual void drain_to_terminal() noexcept=0;
    virtual Ledger ledger_after_drain() const=0;
    virtual Cause cause_after_drain() const=0;
};
struct WindowsInputControl {
    virtual ~WindowsInputControl()=default;
    virtual void request_stop() noexcept=0;
    virtual void drain_to_terminal() noexcept=0;
};
// Explicit platform result, NOT an OwnedExpertInput with a forged Linux contract.
struct WindowsInputCandidate {
    FILE* file=nullptr; // borrowed; anchor owns FILE and both native handles
    std::shared_ptr<void> file_anchor;
    std::shared_ptr<void> admission_anchor;
    std::function<void(FILE*)> validate_binding;
    std::shared_ptr<WindowsInputControl> control;
    WindowsHeldVerifiedAdmission admission;
    std::function<std::shared_ptr<WindowsReader>(size_t required_stage)> make_reader;
};
// trusted_local_path must come from authenticated app-owned LocalFolder selection,
// not a LAN string. CreateFile2 applies the AppContainer namespace boundary.
// Windows sharing != kernel immutability: pre-existing mappings/host/admin writers
// remain outside this proof. Caller must additionally exclude its own writers.
WindowsInputCandidate open_windows_input_candidate(
    const std::wstring& trusted_local_path, const ExpectedFile& expected,
    const ValidateHeldFile& validate_held, Clock::time_point absolute_deadline,
    size_t stage_ceiling=4u*1024u*1024u);
} // namespace ua50::uwp
