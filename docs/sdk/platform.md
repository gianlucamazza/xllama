# Platform Helpers

```cpp
// Number of hardware threads (fallback: 4).
int detect_threads() noexcept;

// Default thread count for llama.cpp (UWP capped at 6).
int detect_threads_llama() noexcept;

// Emit a log line (UWP: OutputDebugStringA, otherwise: stderr).
void log_output(const char* msg) noexcept;
void log_output(const std::string& msg) noexcept;

// Peak working set in MB (0 if unavailable).
std::size_t peak_working_set_mb() noexcept;

// Physical memory available to the process, in MB.
std::size_t avail_phys_mb() noexcept;

// GPU memory info (UWP only).
struct GpuMemInfo {
    std::size_t current_mb = 0;
    std::size_t budget_mb = 0;
    bool available = false;
};
GpuMemInfo gpu_mem_info() noexcept;

// Pin the process CWD to ApplicationData LocalFolder (UWP only).
void set_cwd_to_local_folder() noexcept;
```
