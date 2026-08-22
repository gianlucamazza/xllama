# Path Utilities

## Path Resolution

```cpp
// Resolve a model filename: UWP -> LocalFolder\models\<filename>
std::string resolve_model_path(const std::string& filename);

// Resolve a generic filename: UWP -> LocalFolder\<filename>
std::string resolve_local_path(const std::string& filename);

// Check if a model uses the llama.cpp backend.
bool model_uses_llama_backend(const std::string& model_id);

// Pick a base weights file from a directory.
std::string first_gguf_in_dir(const std::string& path,
                              const std::string& exclude_filename = {});
```

## Model Provisioning

```cpp
// Normalize a path for comparison: backslashes -> '/', ASCII-lowercased.
std::wstring normalize_model_path(std::wstring s);

// True iff every expected file is present in the directory listing.
bool dir_satisfies_expected_files(const std::vector<std::wstring>& present,
                                  const std::vector<std::wstring>& expected);

// True iff a model directory is servable by Session.
bool model_dir_files_ready(const std::vector<std::string>& files);
```

## Manifest Merge

```cpp
// Merge a catalogue override on top of a base catalogue.
// A same-name entry REPLACES the base one; a new name is APPENDED.
template <typename Entry>
void merge_manifest_entries(std::vector<Entry>& base, std::vector<Entry> overrides);
```
