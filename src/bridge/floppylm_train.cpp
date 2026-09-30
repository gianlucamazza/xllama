// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "floppylm_internal.h"
#include "xllama/floppylm.h"
#include "xllama/platform.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace xllama {
namespace {
using namespace floppy;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
void need(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
std::string utf8(const fs::path& p) {
    auto s = p.u8string();
    return std::string(s.begin(), s.end());
}
std::string hex(const unsigned char* p, size_t n) {
    std::ostringstream o;
    for (size_t i = 0; i < n; ++i)
        o << std::hex << std::setw(2) << std::setfill('0') << unsigned(p[i]);
    return o.str();
}
std::string hash_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    need(bool(f), "cannot open bundle input");
    sha256_t ctx;
    sha256_init(&ctx);
    std::array<unsigned char, 65536> buf{};
    while (f) {
        f.read(reinterpret_cast<char*>(buf.data()), buf.size());
        sha256_update(&ctx, buf.data(), size_t(f.gcount()));
    }
    need(f.eof(), "input read failed");
    unsigned char digest[32];
    sha256_final(&ctx, digest);
    return hex(digest, 32);
}
// All destination names are immutable and owned by this run's exclusive directory.
void publish(const fs::path& p, const Bytes& b) {
    need(!fs::exists(p), "output already exists");
    auto tmp = p;
    tmp += ".tmp";
    write_bytes(utf8(tmp), b);
    fs::rename(tmp, p);
}
void publish_json(const fs::path& p, const Json& j) {
    auto s = j.dump(2) + "\n";
    publish(p, Bytes(s.begin(), s.end()));
}
struct Cancelled : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct Spec {
    int batch, branches, threads, checkpoint_every, T, warmup;
    double lr, wd, cooldown, wall;
    Spec(const Json& j, int ctx) {
        const std::vector<std::string> fields = {"tokens",
                                                 "branches",
                                                 "batch",
                                                 "lr",
                                                 "wd",
                                                 "warmup_frac",
                                                 "cooldown_frac",
                                                 "seed",
                                                 "threads",
                                                 "checkpoint_every",
                                                 "wall_limit_seconds"};
        need(j.is_object() && j.size() == fields.size(), "invalid spec fields");
        for (const auto& key : fields)
            need(j.contains(key), "missing spec field");
        for (const auto* key :
             {"tokens", "branches", "batch", "seed", "threads", "checkpoint_every"})
            need(j.at(key).is_number_integer(), "spec counts must be integers");
        batch = j.at("batch");
        branches = j.at("branches");
        threads = j.at("threads");
        checkpoint_every = j.at("checkpoint_every");
        lr = j.at("lr");
        wd = j.at("wd");
        cooldown = j.at("cooldown_frac");
        wall = j.at("wall_limit_seconds");
        int64_t tokens = j.at("tokens");
        double warm = j.at("warmup_frac");
        need(batch >= 1 && batch <= 32 && branches >= 1 && branches <= 8 && threads >= 1 &&
                 threads <= 16 && checkpoint_every >= 1,
             "invalid execution spec");
        need(tokens >= 1 && tokens / (batch * ctx) <= 10000000 && lr > 0 && lr <= 1 && wd >= 0 &&
                 wd <= 10 && warm >= 0 && warm < 1 && cooldown > 0 && cooldown < 1 && wall > 0 &&
                 wall <= 28800,
             "invalid WSD spec");
        T = int(std::max<int64_t>(1, tokens / (batch * ctx)));
        warmup = std::max(1, int(warm * T));
    }
    int end(int branch) const {
        return T * (1 << branch);
    }
    int cd(int branch) const {
        return std::max(1, int(cooldown * end(branch)));
    }
    float rate(int step, int branch, bool cooling) const {
        if (step < warmup)
            return float(lr * (step + 1) / warmup);
        if (!cooling)
            return float(lr);
        return float(lr *
                     std::max(0., 1. - double(step - (end(branch) - cd(branch)) + 1) / cd(branch)));
    }
};
struct Corpus {
    std::ifstream data, offsets;
    int ctx, batch;
    uint64_t size;
    Corpus(const fs::path& root, int c, int b)
        : data(root / "train.bin", std::ios::binary),
          offsets(root / "offsets.bin", std::ios::binary), ctx(c), batch(b),
          size(fs::file_size(root / "train.bin")) {
        need(bool(data) && bool(offsets), "cannot open corpus");
    }
    void get(int step, int item, std::vector<int32_t>& x, std::vector<int32_t>& y) {
        offsets.clear();
        offsets.seekg((int64_t(step) * batch + item) * 8);
        unsigned char raw[8];
        offsets.read(reinterpret_cast<char*>(raw), 8);
        need(bool(offsets), "batch offset stream exhausted");
        uint64_t at = 0;
        for (int i = 0; i < 8; ++i)
            at |= uint64_t(raw[i]) << (8 * i);
        need(at < size - ctx - 1, "batch offset outside corpus");
        data.clear();
        data.seekg(std::streamoff(at));
        std::vector<unsigned char> b(ctx + 1);
        data.read(reinterpret_cast<char*>(b.data()), b.size());
        need(bool(data), "corpus window truncated");
        x.assign(b.begin(), b.end() - 1);
        y.assign(b.begin() + 1, b.end());
    }
};
Json evaluate(const Model& m, const Bytes& val, int threads, const std::function<void()>& check) {
    need(val.size() >= 2, "validation data too short");
    int ctx = m.cfg.ctx;
    size_t done = 0, start = 0, count = 0;
    double nll = 0;
    Engine engine(m.cfg, 1, int(std::min(size_t(ctx), val.size() - 1)), threads, false);
    while (done < val.size() - 1) {
        check();
        size_t len = std::min(size_t(ctx), val.size() - 1 - start);

        std::vector<int32_t> x(val.begin() + start, val.begin() + start + len),
            y(val.begin() + start + 1, val.begin() + start + len + 1);
        auto s = engine.evaluate(m, x, y);
        for (size_t k = done - start; k < len; ++k) {
            const float* p = s.logits.data() + k * m.cfg.vocab;
            float hi = *std::max_element(p, p + m.cfg.vocab);
            double sum = 0;
            for (int v = 0; v < m.cfg.vocab; ++v)
                sum += std::exp(double(p[v] - hi));
            nll += std::log(sum) + hi - p[y[k]];
            ++count;
        }
        done = start + len;
        if (done == val.size() - 1)
            break;
        start = std::min(start + size_t(std::max(1, ctx / 2)), val.size() - 1 - size_t(ctx));
    }
    need(count == val.size() - 1, "validation coverage mismatch");
    return {{"bpb", nll / count / std::log(2.)}, {"evaluated_bytes", count}};
}
} // namespace

bool validate_floppylm_job_json(const std::string& text, TrainingJob& out, std::string* err) {
    try {
        auto j = parse_json(text);
        need(j.is_object(), "job must be an object");
        const std::vector<std::string> allowed = {
            "schema_version",  "name",   "method", "device", "bundle_path",
            "checkpoint_path", "out_dir"};
        for (auto it = j.begin(); it != j.end(); ++it)
            need(std::find(allowed.begin(), allowed.end(), it.key()) != allowed.end(),
                 "unknown FloppyLM job field");
        need(j.at("schema_version") == 1 && j.at("method") == "floppylm", "unsupported job schema");
        TrainingJob job;
        job.method = TrainMethod::FloppyLM;
        job.name = j.at("name");
        job.bundle_path = j.at("bundle_path");
        job.out_dir = j.at("out_dir");
        job.checkpoint_path = j.value("checkpoint_path", std::string{});
        std::string dev = j.at("device");
        need(dev == "host" || dev == "device", "invalid job device");
        job.device = dev == "host" ? TrainDevice::Host : TrainDevice::Device;
        need(!job.name.empty() && !job.bundle_path.empty() && !job.out_dir.empty(),
             "missing FloppyLM job path or name");
        out = std::move(job);
        return true;
    } catch (const std::exception& e) {
        if (err)
            *err = e.what();
        return false;
    }
}

TrainingResult run_floppylm_job(const TrainingJob& job, const DeviceTrainCallbacks& cb) {
    TrainingResult result;
    auto started = Clock::now();
    bool owned = false;
    Json state;
    fs::path output = fs::u8path(job.out_dir);
    std::function<void()> checkpoint;
    auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - started).count(); };
    try {
        need(job.method == TrainMethod::FloppyLM, "wrong training method");
        auto bundle_path = fs::u8path(job.bundle_path);
        auto root = bundle_path.parent_path();
        auto raw = read_bytes(job.bundle_path);
        auto bundle = parse_json(std::string(raw.begin(), raw.end()));
        need(bundle.at("schema_version") == 1 && bundle.at("reference") == "floppy_4mb@2c2e737",
             "unsupported bundle");
        Model trunk(bundle);
        Spec spec(bundle.at("spec"), trunk.cfg.ctx);
#ifdef XLLAMA_UWP
        need(spec.threads <= detect_threads_llama(),
             "thread count exceeds the console CPU thread budget");
#endif
        for (const auto* name : {"train.bin", "val.bin", "offsets.bin"}) {
            const auto& info = bundle.at("files").at(name);
            need(fs::file_size(root / name) == info.at("size").get<uint64_t>(),
                 "bundle file size mismatch");
            need(hash_file(root / name) == info.at("sha256").get<std::string>(),
                 "bundle hash mismatch");
        }
        need(fs::file_size(root / "train.bin") >= uint64_t(trunk.cfg.ctx) + 2,
             "training corpus too short");
        need(fs::file_size(root / "val.bin") <= 1048576, "validation exceeds fixed 1 MiB");
        need(fs::file_size(root / "offsets.bin") ==
                 uint64_t(spec.end(spec.branches - 1)) * spec.batch * 8,
             "wrong offset stream length");
        const auto contract = hash_file(bundle_path);
        double prior = 0;
        int generation = 0;
        state = {{"schema_version", 1},   {"contract", contract},
                 {"branch", 0},           {"phase", "trunk"},
                 {"trunk_step", 0},       {"active_step", 0},
                 {"updates", 0},          {"completed", Json::array()},
                 {"elapsed_seconds", 0.0}};
        std::unique_ptr<Model> active;
        if (!job.checkpoint_path.empty()) {
            auto encoded = read_bytes(job.checkpoint_path);
            need(encoded.size() > 36 && std::string(encoded.begin(), encoded.begin() + 4) == "FLC1",
                 "native FLC1 checkpoint required");
            unsigned char actual[32];
            sha256_hash(actual, encoded.data() + 36, encoded.size() - 36);
            need(std::equal(actual, actual + 32, encoded.begin() + 4),
                 "checkpoint digest mismatch");
            state = Json::from_cbor(encoded.begin() + 36, encoded.end());
            need(state.at("schema_version") == 1 && state.at("contract") == contract,
                 "checkpoint contract mismatch");
            trunk.restore(state.at("trunk"));
            if (!state.at("active").is_null()) {
                active = std::make_unique<Model>(bundle);
                active->restore(state.at("active"));
            }
            prior = state.at("elapsed_seconds");
            need(std::isfinite(prior) && prior >= 0, "invalid checkpoint timing");
            need(state.at("branch").get<int>() >= 0 &&
                     state.at("branch").get<int>() <= spec.branches,
                 "invalid checkpoint branch");
        }
        int branch_index = state.at("branch"), trunk_step = state.at("trunk_step"),
            active_step = state.at("active_step");
        int expected_trunk_max = spec.end(std::min(branch_index, spec.branches - 1)) -
                                 spec.cd(std::min(branch_index, spec.branches - 1));
        need(trunk_step >= 0 && trunk_step <= expected_trunk_max, "invalid checkpoint trunk step");
        need(state.at("completed").is_array() &&
                 state.at("completed").size() == size_t(branch_index),
             "checkpoint completion mismatch");
        int64_t expected_updates = trunk_step;
        for (int i = 0; i < branch_index; ++i) {
            expected_updates += spec.cd(i);
            const auto& item = state.at("completed")[i];
            need(item.at("branch") == i && item.at("end_step") == spec.end(i),
                 "checkpoint branch mismatch");
            auto saved = unpack(item.at("artifact").get_binary());
            need(saved.cfg.json() == trunk.cfg.json(), "checkpoint artifact config mismatch");
        }
        if (active) {
            need(branch_index < spec.branches && trunk_step == expected_trunk_max &&
                     active_step >= trunk_step && active_step <= spec.end(branch_index),
                 "invalid cooldown checkpoint");
            expected_updates += active_step - trunk_step;
            need(state.at("phase") == "cooldown", "checkpoint phase mismatch");
        } else
            need(state.at("phase") == "trunk" ||
                     (branch_index == spec.branches && state.at("phase") == "completed"),
                 "checkpoint phase mismatch");
        need(state.at("updates") == expected_updates, "checkpoint update count mismatch");
        result.last_loss = state.value("last_loss", 0.0);
        need(fs::create_directory(output), "run output directory must be new");
        owned = true;
        auto check = [&] {
            if (cb.abort_flag && cb.abort_flag->load())
                throw Cancelled("cancelled");
            if (prior + elapsed() >= spec.wall)
                throw Cancelled("wall budget exhausted");
        };
        checkpoint = [&] {
            state["trunk"] = trunk.state();
            state["active"] = active ? active->state() : Json();
            state["elapsed_seconds"] = prior + elapsed();
            Bytes b = {'F', 'L', 'C', '1'};
            auto payload = Json::to_cbor(state);
            unsigned char digest[32];
            sha256_hash(digest, payload.data(), payload.size());
            b.insert(b.end(), digest, digest + 32);
            b.insert(b.end(), payload.begin(), payload.end());
            std::ostringstream name;
            name << "checkpoint-" << std::setw(8) << std::setfill('0') << generation++ << ".flc";
            publish(output / name.str(), b);
        };
        try {
            publish_json(output / "running.json",
                         {{"status", "running"}, {"contract", contract}, {"name", job.name}});
            Corpus corpus(root, trunk.cfg.ctx, spec.batch);
            auto val = read_bytes(utf8(root / "val.bin"));
            Engine engine(trunk.cfg, 1, trunk.cfg.ctx, spec.threads, true);
            for (const auto& done : state["completed"]) {
                int branch = done.at("branch");
                publish(output / ("branch-" + std::to_string(branch) + ".flp"),
                        done.at("artifact").get_binary());
            }
            auto update = [&](Model& m, int step, bool cooling, int branch) {
                std::vector<std::vector<float>> gradients;
                double loss = 0;
                for (int b = 0; b < spec.batch; ++b) {
                    check();
                    std::vector<int32_t> x, y;
                    corpus.get(step, b, x, y);
                    auto s = engine.evaluate(m, x, y);
                    if (gradients.empty()) {
                        gradients = s.grads;
                        for (auto& g : gradients)
                            for (auto& v : g)
                                v /= spec.batch;
                    } else
                        for (size_t i = 0; i < gradients.size(); ++i)
                            for (size_t k = 0; k < gradients[i].size(); ++k)
                                gradients[i][k] += s.grads[i][k] / spec.batch;
                    loss += s.loss / spec.batch;
                }
                adam(m, gradients, step + 1, spec.rate(step, branch, cooling), float(spec.wd));
                result.last_loss = loss;
                state["last_loss"] = loss;
                state["updates"] = state["updates"].get<int64_t>() + 1;
                state[cooling ? "active_step" : "trunk_step"] = step + 1;
                if (cb.on_progress) {
                    DeviceTrainProgress p;
                    p.stage = TrainStage::Train;
                    p.ibatch = step + 1;
                    p.ibatch_max = spec.end(branch);
                    p.epoch = branch + 1;
                    p.epochs = spec.branches;
                    p.loss = loss;
                    cb.on_progress(p);
                }
            };
            while (state["branch"].get<int>() < spec.branches) {
                int branch = state["branch"];
                int begin = spec.end(branch) - spec.cd(branch);
                while (state["trunk_step"].get<int>() < begin) {
                    int step = state["trunk_step"];
                    update(trunk, step, false, branch);
                    state["trunk_step"] = step + 1;
                    if (state["updates"].get<int64_t>() % spec.checkpoint_every == 0)
                        checkpoint();
                }
                if (!active) {
                    active = std::make_unique<Model>(trunk);
                    state["active_step"] = begin;
                    state["phase"] = "cooldown";
                    checkpoint();
                }
                while (state["active_step"].get<int>() < spec.end(branch)) {
                    int step = state["active_step"];
                    update(*active, step, true, branch);
                    state["active_step"] = step + 1;
                    if (state["updates"].get<int64_t>() % spec.checkpoint_every == 0)
                        checkpoint();
                }
                check();
                auto artifact = pack(*active);
                auto restored = unpack(artifact);
                need(pack(restored) == artifact, "FLP2 roundtrip failed");
                auto metrics = evaluate(restored, val, spec.threads, check);
                metrics["branch"] = branch;
                metrics["end_step"] = spec.end(branch);
                metrics["artifact_bytes"] = artifact.size();
                metrics["sections"] = artifact_sections(artifact);
                metrics["artifact"] = Json::binary(artifact);
                state["completed"].push_back(metrics);
                publish(output / ("branch-" + std::to_string(branch) + ".flp"), artifact);
                active.reset();
                state["branch"] = branch + 1;
                state["phase"] = "trunk";
                checkpoint();
                if (cb.on_status)
                    cb.on_status("FloppyLM cooldown " + std::to_string(branch) + " completed");
            }
            state["training_tokens"] = state["updates"].get<int64_t>() * spec.batch * trunk.cfg.ctx;
            int64_t params = 0;
            for (const auto& t : trunk.tensors)
                if (!t.norm)
                    params += t.w.size();
            double forward = 2.0 * params + 2.0 * trunk.cfg.layers * trunk.cfg.ctx * trunk.cfg.d;
            state["training_flops"] = 3 * forward * state["training_tokens"].get<int64_t>();
            result.success = true;
            result.floppylm_artifact_path =
                utf8(output / ("branch-" + std::to_string(spec.branches - 1) + ".flp"));
            state["phase"] = "completed";
            checkpoint();
            result.stages_completed = {TrainStage::Prepare, TrainStage::Train,
                                       TrainStage::Evaluate};
        } catch (...) {
            checkpoint();
            throw;
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.error_msg = e.what();
        state["phase"] = dynamic_cast<const Cancelled*>(&e) ? "interrupted" : "failed";
    }
    result.wall_seconds = elapsed();
    result.peak_ws_mb = peak_working_set_mb();
    if (owned) {
        try {
            Json summary = {{"status", result.success
                                           ? "completed"
                                           : state.value("phase", std::string("failed"))},
                            {"error", result.error_msg},
                            {"bundle_sha256", state.value("contract", std::string{})},
                            {"engine_protocol", "floppylm-ggml-cpu-v1"},
                            {"name", job.name},
                            {"wall_seconds", result.wall_seconds},
                            {"peak_ws_mb", result.peak_ws_mb},
                            {"last_loss", result.last_loss},
                            {"training_tokens", state.value("training_tokens", int64_t(0))},
                            {"training_flops", state.value("training_flops", 0.0)},
                            {"cumulative_seconds", state.value("elapsed_seconds", 0.0)},
                            {"artifact", result.floppylm_artifact_path},
                            {"cooldowns", Json::array()}};
            for (auto done : state.value("completed", Json::array())) {
                done.erase("artifact");
                summary["cooldowns"].push_back(done);
            }
            publish_json(output / "result.json", summary);
        } catch (const std::exception& e) {
            result.success = false;
            result.error_msg += "; result: ";
            result.error_msg += e.what();
        }
    }
    return result;
}
} // namespace xllama
