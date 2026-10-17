// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/stop_sequence_filter.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace xllama {

StopSequenceFilter::StopSequenceFilter(std::vector<std::string> stops) {
    m_stops.reserve(stops.size());
    for (std::string& stop : stops) {
        if (!stop.empty())
            m_stops.push_back(std::move(stop));
    }
}

std::string StopSequenceFilter::push(std::string_view piece) {
    if (m_matched)
        return {};
    m_pending.append(piece.data(), piece.size());
    if (m_pending.empty())
        return {};

    // Same order and suffix rule as apply_stop_sequences: the first stop that
    // ends the accumulated text wins, even when a longer one would also match.
    for (const std::string& stop : m_stops) {
        if (m_pending.size() >= stop.size() &&
            m_pending.compare(m_pending.size() - stop.size(), stop.size(), stop) == 0) {
            std::string out = m_pending.substr(0, m_pending.size() - stop.size());
            m_pending.clear();
            m_matched = true;
            return out;
        }
    }

    // Hold the longest suffix that is a proper prefix of a stop. Everything
    // before it cannot be the beginning of a stop and is safe to emit now.
    std::size_t hold = 0;
    for (const std::string& stop : m_stops) {
        const std::size_t limit = std::min(m_pending.size(), stop.size() - 1);
        for (std::size_t len = limit; len > hold; --len) {
            if (m_pending.compare(m_pending.size() - len, len, stop, 0, len) == 0) {
                hold = len;
                break;
            }
        }
    }
    if (hold == 0) {
        std::string out = std::move(m_pending);
        m_pending.clear();
        return out;
    }
    std::string out = m_pending.substr(0, m_pending.size() - hold);
    m_pending.erase(0, m_pending.size() - hold);
    return out;
}

std::string StopSequenceFilter::finish() {
    if (m_matched) {
        m_pending.clear();
        return {};
    }
    std::string out = std::move(m_pending);
    m_pending.clear();
    return out;
}

} // namespace xllama
