// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Withholds stop-sequence text from a streamed reply.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace xllama {

// Stream-side counterpart of apply_stop_sequences(). The decode loop reports a
// token piece before it checks the stop list, so a piece can contain the whole
// stop or just its beginning. This filter holds the longest suffix that could
// still become a stop and returns only text that is safe to send; when a stop
// completes, the stop is discarded and later pieces are dropped.
//
// Matching mirrors apply_stop_sequences exactly: only a suffix of the
// accumulated text matches, stops are tested in order and the first match wins,
// and empty stops are ignored. finish() releases a held suffix when generation
// ends without a match (EOG, token cap), because that text is not a stop.
class StopSequenceFilter {
  public:
    explicit StopSequenceFilter(std::vector<std::string> stops);

    // Text safe to emit for this piece. Empty while the suffix is ambiguous or
    // after a stop has matched.
    std::string push(std::string_view piece);

    // Releases the held suffix when no stop matched. Empty after a match.
    std::string finish();

    bool matched() const {
        return m_matched;
    }

  private:
    std::vector<std::string> m_stops; // empty stops removed
    std::string m_pending;
    bool m_matched = false;
};

} // namespace xllama
