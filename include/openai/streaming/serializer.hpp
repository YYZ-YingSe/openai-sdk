#pragma once

#include "openai/core/result.hpp"
#include "openai/core/serialization.hpp"
#include "openai/streaming/response_events.hpp"

namespace openai::streaming {

[[nodiscard]] auto serialize_response_event_json(
    const response_event &value,
    const core::serialize_options &options = {}) -> core::result<std::string>;

[[nodiscard]] auto serialize_sse_event(const sse_event &value)
    -> core::result<std::string>;

[[nodiscard]] auto serialize_sse_stream(const std::vector<sse_event> &events)
    -> core::result<std::string>;

} // namespace openai::streaming
