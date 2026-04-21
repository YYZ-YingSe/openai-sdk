#pragma once

#include "openai/chat_completions/types.hpp"
#include "openai/core/result.hpp"
#include "openai/core/serialization.hpp"

namespace openai::chat_completions {

[[nodiscard]] auto serialize_request(const request &value,
                                     const core::serialize_options &options = {})
    -> core::result<std::string>;

[[nodiscard]] auto serialize_completion(
    const completion &value,
    const core::serialize_options &options = {}) -> core::result<std::string>;

[[nodiscard]] auto serialize_completion_chunk(
    const completion_chunk &value,
    const core::serialize_options &options = {}) -> core::result<std::string>;

} // namespace openai::chat_completions
