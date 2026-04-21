#pragma once

#include <string_view>

#include "openai/chat_completions/types.hpp"
#include "openai/core/result.hpp"
#include "openai/responses/extensions.hpp"

namespace openai::chat_completions {

[[nodiscard]] auto parse_request(
    std::string_view json_text,
    const responses::parse_options &options = {})
    -> core::result<request>;

[[nodiscard]] auto parse_completion(
    std::string_view json_text,
    const responses::parse_options &options = {})
    -> core::result<completion>;

[[nodiscard]] auto parse_completion_chunk(
    std::string_view json_text,
    const responses::parse_options &options = {})
    -> core::result<completion_chunk>;

} // namespace openai::chat_completions
