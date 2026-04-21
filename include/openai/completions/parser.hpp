#pragma once

#include <string_view>

#include "openai/completions/types.hpp"
#include "openai/core/result.hpp"

namespace openai::completions {

[[nodiscard]] auto parse_request(std::string_view json_text)
    -> core::result<request>;

[[nodiscard]] auto parse_response(std::string_view json_text)
    -> core::result<response>;

[[nodiscard]] auto parse_chunk(std::string_view json_text)
    -> core::result<chunk>;

} // namespace openai::completions
