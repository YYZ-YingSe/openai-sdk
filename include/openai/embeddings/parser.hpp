#pragma once

#include <string_view>

#include "openai/core/result.hpp"
#include "openai/embeddings/types.hpp"

namespace openai::embeddings {

[[nodiscard]] auto parse_request(std::string_view json_text)
    -> core::result<request>;

[[nodiscard]] auto parse_response(std::string_view json_text)
    -> core::result<response>;

} // namespace openai::embeddings
