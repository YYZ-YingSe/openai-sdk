#pragma once

#include <string_view>

#include "openai/core/result.hpp"
#include "openai/models/types.hpp"

namespace openai::models {

[[nodiscard]] auto parse_model(std::string_view json_text)
    -> core::result<model>;

[[nodiscard]] auto parse_list_response(std::string_view json_text)
    -> core::result<list_response>;

} // namespace openai::models
