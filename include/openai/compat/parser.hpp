#pragma once

#include <string_view>

#include "openai/compat/types.hpp"
#include "openai/core/result.hpp"
#include "openai/responses/extensions.hpp"

namespace openai::compat {

[[nodiscard]] auto parse_request_json(
    std::string_view json_text, const operation_hint &hint = {},
    const responses::parse_options &options = {}) -> core::result<request_document>;

[[nodiscard]] auto parse_response_json(
    std::string_view json_text, const operation_hint &hint = {},
    const responses::parse_options &options = {}) -> core::result<response_document>;

[[nodiscard]] auto parse_document_json(
    std::string_view json_text, const responses::parse_options &options = {})
    -> core::result<response_document>;

} // namespace openai::compat
