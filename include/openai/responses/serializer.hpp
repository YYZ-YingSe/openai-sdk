#pragma once

#include <string_view>

#include "openai/core/result.hpp"
#include "openai/core/serialization.hpp"
#include "openai/responses/types.hpp"

namespace openai::responses {

[[nodiscard]] auto serialize_request(const request &value,
                                     const core::serialize_options &options = {})
    -> core::result<std::string>;

[[nodiscard]] auto serialize_response(const response &value,
                                      const core::serialize_options &options = {})
    -> core::result<std::string>;

[[nodiscard]] auto serialize_item(const item &value,
                                  const core::serialize_options &options = {})
    -> core::result<std::string>;

[[nodiscard]] auto serialize_message_content_part(
    const message_content_part &value,
    const core::serialize_options &options = {}) -> core::result<std::string>;

[[nodiscard]] auto serialize_tool_definition(
    const tool_definition &value,
    const core::serialize_options &options = {}) -> core::result<std::string>;

[[nodiscard]] auto serialize_tool_choice(
    const tool_choice &value,
    const core::serialize_options &options = {}) -> core::result<std::string>;

[[nodiscard]] auto serialize_response_error(
    const response_error &value,
    const core::serialize_options &options = {}) -> core::result<std::string>;

} // namespace openai::responses
