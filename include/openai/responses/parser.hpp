#pragma once

#include <string_view>

#include "openai/core/result.hpp"
#include "openai/responses/extensions.hpp"
#include "openai/responses/types.hpp"

namespace openai::responses {

[[nodiscard]] auto parse_request(std::string_view json_text,
                                 const parse_options &options = {})
    -> core::result<request>;

[[nodiscard]] auto parse_response(std::string_view json_text,
                                  const parse_options &options = {})
    -> core::result<response>;

[[nodiscard]] auto parse_item_json(std::string_view json_text,
                                   const parse_options &options = {})
    -> core::result<item>;

[[nodiscard]] auto
parse_message_content_part_json(std::string_view json_text,
                                const parse_options &options = {})
    -> core::result<message_content_part>;

[[nodiscard]] auto parse_tool_definition_json(std::string_view json_text,
                                              const parse_options &options = {})
    -> core::result<tool_definition>;

[[nodiscard]] auto parse_tool_choice_json(std::string_view json_text,
                                          const parse_options &options = {})
    -> core::result<tool_choice>;

[[nodiscard]] auto parse_response_error_json(std::string_view json_text,
                                             const parse_options &options = {})
    -> core::result<response_error>;

} // namespace openai::responses
