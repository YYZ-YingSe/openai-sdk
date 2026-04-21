#pragma once

#include <string_view>
#include <vector>

#include "openai/core/result.hpp"
#include "openai/wire/types.hpp"

namespace openai::wire {

[[nodiscard]] auto parse_headers(std::string_view text)
    -> core::result<std::vector<header>>;

[[nodiscard]] auto parse_query_string(std::string_view text)
    -> core::result<query_string>;

[[nodiscard]] auto parse_content_disposition(std::string_view value)
    -> core::result<content_disposition>;

[[nodiscard]] auto parse_multipart_body(std::string_view body,
                                        std::string_view boundary)
    -> core::result<multipart_body>;

[[nodiscard]] auto parse_binary_payload(std::string_view body,
                                        std::optional<std::string> content_type = std::nullopt)
    -> core::result<binary_payload>;

} // namespace openai::wire
