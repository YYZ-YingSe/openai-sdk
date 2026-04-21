#pragma once

#include <string>
#include <string_view>

namespace openai::core {

[[nodiscard]] inline auto copy_string(const std::string_view value)
    -> std::string {
  return std::string{value};
}

[[nodiscard]] inline auto trim_ascii_whitespace(std::string_view value)
    -> std::string_view {
  while (!value.empty() &&
         (value.front() == ' ' || value.front() == '\t' ||
          value.front() == '\r' || value.front() == '\n')) {
    value.remove_prefix(1U);
  }
  while (!value.empty() &&
         (value.back() == ' ' || value.back() == '\t' ||
          value.back() == '\r' || value.back() == '\n')) {
    value.remove_suffix(1U);
  }
  return value;
}

} // namespace openai::core
