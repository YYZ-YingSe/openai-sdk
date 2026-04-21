#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "openai/core/result.hpp"

namespace openai::streaming {

struct sse_event {
  std::string event{"message"};
  std::string data{};
  std::optional<std::string> id{};
  std::optional<std::int64_t> retry{};
};

class sse_parser {
public:
  auto feed(std::string_view chunk) -> std::vector<sse_event>;
  auto flush() -> std::vector<sse_event>;

private:
  auto parse_available() -> std::vector<sse_event>;

  std::string buffer_{};
  bool pending_cr_{false};
};

[[nodiscard]] auto parse_sse_stream(std::string_view stream_text)
    -> core::result<std::vector<sse_event>>;

} // namespace openai::streaming
