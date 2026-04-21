#include "openai/streaming/sse.hpp"

#include <string>

#include "openai/core/error.hpp"
#include "openai/core/string_view.hpp"

namespace openai::streaming {
namespace {

[[nodiscard]] auto parse_block(std::string_view block) -> std::optional<sse_event> {
  sse_event parsed{};
  bool saw_field = false;

  std::size_t start = 0U;
  while (start <= block.size()) {
    const auto end = block.find('\n', start);
    auto line = end == std::string_view::npos ? block.substr(start)
                                              : block.substr(start, end - start);
    if (!line.empty() && line.front() == ':') {
      if (end == std::string_view::npos) {
        break;
      }
      start = end + 1U;
      continue;
    }

    const auto separator = line.find(':');
    const auto field = separator == std::string_view::npos
                           ? line
                           : line.substr(0U, separator);
    auto value = separator == std::string_view::npos
                     ? std::string_view{}
                     : line.substr(separator + 1U);
    if (!value.empty() && value.front() == ' ') {
      value.remove_prefix(1U);
    }

    if (field == "event") {
      parsed.event = std::string{value};
      saw_field = true;
    } else if (field == "data") {
      if (!parsed.data.empty()) {
        parsed.data.push_back('\n');
      }
      parsed.data.append(value.data(), value.size());
      saw_field = true;
    } else if (field == "id") {
      parsed.id = std::string{value};
      saw_field = true;
    } else if (field == "retry") {
      try {
        parsed.retry = std::stoll(std::string{value});
      } catch (...) {
        parsed.retry.reset();
      }
      saw_field = true;
    }

    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1U;
  }

  if (!saw_field) {
    return std::nullopt;
  }
  return parsed;
}

} // namespace

auto sse_parser::feed(std::string_view chunk) -> std::vector<sse_event> {
  if (pending_cr_ && !chunk.empty() && chunk.front() == '\n') {
    chunk.remove_prefix(1U);
  }
  pending_cr_ = false;

  for (std::size_t index = 0U; index < chunk.size(); ++index) {
    const auto current = chunk[index];
    if (current == '\r') {
      buffer_.push_back('\n');
      pending_cr_ = true;
      continue;
    }
    if (current == '\n' && pending_cr_) {
      pending_cr_ = false;
      continue;
    }
    pending_cr_ = false;
    buffer_.push_back(current);
  }

  return parse_available();
}

auto sse_parser::flush() -> std::vector<sse_event> {
  auto events = parse_available();
  if (!buffer_.empty()) {
    if (auto parsed = parse_block(buffer_); parsed.has_value()) {
      events.push_back(std::move(*parsed));
    }
    buffer_.clear();
  }
  pending_cr_ = false;
  return events;
}

auto sse_parser::parse_available() -> std::vector<sse_event> {
  std::vector<sse_event> events;

  std::size_t separator = 0U;
  while ((separator = buffer_.find("\n\n")) != std::string::npos) {
    const std::string_view block{buffer_.data(), separator};
    if (auto parsed = parse_block(block); parsed.has_value()) {
      events.push_back(std::move(*parsed));
    }
    buffer_.erase(0U, separator + 2U);
  }

  return events;
}

auto parse_sse_stream(std::string_view stream_text)
    -> core::result<std::vector<sse_event>> {
  sse_parser parser;
  auto events = parser.feed(stream_text);
  auto trailing = parser.flush();
  events.insert(events.end(),
                std::make_move_iterator(trailing.begin()),
                std::make_move_iterator(trailing.end()));
  return core::result<std::vector<sse_event>>::success(std::move(events));
}

} // namespace openai::streaming
