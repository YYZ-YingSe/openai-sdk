#include "openai/wire/parser.hpp"

#include <algorithm>
#include <cctype>

#include "openai/core/error.hpp"

namespace openai::wire {
namespace {

using core::errc;
using core::make_error;
using core::result;

[[nodiscard]] auto trim(std::string_view text) -> std::string_view {
  auto start = std::size_t{0};
  auto end = text.size();
  while (start < end &&
         std::isspace(static_cast<unsigned char>(text[start])) != 0) {
    ++start;
  }
  while (end > start &&
         std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  return text.substr(start, end - start);
}

[[nodiscard]] auto unquote(std::string_view text) -> std::string {
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
    return std::string{text.substr(1, text.size() - 2)};
  }
  return std::string{text};
}

[[nodiscard]] auto hex_value(const char ch) -> int {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return 10 + (ch - 'a');
  }
  if (ch >= 'A' && ch <= 'F') {
    return 10 + (ch - 'A');
  }
  return -1;
}

[[nodiscard]] auto decode_component(std::string_view text) -> result<std::string> {
  std::string decoded{};
  decoded.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    const auto ch = text[i];
    if (ch == '+') {
      decoded.push_back(' ');
      continue;
    }
    if (ch == '%') {
      if (i + 2 >= text.size()) {
        return result<std::string>::failure(make_error(
            errc::parse_error, "query string contains truncated percent-encoding"));
      }
      const auto high = hex_value(text[i + 1]);
      const auto low = hex_value(text[i + 2]);
      if (high < 0 || low < 0) {
        return result<std::string>::failure(make_error(
            errc::parse_error, "query string contains invalid percent-encoding"));
      }
      decoded.push_back(static_cast<char>((high << 4) | low));
      i += 2;
      continue;
    }
    decoded.push_back(ch);
  }
  return result<std::string>::success(std::move(decoded));
}

[[nodiscard]] auto split_lines(std::string_view text) -> std::vector<std::string_view> {
  std::vector<std::string_view> lines{};
  std::size_t offset = 0;
  while (offset <= text.size()) {
    const auto next = text.find('\n', offset);
    if (next == std::string_view::npos) {
      lines.push_back(text.substr(offset));
      break;
    }
    auto line = text.substr(offset, next - offset);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    lines.push_back(line);
    offset = next + 1;
  }
  return lines;
}

[[nodiscard]] auto parse_part_headers(std::string_view block)
    -> result<std::vector<header>> {
  return parse_headers(block);
}

} // namespace

auto parse_headers(std::string_view text) -> core::result<std::vector<header>> {
  std::vector<header> parsed{};
  for (const auto line : split_lines(text)) {
    if (trim(line).empty()) {
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string_view::npos) {
      return result<std::vector<header>>::failure(make_error(
          errc::parse_error, "wire header line is missing ':' separator"));
    }
    header current{};
    current.name = std::string{trim(line.substr(0, colon))};
    current.value = std::string{trim(line.substr(colon + 1))};
    if (current.name.empty()) {
      return result<std::vector<header>>::failure(
          make_error(errc::parse_error, "wire header name must not be empty"));
    }
    parsed.push_back(std::move(current));
  }
  return result<std::vector<header>>::success(std::move(parsed));
}

auto parse_query_string(std::string_view text) -> core::result<query_string> {
  query_string parsed{};
  if (!text.empty() && text.front() == '?') {
    text.remove_prefix(1);
  }
  parsed.raw_query = std::string{text};
  if (text.empty()) {
    return result<query_string>::success(std::move(parsed));
  }
  std::size_t offset = 0;
  while (offset <= text.size()) {
    const auto next = text.find('&', offset);
    const auto token = text.substr(offset, next == std::string_view::npos
                                               ? std::string_view::npos
                                               : next - offset);
    if (!token.empty()) {
      const auto equal = token.find('=');
      query_parameter parameter{};
      if (equal == std::string_view::npos) {
        auto name = decode_component(token);
        if (name.has_error()) {
          return result<query_string>::failure(name.error());
        }
        parameter.name = std::move(name.value());
      } else {
        auto name = decode_component(token.substr(0, equal));
        if (name.has_error()) {
          return result<query_string>::failure(name.error());
        }
        auto value = decode_component(token.substr(equal + 1));
        if (value.has_error()) {
          return result<query_string>::failure(value.error());
        }
        parameter.name = std::move(name.value());
        parameter.value = std::move(value.value());
      }
      parsed.parameters.push_back(std::move(parameter));
    }
    if (next == std::string_view::npos) {
      break;
    }
    offset = next + 1;
  }
  return result<query_string>::success(std::move(parsed));
}

auto parse_content_disposition(std::string_view value)
    -> core::result<content_disposition> {
  content_disposition parsed{};
  auto remainder = trim(value);
  const auto first_semicolon = remainder.find(';');
  if (first_semicolon == std::string_view::npos) {
    parsed.type = std::string{trim(remainder)};
    return result<content_disposition>::success(std::move(parsed));
  }
  parsed.type = std::string{trim(remainder.substr(0, first_semicolon))};
  remainder.remove_prefix(first_semicolon + 1);
  while (!remainder.empty()) {
    const auto next_semicolon = remainder.find(';');
    auto token = trim(remainder.substr(0, next_semicolon));
    if (!token.empty()) {
      const auto equal = token.find('=');
      if (equal == std::string_view::npos) {
        return result<content_disposition>::failure(make_error(
            errc::parse_error,
            "content-disposition parameter is missing '=' separator"));
      }
      const auto key = std::string{trim(token.substr(0, equal))};
      const auto raw_value = trim(token.substr(equal + 1));
      const auto parsed_value = unquote(raw_value);
      parsed.parameters[key] = parsed_value;
      if (key == "name") {
        parsed.name = parsed_value;
      } else if (key == "filename") {
        parsed.filename = parsed_value;
      }
    }
    if (next_semicolon == std::string_view::npos) {
      break;
    }
    remainder.remove_prefix(next_semicolon + 1);
  }
  return result<content_disposition>::success(std::move(parsed));
}

auto parse_multipart_body(std::string_view body, std::string_view boundary)
    -> core::result<multipart_body> {
  if (boundary.empty()) {
    return result<multipart_body>::failure(
        make_error(errc::invalid_argument, "multipart boundary must not be empty"));
  }

  multipart_body parsed{};
  parsed.boundary = std::string{boundary};
  parsed.raw_body = std::string{body};

  const std::string delimiter = "--" + std::string{boundary};
  std::size_t cursor = 0;

  while (true) {
    const auto boundary_pos = body.find(delimiter, cursor);
    if (boundary_pos == std::string_view::npos) {
      break;
    }
    auto part_start = boundary_pos + delimiter.size();
    if (part_start + 1 < body.size() && body.substr(part_start, 2) == "--") {
      break;
    }
    if (part_start < body.size() && body[part_start] == '\r') {
      part_start += 2;
    } else if (part_start < body.size() && body[part_start] == '\n') {
      part_start += 1;
    }

    const auto next_boundary = body.find(delimiter, part_start);
    if (next_boundary == std::string_view::npos) {
      return result<multipart_body>::failure(make_error(
          errc::parse_error, "multipart body is missing terminating boundary"));
    }

    auto part_block = body.substr(part_start, next_boundary - part_start);
    if (part_block.size() >= 2 &&
        part_block.substr(part_block.size() - 2) == "\r\n") {
      part_block.remove_suffix(2);
    } else if (!part_block.empty() && part_block.back() == '\n') {
      part_block.remove_suffix(1);
    }

    if (!trim(part_block).empty()) {
      const auto header_end = part_block.find("\r\n\r\n");
      const auto alt_header_end = part_block.find("\n\n");
      std::size_t split = std::string_view::npos;
      std::size_t body_offset = 0;
      if (header_end != std::string_view::npos) {
        split = header_end;
        body_offset = 4;
      } else if (alt_header_end != std::string_view::npos) {
        split = alt_header_end;
        body_offset = 2;
      } else {
        return result<multipart_body>::failure(make_error(
            errc::parse_error, "multipart part is missing header/body separator"));
      }

      multipart_part part{};
      auto headers = parse_part_headers(part_block.substr(0, split));
      if (headers.has_error()) {
        return result<multipart_body>::failure(headers.error());
      }
      part.headers = std::move(headers.value());
      part.body = std::string{part_block.substr(split + body_offset)};

      for (const auto &item : part.headers) {
        if (item.name == "Content-Disposition" ||
            item.name == "content-disposition") {
          auto disposition = parse_content_disposition(item.value);
          if (disposition.has_error()) {
            return result<multipart_body>::failure(disposition.error());
          }
          part.disposition = std::move(disposition.value());
        } else if (item.name == "Content-Type" || item.name == "content-type") {
          part.content_type = item.value;
        }
      }

      parsed.parts.push_back(std::move(part));
    }
    cursor = next_boundary;
  }

  if (parsed.parts.empty()) {
    return result<multipart_body>::failure(
        make_error(errc::parse_error, "multipart body did not contain any parts"));
  }
  return result<multipart_body>::success(std::move(parsed));
}

auto parse_binary_payload(std::string_view body,
                          std::optional<std::string> content_type)
    -> core::result<binary_payload> {
  binary_payload parsed{};
  parsed.content_type = std::move(content_type);
  parsed.data = std::string{body};
  return result<binary_payload>::success(std::move(parsed));
}

} // namespace openai::wire
