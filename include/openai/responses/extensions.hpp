#pragma once

#include <functional>
#include <map>
#include <string>
#include <string_view>

#include "openai/core/json.hpp"
#include "openai/core/result.hpp"
#include "openai/responses/types.hpp"

namespace openai::responses {

struct parse_options;

using extension_parser =
    std::function<core::result<raw_json>(std::string_view type,
                                         const core::json_value &value,
                                         const parse_options &options)>;

class registry {
public:
  auto register_content_part(std::string type, extension_parser parser)
      -> void {
    content_part_parsers_[std::move(type)] = std::move(parser);
  }

  auto register_item(std::string type, extension_parser parser) -> void {
    item_parsers_[std::move(type)] = std::move(parser);
  }

  auto register_tool(std::string type, extension_parser parser) -> void {
    tool_parsers_[std::move(type)] = std::move(parser);
  }

  [[nodiscard]] auto find_content_part(std::string_view type) const noexcept
      -> const extension_parser * {
    if (const auto it = content_part_parsers_.find(type);
        it != content_part_parsers_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  [[nodiscard]] auto find_item(std::string_view type) const noexcept
      -> const extension_parser * {
    if (const auto it = item_parsers_.find(type); it != item_parsers_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  [[nodiscard]] auto find_tool(std::string_view type) const noexcept
      -> const extension_parser * {
    if (const auto it = tool_parsers_.find(type); it != tool_parsers_.end()) {
      return &it->second;
    }
    return nullptr;
  }

private:
  std::map<std::string, extension_parser, std::less<>> content_part_parsers_{};
  std::map<std::string, extension_parser, std::less<>> item_parsers_{};
  std::map<std::string, extension_parser, std::less<>> tool_parsers_{};
};

struct parse_options {
  const registry *registry{nullptr};
  bool strict{false};
};

[[nodiscard]] inline auto make_raw_json(std::string_view type,
                                        const core::json_value &value)
    -> core::result<raw_json> {
  auto serialized = core::json_to_string(value);
  if (serialized.has_error()) {
    return core::result<raw_json>::failure(serialized.error());
  }
  auto structured = core::json_to_value(value);
  if (structured.has_error()) {
    return core::result<raw_json>::failure(structured.error());
  }

  raw_json parsed{};
  parsed.type = std::string{type};
  parsed.json = std::move(serialized.value());
  parsed.data = std::move(structured.value());
  return core::result<raw_json>::success(std::move(parsed));
}

} // namespace openai::responses
