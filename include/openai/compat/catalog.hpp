#pragma once

#include <array>
#include <string_view>

namespace openai::compat {

enum class api_family {
  responses,
  chat_completions,
  completions,
  embeddings,
  models,
  unknown,
};

struct api_family_spec {
  api_family family{};
  std::string_view name{};
  std::string_view path{};
};

inline constexpr std::array<api_family_spec, 5> api_families{{
    {api_family::responses, "responses", "responses"},
    {api_family::chat_completions, "chat_completions", "chat/completions"},
    {api_family::completions, "completions", "completions"},
    {api_family::embeddings, "embeddings", "embeddings"},
    {api_family::models, "models", "models"},
}};

[[nodiscard]] inline auto to_string(api_family family) noexcept
    -> std::string_view {
  for (const auto &spec : api_families) {
    if (spec.family == family) {
      return spec.name;
    }
  }
  return "unknown";
}

} // namespace openai::compat
