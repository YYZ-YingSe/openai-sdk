#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "openai/compat/catalog.hpp"

namespace openai::capabilities {

enum class support_level : std::uint8_t {
  unsupported = 0,
  passthrough = 1,
  emulated = 2,
  native = 3,
};

[[nodiscard]] inline auto min_support_level(support_level left,
                                            support_level right) noexcept
    -> support_level {
  return static_cast<support_level>(
      std::min(static_cast<std::uint8_t>(left),
               static_cast<std::uint8_t>(right)));
}

[[nodiscard]] inline auto is_supported(support_level level) noexcept -> bool {
  return level != support_level::unsupported;
}

struct family_capabilities {
  support_level request_support{support_level::native};
  support_level response_support{support_level::native};
  support_level streaming{support_level::native};
  support_level tools{support_level::native};
  support_level structured_outputs{support_level::native};
  support_level parallel_tool_calls{support_level::native};
};

struct tool_capabilities {
  support_level function_tools{support_level::native};
  support_level file_search{support_level::native};
  support_level tool_search{support_level::native};
  support_level web_search{support_level::native};
  support_level computer_use{support_level::native};
  support_level namespace_tools{support_level::native};
  support_level local_shell{support_level::native};
  support_level shell{support_level::native};
  support_level function_shell{support_level::native};
  support_level custom_tools{support_level::native};
  support_level image_generation{support_level::native};
  support_level code_interpreter{support_level::native};
  support_level mcp{support_level::native};
  support_level apply_patch{support_level::native};
};

struct modality_capabilities {
  support_level text_input{support_level::native};
  support_level text_output{support_level::native};
  support_level image_input{support_level::native};
  support_level image_output{support_level::native};
  support_level audio_input{support_level::native};
  support_level audio_output{support_level::native};
};

struct structured_output_capabilities {
  support_level text{support_level::native};
  support_level json_schema{support_level::native};
  support_level strict{support_level::native};
};

struct sampling_capabilities {
  support_level temperature{support_level::native};
  support_level top_p{support_level::native};
  support_level n{support_level::native};
  support_level stop{support_level::native};
  support_level logprobs{support_level::native};
  support_level top_logprobs{support_level::native};
  support_level seed{support_level::native};
  support_level best_of{support_level::native};
  support_level max_tokens{support_level::native};
  support_level max_completion_tokens{support_level::native};
  support_level max_output_tokens{support_level::native};
};

struct provider_profile {
  std::string id{};
  std::set<compat::api_family> supported_families{};
  family_capabilities responses{};
  family_capabilities chat_completions{};
  family_capabilities completions{};
  family_capabilities embeddings{};
  family_capabilities models{};
  tool_capabilities tools{};
  modality_capabilities modalities{};
  structured_output_capabilities structured_outputs{};
  sampling_capabilities sampling{};
  std::optional<std::int64_t> max_tools{};
  std::optional<std::int64_t> max_stop_sequences{};
  std::optional<std::int64_t> max_n{};
  std::optional<std::int64_t> max_top_logprobs{};
  std::optional<std::int64_t> max_dimensions{};
  support_level raw_passthrough{support_level::native};
};

struct model_profile {
  std::string id{};
  std::set<compat::api_family> supported_families{};
  family_capabilities responses{};
  family_capabilities chat_completions{};
  family_capabilities completions{};
  family_capabilities embeddings{};
  family_capabilities models{};
  tool_capabilities tools{};
  modality_capabilities modalities{};
  structured_output_capabilities structured_outputs{};
  sampling_capabilities sampling{};
  std::optional<std::int64_t> max_tools{};
  std::optional<std::int64_t> max_stop_sequences{};
  std::optional<std::int64_t> max_n{};
  std::optional<std::int64_t> max_top_logprobs{};
  std::optional<std::int64_t> max_dimensions{};
  support_level raw_passthrough{support_level::native};
};

struct effective_profile {
  compat::api_family family{compat::api_family::unknown};
  std::string operation{};
  bool streaming{false};
  std::set<compat::api_family> supported_families{};
  family_capabilities responses{};
  family_capabilities chat_completions{};
  family_capabilities completions{};
  family_capabilities embeddings{};
  family_capabilities models{};
  tool_capabilities tools{};
  modality_capabilities modalities{};
  structured_output_capabilities structured_outputs{};
  sampling_capabilities sampling{};
  std::optional<std::int64_t> max_tools{};
  std::optional<std::int64_t> max_stop_sequences{};
  std::optional<std::int64_t> max_n{};
  std::optional<std::int64_t> max_top_logprobs{};
  std::optional<std::int64_t> max_dimensions{};
  support_level raw_passthrough{support_level::native};
  std::vector<std::string> diagnostics{};
};

struct validation_issue {
  std::string code{};
  std::string message{};
  std::string field{};
};

struct validation_report {
  bool ok{true};
  std::vector<validation_issue> issues{};
};

} // namespace openai::capabilities
