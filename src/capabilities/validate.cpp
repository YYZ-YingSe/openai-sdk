#include "openai/capabilities/validate.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

namespace openai::capabilities {
namespace {

auto add_issue(validation_report &report, std::string code, std::string message,
               std::string field) -> void {
  report.ok = false;
  report.issues.push_back(
      validation_issue{std::move(code), std::move(message), std::move(field)});
}

auto family_capabilities_for(const effective_profile &profile,
                             const compat::api_family family)
    -> const family_capabilities & {
  switch (family) {
  case compat::api_family::responses:
    return profile.responses;
  case compat::api_family::chat_completions:
    return profile.chat_completions;
  case compat::api_family::completions:
    return profile.completions;
  case compat::api_family::embeddings:
    return profile.embeddings;
  case compat::api_family::models:
    return profile.models;
  case compat::api_family::unknown:
    break;
  }
  return profile.responses;
}

auto validate_family_support(validation_report &report,
                             const effective_profile &profile,
                             const compat::api_family family) -> void {
  if (family == compat::api_family::unknown) {
    return;
  }
  if (profile.supported_families.count(family) == 0U) {
    add_issue(report, "family.unsupported",
              "requested API family is not supported by the effective profile",
              "family");
  }
}

auto validate_positive(validation_report &report,
                       const std::optional<std::int64_t> &value,
                       std::string_view field) -> void {
  if (value.has_value() && *value <= 0) {
    add_issue(report, "value.non_positive",
              std::string{field} + " must be greater than zero",
              std::string{field});
  }
}

auto validate_non_negative(validation_report &report,
                           const std::optional<std::int64_t> &value,
                           std::string_view field) -> void {
  if (value.has_value() && *value < 0) {
    add_issue(report, "value.negative",
              std::string{field} + " must be non-negative",
              std::string{field});
  }
}

auto validate_finite(validation_report &report,
                     const std::optional<double> &value,
                     std::string_view field) -> void {
  if (value.has_value() && !std::isfinite(*value)) {
    add_issue(report, "value.non_finite",
              std::string{field} + " must be finite", std::string{field});
  }
}

auto validate_stream_options_dependency(validation_report &report,
                                        const bool has_stream_options,
                                        const std::optional<bool> &stream,
                                        std::string_view field) -> void {
  if (has_stream_options && stream.value_or(false) == false) {
    add_issue(report, "stream_options.requires_stream",
              std::string{field} + " requires stream=true",
              std::string{field});
  }
}

auto validate_raw_passthrough(validation_report &report,
                              const effective_profile &profile,
                              std::string_view field) -> void {
  if (!is_supported(profile.raw_passthrough)) {
    add_issue(report, "raw_passthrough.unsupported",
              std::string{field} +
                  " requires raw passthrough support in the effective profile",
              std::string{field});
  }
}

auto validate_tool(validation_report &report, const effective_profile &profile,
                   const responses::tool_definition &tool) -> void {
  std::visit(
      [&](const auto &current) {
        using current_t = std::decay_t<decltype(current)>;
        if constexpr (std::is_same_v<current_t, responses::function_tool>) {
          if (!is_supported(profile.tools.function_tools)) {
            add_issue(report, "tool.function.unsupported",
                      "function tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t,
                                            responses::file_search_tool>) {
          if (!is_supported(profile.tools.file_search)) {
            add_issue(report, "tool.file_search.unsupported",
                      "file_search tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t,
                                            responses::tool_search_tool>) {
          if (!is_supported(profile.tools.tool_search)) {
            add_issue(report, "tool.tool_search.unsupported",
                      "tool_search tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t,
                                            responses::web_search_tool>) {
          if (!is_supported(profile.tools.web_search)) {
            add_issue(report, "tool.web_search.unsupported",
                      "web_search tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t,
                                            responses::computer_tool>) {
          if (!is_supported(profile.tools.computer_use)) {
            add_issue(report, "tool.computer_use.unsupported",
                      "computer_use tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t,
                                            responses::namespace_tool>) {
          if (!is_supported(profile.tools.namespace_tools)) {
            add_issue(report, "tool.namespace.unsupported",
                      "namespace tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t,
                                            responses::local_shell_tool>) {
          if (!is_supported(profile.tools.local_shell)) {
            add_issue(report, "tool.local_shell.unsupported",
                      "local_shell tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t, responses::shell_tool>) {
          if (!is_supported(profile.tools.shell)) {
            add_issue(report, "tool.shell.unsupported",
                      "shell tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t,
                                            responses::function_shell_tool>) {
          if (!is_supported(profile.tools.function_shell)) {
            add_issue(report, "tool.function_shell.unsupported",
                      "function_shell tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t,
                                            responses::custom_tool>) {
          if (!is_supported(profile.tools.custom_tools)) {
            add_issue(report, "tool.custom.unsupported",
                      "custom tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<
                                 current_t, responses::image_generation_tool>) {
          if (!is_supported(profile.tools.image_generation)) {
            add_issue(report, "tool.image_generation.unsupported",
                      "image_generation tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<
                                 current_t, responses::code_interpreter_tool>) {
          if (!is_supported(profile.tools.code_interpreter)) {
            add_issue(report, "tool.code_interpreter.unsupported",
                      "code_interpreter tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t, responses::mcp_tool>) {
          if (!is_supported(profile.tools.mcp)) {
            add_issue(report, "tool.mcp.unsupported",
                      "mcp tools are unsupported", "tools");
          }
        } else if constexpr (std::is_same_v<current_t,
                                            responses::apply_patch_tool>) {
          if (!is_supported(profile.tools.apply_patch)) {
            add_issue(report, "tool.apply_patch.unsupported",
                      "apply_patch tools are unsupported", "tools");
          }
        } else {
          validate_raw_passthrough(report, profile, "tools");
        }
      },
      tool);
}

auto validate_responses_message_content(validation_report &report,
                                        const effective_profile &profile,
                                        const responses::message_content_part &part)
    -> void {
  std::visit(
      [&](const auto &current) {
        using current_t = std::decay_t<decltype(current)>;
        if constexpr (std::is_same_v<current_t, responses::input_image_content>) {
          if (!is_supported(profile.modalities.image_input)) {
            add_issue(report, "modality.image_input.unsupported",
                      "image input is unsupported", "input");
          }
        } else if constexpr (std::is_same_v<current_t, responses::raw_json>) {
          validate_raw_passthrough(report, profile, "input");
        }
      },
      part);
}

auto validate_responses_input(validation_report &report,
                              const effective_profile &profile,
                              const std::optional<responses::input_payload> &input)
    -> void {
  if (!input.has_value()) {
    return;
  }
  if (const auto *items = std::get_if<std::vector<responses::item>>(&*input);
      items != nullptr) {
    for (const auto &item : *items) {
      if (const auto *message = std::get_if<responses::message_item>(&item);
          message != nullptr) {
        for (const auto &part : message->content) {
          validate_responses_message_content(report, profile, part);
        }
      } else if (std::holds_alternative<responses::raw_json>(item)) {
        validate_raw_passthrough(report, profile, "input");
      }
    }
  }
}

auto validate_structured_output(validation_report &report,
                                const effective_profile &profile,
                                const responses::response_text_config &text,
                                std::string_view field) -> void {
  if (text.schema_json.has_value() && !is_supported(profile.structured_outputs.json_schema)) {
    add_issue(report, "structured_output.json_schema.unsupported",
              "json_schema structured outputs are unsupported",
              std::string{field});
  }
  if (text.strict.value_or(false) &&
      !is_supported(profile.structured_outputs.strict)) {
    add_issue(report, "structured_output.strict.unsupported",
              "strict structured outputs are unsupported",
              std::string{field});
  }
}

auto validate_chat_message_content(validation_report &report,
                                   const effective_profile &profile,
                                   const chat_completions::message &message)
    -> void {
  if (const auto *parts =
          std::get_if<std::vector<chat_completions::content_part>>(
              &message.content);
      parts != nullptr) {
    for (const auto &part : *parts) {
      std::visit(
          [&](const auto &current) {
            using current_t = std::decay_t<decltype(current)>;
            if constexpr (std::is_same_v<current_t,
                                         chat_completions::image_content_part>) {
              if (!is_supported(profile.modalities.image_input)) {
                add_issue(report, "modality.image_input.unsupported",
                          "image input is unsupported", "messages");
              }
            } else if constexpr (std::is_same_v<
                                     current_t,
                                     chat_completions::audio_content_part>) {
              if (!is_supported(profile.modalities.audio_input)) {
                add_issue(report, "modality.audio_input.unsupported",
                          "audio input is unsupported", "messages");
              }
            } else if constexpr (std::is_same_v<current_t,
                                                responses::raw_json>) {
              validate_raw_passthrough(report, profile, "messages");
            }
          },
          part);
    }
  }
}

auto infer_family(const compat::request_document &request,
                  const compat::operation_hint &hint) -> compat::api_family {
  if (hint.family != compat::api_family::unknown) {
    return hint.family;
  }
  if (std::holds_alternative<responses::request>(request)) {
    return compat::api_family::responses;
  }
  if (std::holds_alternative<chat_completions::request>(request)) {
    return compat::api_family::chat_completions;
  }
  if (std::holds_alternative<completions::request>(request)) {
    return compat::api_family::completions;
  }
  if (std::holds_alternative<embeddings::request>(request)) {
    return compat::api_family::embeddings;
  }
  return compat::api_family::unknown;
}

} // namespace

auto validate_request(const responses::request &request,
                      const effective_profile &profile,
                      const compat::api_family family) -> validation_report {
  validation_report report{};
  validate_family_support(report, profile, family);
  const auto &family_cap = family_capabilities_for(profile, family);
  if (!is_supported(family_cap.request_support)) {
    add_issue(report, "request.unsupported",
              "requests are unsupported for this family", "request");
  }
  validate_positive(report, request.max_output_tokens, "max_output_tokens");
  validate_positive(report, request.max_tool_calls, "max_tool_calls");
  if (request.top_logprobs.has_value()) {
    validate_non_negative(
        report,
        std::optional<std::int64_t>{
            static_cast<std::int64_t>(*request.top_logprobs)},
        "top_logprobs");
  }
  validate_finite(report, request.temperature, "temperature");
  validate_finite(report, request.top_p, "top_p");
  validate_stream_options_dependency(report, request.stream_options_value.has_value(),
                                     request.stream, "stream_options");
  if (request.stream.value_or(false) && !is_supported(family_cap.streaming)) {
    add_issue(report, "streaming.unsupported",
              "streaming is unsupported for this family", "stream");
  }
  if (request.parallel_tool_calls.value_or(false) &&
      !is_supported(family_cap.parallel_tool_calls)) {
    add_issue(report, "parallel_tool_calls.unsupported",
              "parallel_tool_calls are unsupported", "parallel_tool_calls");
  }
  if ((request.tool_choice_value.has_value() ||
       request.parallel_tool_calls.value_or(false) ||
       request.max_tool_calls.has_value()) &&
      request.tools.empty()) {
    add_issue(report, "tools.required",
              "tools must be present when tool control fields are set", "tools");
  }
  if (!request.tools.empty()) {
    if (!is_supported(family_cap.tools)) {
      add_issue(report, "tools.unsupported",
                "tools are unsupported for this family", "tools");
    }
    for (const auto &tool : request.tools) {
      validate_tool(report, profile, tool);
    }
    if (profile.max_tools.has_value() &&
        static_cast<std::int64_t>(request.tools.size()) > *profile.max_tools) {
      add_issue(report, "tools.limit_exceeded",
                "tool count exceeds effective profile limit", "tools");
    }
  }
  if (request.text.has_value()) {
    validate_structured_output(report, profile, *request.text, "text");
  }
  if (request.conversation.has_value() || request.prompt.has_value()) {
    validate_raw_passthrough(report, profile, "conversation");
  }
  validate_responses_input(report, profile, request.input);
  report.ok = report.issues.empty();
  return report;
}

auto validate_request(const chat_completions::request &request,
                      const effective_profile &profile,
                      const compat::api_family family) -> validation_report {
  validation_report report{};
  validate_family_support(report, profile, family);
  const auto &family_cap = family_capabilities_for(profile, family);
  if (!is_supported(family_cap.request_support)) {
    add_issue(report, "request.unsupported",
              "requests are unsupported for this family", "request");
  }
  if (request.messages.empty()) {
    add_issue(report, "messages.required",
              "chat_completions requests require at least one message",
              "messages");
  }
  validate_positive(report, request.max_completion_tokens,
                    "max_completion_tokens");
  validate_positive(report, request.max_tokens, "max_tokens");
  validate_positive(report, request.n, "n");
  validate_positive(report, request.seed, "seed");
  validate_non_negative(report, request.top_logprobs, "top_logprobs");
  validate_finite(report, request.frequency_penalty, "frequency_penalty");
  validate_finite(report, request.presence_penalty, "presence_penalty");
  validate_finite(report, request.temperature, "temperature");
  validate_finite(report, request.top_p, "top_p");
  validate_stream_options_dependency(report, request.stream_options_value.has_value(),
                                     request.stream, "stream_options");
  if (request.stream.value_or(false) && !is_supported(family_cap.streaming)) {
    add_issue(report, "streaming.unsupported",
              "streaming is unsupported for this family", "stream");
  }
  if (request.parallel_tool_calls.value_or(false) &&
      !is_supported(family_cap.parallel_tool_calls)) {
    add_issue(report, "parallel_tool_calls.unsupported",
              "parallel_tool_calls are unsupported", "parallel_tool_calls");
  }
  if (request.top_logprobs.has_value() && request.logprobs.value_or(false) == false) {
    add_issue(report, "top_logprobs.requires_logprobs",
              "top_logprobs requires logprobs=true", "top_logprobs");
  }
  if ((request.tool_choice_value.has_value() ||
       request.parallel_tool_calls.value_or(false)) &&
      request.tools.empty()) {
    add_issue(report, "tools.required",
              "tools must be present when tool control fields are set", "tools");
  }
  if (!request.tools.empty()) {
    if (!is_supported(family_cap.tools)) {
      add_issue(report, "tools.unsupported",
                "tools are unsupported for this family", "tools");
    }
    for (const auto &tool : request.tools) {
      validate_tool(report, profile, tool);
    }
    if (profile.max_tools.has_value() &&
        static_cast<std::int64_t>(request.tools.size()) > *profile.max_tools) {
      add_issue(report, "tools.limit_exceeded",
                "tool count exceeds effective profile limit", "tools");
    }
  }
  if (request.response_format.has_value()) {
    validate_structured_output(report, profile, *request.response_format,
                               "response_format");
  }
  if (request.audio.has_value() &&
      !is_supported(profile.modalities.audio_output)) {
    add_issue(report, "modality.audio_output.unsupported",
              "audio output configuration is unsupported", "audio");
  }
  for (const auto &message : request.messages) {
    validate_chat_message_content(report, profile, message);
  }
  for (const auto &modality : request.modalities) {
    if (modality == "audio" &&
        !is_supported(profile.modalities.audio_output)) {
      add_issue(report, "modality.audio_output.unsupported",
                "audio output modality is unsupported", "modalities");
    }
    if (modality == "text" &&
        !is_supported(profile.modalities.text_output)) {
      add_issue(report, "modality.text_output.unsupported",
                "text output modality is unsupported", "modalities");
    }
  }
  report.ok = report.issues.empty();
  return report;
}

auto validate_request(const completions::request &request,
                      const effective_profile &profile,
                      const compat::api_family family) -> validation_report {
  validation_report report{};
  validate_family_support(report, profile, family);
  const auto &family_cap = family_capabilities_for(profile, family);
  if (!is_supported(family_cap.request_support)) {
    add_issue(report, "request.unsupported",
              "requests are unsupported for this family", "request");
  }
  if (request.model.empty()) {
    add_issue(report, "model.required", "model is required", "model");
  }
  validate_positive(report, request.best_of, "best_of");
  validate_positive(report, request.max_tokens, "max_tokens");
  validate_positive(report, request.n, "n");
  validate_positive(report, request.seed, "seed");
  validate_non_negative(report, request.logprobs, "logprobs");
  validate_finite(report, request.frequency_penalty, "frequency_penalty");
  validate_finite(report, request.presence_penalty, "presence_penalty");
  validate_finite(report, request.temperature, "temperature");
  validate_finite(report, request.top_p, "top_p");
  validate_stream_options_dependency(report, request.stream_options_value.has_value(),
                                     request.stream, "stream_options");
  if (request.stream.value_or(false) && !is_supported(family_cap.streaming)) {
    add_issue(report, "streaming.unsupported",
              "streaming is unsupported for this family", "stream");
  }
  if (request.best_of.has_value() && request.n.has_value() &&
      *request.best_of < *request.n) {
    add_issue(report, "best_of.lt_n",
              "best_of must be greater than or equal to n", "best_of");
  }
  if (request.temperature.has_value() &&
      !is_supported(profile.sampling.temperature)) {
    add_issue(report, "sampling.temperature.unsupported",
              "temperature is unsupported", "temperature");
  }
  if (request.top_p.has_value() && !is_supported(profile.sampling.top_p)) {
    add_issue(report, "sampling.top_p.unsupported", "top_p is unsupported",
              "top_p");
  }
  if (request.n.has_value() && !is_supported(profile.sampling.n)) {
    add_issue(report, "sampling.n.unsupported", "n is unsupported", "n");
  }
  if (!std::holds_alternative<std::monostate>(request.stop) &&
      !is_supported(profile.sampling.stop)) {
    add_issue(report, "sampling.stop.unsupported", "stop is unsupported",
              "stop");
  }
  if (request.logprobs.has_value() &&
      !is_supported(profile.sampling.logprobs)) {
    add_issue(report, "sampling.logprobs.unsupported",
              "logprobs is unsupported", "logprobs");
  }
  if (request.best_of.has_value() && !is_supported(profile.sampling.best_of)) {
    add_issue(report, "sampling.best_of.unsupported",
              "best_of is unsupported", "best_of");
  }
  if (request.max_tokens.has_value() &&
      !is_supported(profile.sampling.max_tokens)) {
    add_issue(report, "sampling.max_tokens.unsupported",
              "max_tokens is unsupported", "max_tokens");
  }
  if (request.seed.has_value() && !is_supported(profile.sampling.seed)) {
    add_issue(report, "sampling.seed.unsupported", "seed is unsupported",
              "seed");
  }
  report.ok = report.issues.empty();
  return report;
}

auto validate_request(const embeddings::request &request,
                      const effective_profile &profile,
                      const compat::api_family family) -> validation_report {
  validation_report report{};
  validate_family_support(report, profile, family);
  const auto &family_cap = family_capabilities_for(profile, family);
  if (!is_supported(family_cap.request_support)) {
    add_issue(report, "request.unsupported",
              "requests are unsupported for this family", "request");
  }
  if (request.model.empty()) {
    add_issue(report, "model.required", "model is required", "model");
  }
  validate_positive(report, request.dimensions, "dimensions");
  if (request.dimensions.has_value() && profile.max_dimensions.has_value() &&
      *request.dimensions > *profile.max_dimensions) {
    add_issue(report, "dimensions.limit_exceeded",
              "dimensions exceeds effective profile limit", "dimensions");
  }
  report.ok = report.issues.empty();
  return report;
}

auto validate_request(const compat::request_document &request,
                      const effective_profile &profile,
                      const compat::operation_hint &hint) -> validation_report {
  switch (infer_family(request, hint)) {
  case compat::api_family::responses:
    if (const auto *typed = std::get_if<responses::request>(&request);
        typed != nullptr) {
      return validate_request(*typed, profile, compat::api_family::responses);
    }
    break;
  case compat::api_family::chat_completions:
    if (const auto *typed = std::get_if<chat_completions::request>(&request);
        typed != nullptr) {
      return validate_request(*typed, profile,
                              compat::api_family::chat_completions);
    }
    break;
  case compat::api_family::completions:
    if (const auto *typed = std::get_if<completions::request>(&request);
        typed != nullptr) {
      return validate_request(*typed, profile, compat::api_family::completions);
    }
    break;
  case compat::api_family::embeddings:
    if (const auto *typed = std::get_if<embeddings::request>(&request);
        typed != nullptr) {
      return validate_request(*typed, profile, compat::api_family::embeddings);
    }
    break;
  case compat::api_family::models:
  case compat::api_family::unknown:
    break;
  }

  validation_report report{};
  validate_family_support(report, profile, infer_family(request, hint));
  if (const auto *generic = std::get_if<compat::generic_object>(&request);
      generic != nullptr) {
    if (!is_supported(profile.raw_passthrough)) {
      add_issue(report, "raw_passthrough.unsupported",
                "generic request passthrough is unsupported", "request");
    }
    if (!generic->data.is_object()) {
      add_issue(report, "generic_request.invalid",
                "generic request document must be an object", "request");
    }
  }
  report.ok = report.issues.empty();
  return report;
}

} // namespace openai::capabilities
