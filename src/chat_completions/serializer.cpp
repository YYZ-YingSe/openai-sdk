#include "openai/chat_completions/serializer.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "openai/core/error.hpp"
#include "openai/core/json.hpp"
#include "openai/responses/serializer.hpp"

namespace openai::chat_completions {
namespace {

using core::errc;
using core::make_error;
using core::parse_json;
using core::result;

using object_t = core::value::object;
using array_t = core::value::array;

auto to_json_value(std::string_view json_text) -> result<core::value> {
  auto document = parse_json(json_text);
  if (document.has_error()) {
    return result<core::value>::failure(document.error());
  }
  return core::json_to_value(document.value());
}

auto uint64_to_value(std::uint64_t current) -> result<core::value> {
  if (current >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return result<core::value>::failure(
        make_error(errc::not_supported, "uint64 value exceeds int64 range"));
  }
  return result<core::value>::success(
      core::value{static_cast<std::int64_t>(current)});
}

auto raw_json_to_value(const responses::raw_json &value) -> result<core::value> {
  if (!value.data.is_null() || value.json.empty()) {
    return result<core::value>::success(value.data);
  }
  return to_json_value(value.json);
}

auto insert_value(object_t &object, std::string_view key, core::value value)
    -> void {
  object.emplace(std::string{key}, std::move(value));
}

auto insert_optional_string(object_t &object, std::string_view key,
                            const std::optional<std::string> &value) -> void {
  if (value.has_value()) {
    insert_value(object, key, core::value{*value});
  }
}

auto insert_optional_bool(object_t &object, std::string_view key,
                          const std::optional<bool> &value) -> void {
  if (value.has_value()) {
    insert_value(object, key, core::value{*value});
  }
}

auto insert_optional_int64(object_t &object, std::string_view key,
                           const std::optional<std::int64_t> &value) -> void {
  if (value.has_value()) {
    insert_value(object, key, core::value{*value});
  }
}

auto insert_optional_uint64(object_t &object, std::string_view key,
                            const std::optional<std::uint64_t> &value)
    -> result<void> {
  if (!value.has_value()) {
    return result<void>::success();
  }
  auto converted = uint64_to_value(*value);
  if (converted.has_error()) {
    return result<void>::failure(converted.error());
  }
  insert_value(object, key, std::move(converted.value()));
  return result<void>::success();
}

auto insert_optional_double(object_t &object, std::string_view key,
                            const std::optional<double> &value) -> void {
  if (value.has_value()) {
    insert_value(object, key, core::value{*value});
  }
}

auto insert_string_map(object_t &object, std::string_view key,
                       const std::map<std::string, std::string> &values)
    -> void {
  if (values.empty()) {
    return;
  }
  object_t mapped{};
  for (const auto &[name, value] : values) {
    mapped.emplace(name, core::value{value});
  }
  insert_value(object, key, core::value{std::move(mapped)});
}

auto insert_string_array(object_t &object, std::string_view key,
                         const std::vector<std::string> &values) -> void {
  if (values.empty()) {
    return;
  }
  array_t mapped{};
  mapped.reserve(values.size());
  for (const auto &value : values) {
    mapped.emplace_back(value);
  }
  insert_value(object, key, core::value{std::move(mapped)});
}

auto serialize_stop_sequence(const stop_sequence &value)
    -> std::optional<core::value> {
  return std::visit(
      []<typename stop_t>(const stop_t &current) -> std::optional<core::value> {
        using current_t = std::decay_t<stop_t>;
        if constexpr (std::is_same_v<current_t, std::monostate>) {
          return std::nullopt;
        } else if constexpr (std::is_same_v<current_t, std::string>) {
          return core::value{current};
        } else {
          array_t mapped{};
          mapped.reserve(current.size());
          for (const auto &entry : current) {
            mapped.emplace_back(entry);
          }
          return core::value{std::move(mapped)};
        }
      },
      value);
}

auto serialize_response_format_value(const responses::response_text_config &value)
    -> result<core::value> {
  object_t object{};
  if (!value.type.empty()) {
    insert_value(object, "type", core::value{value.type});
  }
  if (value.schema_name.has_value() || value.schema_description.has_value() ||
      value.strict.has_value() || value.schema_json.has_value()) {
    object_t schema{};
    insert_optional_string(schema, "name", value.schema_name);
    insert_optional_string(schema, "description", value.schema_description);
    insert_optional_bool(schema, "strict", value.strict);
    if (value.schema_json.has_value()) {
      auto parsed = to_json_value(*value.schema_json);
      if (parsed.has_error()) {
        return result<core::value>::failure(parsed.error());
      }
      insert_value(schema, "schema", std::move(parsed.value()));
    }
    insert_value(object, "json_schema", core::value{std::move(schema)});
  }
  if (value.verbosity.has_value()) {
    insert_value(object, "verbosity", core::value{*value.verbosity});
  }
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_content_part_value(const content_part &value)
    -> result<core::value> {
  return std::visit(
      []<typename part_t>(const part_t &current) -> result<core::value> {
        using current_t = std::decay_t<part_t>;
        if constexpr (std::is_same_v<current_t, text_content_part>) {
          object_t object{};
          insert_value(object, "type", core::value{"text"});
          insert_value(object, "text", core::value{current.text});
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, image_content_part>) {
          object_t object{};
          insert_value(object, "type", core::value{"image_url"});
          object_t image{};
          insert_value(image, "url", core::value{current.url});
          insert_optional_string(image, "detail", current.detail);
          insert_value(object, "image_url", core::value{std::move(image)});
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, audio_content_part>) {
          object_t object{};
          insert_value(object, "type", core::value{"input_audio"});
          object_t audio{};
          insert_value(audio, "data", core::value{current.data});
          insert_value(audio, "format", core::value{current.format});
          insert_value(object, "input_audio", core::value{std::move(audio)});
          return result<core::value>::success(core::value{std::move(object)});
        } else {
          return raw_json_to_value(current);
        }
      },
      value);
}

auto serialize_tool_call_value(const tool_call &value) -> result<core::value> {
  object_t object{};
  insert_optional_string(object, "id", value.id);
  if (value.type.has_value()) {
    insert_value(object, "type", core::value{*value.type});
  } else if (value.name.has_value() || value.arguments.has_value()) {
    insert_value(object, "type", core::value{"function"});
  }
  if (value.name.has_value() || value.arguments.has_value()) {
    object_t function{};
    insert_optional_string(function, "name", value.name);
    insert_optional_string(function, "arguments", value.arguments);
    insert_value(object, "function", core::value{std::move(function)});
  }
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_message_value(const message &value) -> result<core::value> {
  object_t object{};
  insert_optional_string(object, "role", value.role);
  insert_optional_string(object, "refusal", value.refusal);
  insert_optional_string(object, "name", value.name);
  insert_optional_string(object, "tool_call_id", value.tool_call_id);

  if (const auto *text = std::get_if<std::string>(&value.content);
      text != nullptr) {
    insert_value(object, "content", core::value{*text});
  } else if (const auto *parts =
                 std::get_if<std::vector<content_part>>(&value.content);
             parts != nullptr) {
    array_t content{};
    content.reserve(parts->size());
    for (const auto &entry : *parts) {
      auto serialized = serialize_content_part_value(entry);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      content.push_back(std::move(serialized.value()));
    }
    insert_value(object, "content", core::value{std::move(content)});
  }

  if (!value.tool_calls.empty()) {
    array_t tool_calls{};
    tool_calls.reserve(value.tool_calls.size());
    for (const auto &entry : value.tool_calls) {
      auto serialized = serialize_tool_call_value(entry);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      tool_calls.push_back(std::move(serialized.value()));
    }
    insert_value(object, "tool_calls", core::value{std::move(tool_calls)});
  }

  if (value.function_call_value.has_value()) {
    object_t function{};
    insert_optional_string(function, "name", value.function_call_value->name);
    insert_optional_string(function, "arguments",
                           value.function_call_value->arguments);
    insert_value(object, "function_call", core::value{std::move(function)});
  }

  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_choice_value(const choice &value, const bool chunk)
    -> result<core::value> {
  object_t object{};
  insert_value(object, "index", core::value{value.index});

  if (chunk) {
    if (value.delta.has_value()) {
      auto serialized = serialize_message_value(*value.delta);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      insert_value(object, "delta", std::move(serialized.value()));
    } else if (value.message_value.has_value()) {
      auto serialized = serialize_message_value(*value.message_value);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      insert_value(object, "delta", std::move(serialized.value()));
    }
    if (value.finish_reason.has_value()) {
      insert_value(object, "finish_reason", core::value{*value.finish_reason});
    } else {
      insert_value(object, "finish_reason", core::value{nullptr});
    }
  } else {
    if (value.message_value.has_value()) {
      auto serialized = serialize_message_value(*value.message_value);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      insert_value(object, "message", std::move(serialized.value()));
    }
    insert_optional_string(object, "finish_reason", value.finish_reason);
    if (value.delta.has_value() && !value.message_value.has_value()) {
      auto serialized = serialize_message_value(*value.delta);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      insert_value(object, "message", std::move(serialized.value()));
    }
  }

  if (value.logprobs_value.has_value()) {
    insert_value(object, "logprobs", *value.logprobs_value);
  }

  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_usage_value(const usage &value) -> result<core::value> {
  object_t object{};
  insert_optional_int64(object, "prompt_tokens", value.prompt_tokens);
  insert_optional_int64(object, "completion_tokens", value.completion_tokens);
  insert_optional_int64(object, "total_tokens", value.total_tokens);
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_tool_definition_through_responses(
    const responses::tool_definition &value) -> result<core::value> {
  auto json = responses::serialize_tool_definition(value);
  if (json.has_error()) {
    return result<core::value>::failure(json.error());
  }
  return to_json_value(json.value());
}

auto serialize_tool_choice_through_responses(const responses::tool_choice &value)
    -> result<core::value> {
  auto json = responses::serialize_tool_choice(value);
  if (json.has_error()) {
    return result<core::value>::failure(json.error());
  }
  return to_json_value(json.value());
}

auto serialize_request_value(const request &value) -> result<core::value> {
  object_t object{};
  insert_optional_string(object, "model", value.model);
  if (!value.messages.empty()) {
    array_t messages{};
    messages.reserve(value.messages.size());
    for (const auto &entry : value.messages) {
      auto serialized = serialize_message_value(entry);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      messages.push_back(std::move(serialized.value()));
    }
    insert_value(object, "messages", core::value{std::move(messages)});
  }
  if (!value.tools.empty()) {
    array_t tools{};
    tools.reserve(value.tools.size());
    for (const auto &entry : value.tools) {
      auto serialized = serialize_tool_definition_through_responses(entry);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      tools.push_back(std::move(serialized.value()));
    }
    insert_value(object, "tools", core::value{std::move(tools)});
  }
  if (value.tool_choice_value.has_value()) {
    auto serialized =
        serialize_tool_choice_through_responses(*value.tool_choice_value);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "tool_choice", std::move(serialized.value()));
  }
  if (value.response_format.has_value()) {
    auto serialized = serialize_response_format_value(*value.response_format);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "response_format", std::move(serialized.value()));
  }
  if (value.audio.has_value()) {
    insert_value(object, "audio", *value.audio);
  }
  if (value.function_call.has_value()) {
    insert_value(object, "function_call", *value.function_call);
  }
  if (!value.functions.empty()) {
    array_t functions{};
    functions.reserve(value.functions.size());
    for (const auto &entry : value.functions) {
      functions.push_back(entry);
    }
    insert_value(object, "functions", core::value{std::move(functions)});
  }
  insert_optional_double(object, "frequency_penalty", value.frequency_penalty);
  if (value.logit_bias.has_value()) {
    insert_value(object, "logit_bias", *value.logit_bias);
  }
  insert_optional_bool(object, "logprobs", value.logprobs);
  insert_optional_int64(object, "max_completion_tokens",
                        value.max_completion_tokens);
  insert_optional_int64(object, "max_tokens", value.max_tokens);
  insert_string_array(object, "modalities", value.modalities);
  insert_optional_int64(object, "n", value.n);
  insert_optional_bool(object, "parallel_tool_calls",
                       value.parallel_tool_calls);
  if (value.prediction.has_value()) {
    insert_value(object, "prediction", *value.prediction);
  }
  insert_optional_double(object, "presence_penalty", value.presence_penalty);
  insert_optional_string(object, "prompt_cache_key", value.prompt_cache_key);
  insert_optional_string(object, "prompt_cache_retention",
                         value.prompt_cache_retention);
  insert_optional_string(object, "reasoning_effort", value.reasoning_effort);
  insert_optional_string(object, "safety_identifier",
                         value.safety_identifier);
  insert_optional_int64(object, "seed", value.seed);
  insert_optional_string(object, "service_tier", value.service_tier);
  if (const auto stop = serialize_stop_sequence(value.stop); stop.has_value()) {
    insert_value(object, "stop", *stop);
  }
  insert_optional_bool(object, "store", value.store);
  insert_optional_double(object, "temperature", value.temperature);
  insert_optional_int64(object, "top_logprobs", value.top_logprobs);
  insert_optional_double(object, "top_p", value.top_p);
  insert_optional_bool(object, "stream", value.stream);
  if (value.stream_options_value.has_value()) {
    object_t stream_options{};
    insert_optional_bool(stream_options, "include_obfuscation",
                         value.stream_options_value->include_obfuscation);
    insert_optional_bool(stream_options, "include_usage",
                         value.stream_options_value->include_usage);
    insert_value(object, "stream_options",
                 core::value{std::move(stream_options)});
  }
  insert_string_map(object, "metadata", value.metadata);
  insert_optional_string(object, "user", value.user);
  if (value.web_search_options.has_value()) {
    insert_value(object, "web_search_options", *value.web_search_options);
  }

  return result<core::value>::success(core::value{std::move(object)});
}

template <typename output_t>
auto serialize_completion_like(const output_t &value, const bool chunk)
    -> result<core::value> {
  object_t object{};
  insert_optional_string(object, "id", value.id);
  insert_optional_string(object, "object", value.object);
  insert_optional_string(object, "model", value.model);
  if (auto inserted = insert_optional_uint64(object, "created", value.created);
      inserted.has_error()) {
    return result<core::value>::failure(inserted.error());
  }
  insert_optional_string(object, "service_tier", value.service_tier);
  insert_optional_string(object, "system_fingerprint",
                         value.system_fingerprint);
  if (!value.choices.empty()) {
    array_t choices{};
    choices.reserve(value.choices.size());
    for (const auto &entry : value.choices) {
      auto serialized = serialize_choice_value(entry, chunk);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      choices.push_back(std::move(serialized.value()));
    }
    insert_value(object, "choices", core::value{std::move(choices)});
  }
  if (value.usage_value.has_value()) {
    auto serialized = serialize_usage_value(*value.usage_value);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "usage", std::move(serialized.value()));
  }
  return result<core::value>::success(core::value{std::move(object)});
}

} // namespace

auto serialize_request(const request &value,
                       const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_request_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_completion(const completion &value,
                          const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_completion_like(value, false);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_completion_chunk(const completion_chunk &value,
                                const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_completion_like(value, true);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

} // namespace openai::chat_completions
