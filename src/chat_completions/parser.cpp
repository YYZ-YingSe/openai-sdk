#include "openai/chat_completions/parser.hpp"

#include <utility>

#include "openai/core/json.hpp"
#include "openai/responses/parser.hpp"

namespace openai::chat_completions {
namespace {

using core::errc;
using core::json_member;
using core::json_to_string;
using core::make_error;
using core::parse_json;
using core::result;

[[nodiscard]] auto parse_optional_core_value_field(const core::json_value &value,
                                                   std::string_view key)
    -> result<std::optional<core::value>> {
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::optional<core::value>>::success(std::nullopt);
  }
  auto converted = core::json_to_value(*member);
  if (converted.has_error()) {
    return result<std::optional<core::value>>::failure(converted.error());
  }
  return result<std::optional<core::value>>::success(std::move(converted.value()));
}

[[nodiscard]] auto parse_stop_sequence_value(const core::json_value &value)
    -> result<stop_sequence> {
  if (value.IsNull()) {
    return result<stop_sequence>::success(stop_sequence{std::monostate{}});
  }
  if (value.IsString()) {
    return result<stop_sequence>::success(stop_sequence{core::json_string(value)});
  }
  if (!value.IsArray()) {
    return result<stop_sequence>::failure(make_error(
        errc::type_mismatch, "stop must be null, string, or array of strings"));
  }
  std::vector<std::string> sequences{};
  sequences.reserve(value.Size());
  for (const auto &entry : value.GetArray()) {
    if (!entry.IsString()) {
      return result<stop_sequence>::failure(make_error(
          errc::type_mismatch, "stop array entries must be strings"));
    }
    sequences.push_back(core::json_string(entry));
  }
  return result<stop_sequence>::success(stop_sequence{std::move(sequences)});
}

[[nodiscard]] auto parse_stream_options_value(const core::json_value &value)
    -> result<stream_options> {
  if (!value.IsObject()) {
    return result<stream_options>::failure(make_error(
        errc::type_mismatch, "stream_options must be an object"));
  }
  stream_options parsed{};
  if (auto field = core::optional_bool(value, "include_obfuscation");
      field.has_error()) {
    return result<stream_options>::failure(field.error());
  } else {
    parsed.include_obfuscation = field.value();
  }
  if (auto field = core::optional_bool(value, "include_usage"); field.has_error()) {
    return result<stream_options>::failure(field.error());
  } else {
    parsed.include_usage = field.value();
  }
  return result<stream_options>::success(std::move(parsed));
}

[[nodiscard]] auto optional_response_format(const core::json_value &value)
    -> result<std::optional<responses::response_text_config>> {
  const auto *member = json_member(value, "response_format");
  if (member == nullptr || member->IsNull()) {
    return result<std::optional<responses::response_text_config>>::success(
        std::nullopt);
  }
  if (!member->IsObject()) {
    return result<std::optional<responses::response_text_config>>::failure(
        make_error(errc::type_mismatch,
                   "response_format must be an object"));
  }

  responses::response_text_config parsed{};
  if (auto type = core::optional_string(*member, "type"); type.has_error()) {
    return result<std::optional<responses::response_text_config>>::failure(
        type.error());
  } else if (type.value().has_value()) {
    parsed.type = std::move(*type.value());
  }

  const auto *schema = json_member(*member, "json_schema");
  if (schema != nullptr && schema->IsObject()) {
    if (auto field = core::optional_string(*schema, "name"); field.has_error()) {
      return result<std::optional<responses::response_text_config>>::failure(
          field.error());
    } else {
      parsed.schema_name = std::move(field.value());
    }
    if (auto field = core::optional_string(*schema, "description");
        field.has_error()) {
      return result<std::optional<responses::response_text_config>>::failure(
          field.error());
    } else {
      parsed.schema_description = std::move(field.value());
    }
    if (auto field = core::optional_bool(*schema, "strict"); field.has_error()) {
      return result<std::optional<responses::response_text_config>>::failure(
          field.error());
    } else {
      parsed.strict = field.value();
    }
    const auto *schema_body = json_member(*schema, "schema");
    if (schema_body != nullptr && !schema_body->IsNull()) {
      auto serialized = json_to_string(*schema_body);
      if (serialized.has_error()) {
        return result<std::optional<responses::response_text_config>>::failure(
            serialized.error());
      }
      parsed.schema_json = std::move(serialized.value());
    }
  }

  return result<std::optional<responses::response_text_config>>::success(
      std::move(parsed));
}

[[nodiscard]] auto parse_content_part_value(const core::json_value &value)
    -> result<content_part> {
  if (value.IsString()) {
    return result<content_part>::success(content_part{text_content_part{
        core::json_string(value)}});
  }
  if (!value.IsObject()) {
    return result<content_part>::failure(
        make_error(errc::type_mismatch, "chat content part must be an object"));
  }

  auto type = core::optional_string(value, "type");
  if (type.has_error()) {
    return result<content_part>::failure(type.error());
  }
  const auto kind = type.value().value_or("text");

  if (kind == "text") {
    auto text = core::required_string(value, "text");
    if (text.has_error()) {
      return result<content_part>::failure(text.error());
    }
    return result<content_part>::success(
        content_part{text_content_part{std::move(text.value())}});
  }
  if (kind == "image_url") {
    image_content_part parsed{};
    const auto *image = json_member(value, "image_url");
    if (image != nullptr) {
      if (image->IsString()) {
        parsed.url = core::json_string(*image);
      } else if (image->IsObject()) {
        auto url = core::required_string(*image, "url");
        if (url.has_error()) {
          return result<content_part>::failure(url.error());
        }
        parsed.url = std::move(url.value());
        if (auto detail = core::optional_string(*image, "detail");
            detail.has_error()) {
          return result<content_part>::failure(detail.error());
        } else {
          parsed.detail = std::move(detail.value());
        }
      } else {
        return result<content_part>::failure(make_error(
            errc::type_mismatch, "image_url must be a string or object"));
      }
    }
    return result<content_part>::success(content_part{std::move(parsed)});
  }
  if (kind == "input_audio") {
    audio_content_part parsed{};
    const auto *audio = json_member(value, "input_audio");
    if (audio != nullptr && audio->IsObject()) {
      auto data = core::required_string(*audio, "data");
      if (data.has_error()) {
        return result<content_part>::failure(data.error());
      }
      parsed.data = std::move(data.value());
      auto format = core::required_string(*audio, "format");
      if (format.has_error()) {
        return result<content_part>::failure(format.error());
      }
      parsed.format = std::move(format.value());
    } else {
      auto data = core::required_string(value, "data");
      if (data.has_error()) {
        return result<content_part>::failure(data.error());
      }
      parsed.data = std::move(data.value());
      auto format = core::required_string(value, "format");
      if (format.has_error()) {
        return result<content_part>::failure(format.error());
      }
      parsed.format = std::move(format.value());
    }
    return result<content_part>::success(content_part{std::move(parsed)});
  }

  auto serialized = json_to_string(value);
  if (serialized.has_error()) {
    return result<content_part>::failure(serialized.error());
  }
  auto structured = core::json_to_value(value);
  if (structured.has_error()) {
    return result<content_part>::failure(structured.error());
  }
  return result<content_part>::success(content_part{responses::raw_json{
      kind, std::move(serialized.value()), std::move(structured.value())}});
}

[[nodiscard]] auto parse_tool_call_value(const core::json_value &value)
    -> result<tool_call> {
  if (!value.IsObject()) {
    return result<tool_call>::failure(
        make_error(errc::type_mismatch, "tool call must be an object"));
  }

  tool_call parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<tool_call>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());

  if (auto field = core::optional_string(value, "id"); field.has_error()) {
    return result<tool_call>::failure(field.error());
  } else {
    parsed.id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "type"); field.has_error()) {
    return result<tool_call>::failure(field.error());
  } else {
    parsed.type = std::move(field.value());
  }

  const auto *function = json_member(value, "function");
  if (function != nullptr && function->IsObject()) {
    if (auto field = core::optional_string(*function, "name"); field.has_error()) {
      return result<tool_call>::failure(field.error());
    } else {
      parsed.name = std::move(field.value());
    }
    if (auto field = core::optional_string(*function, "arguments");
        field.has_error()) {
      return result<tool_call>::failure(field.error());
    } else {
      parsed.arguments = std::move(field.value());
    }
  }

  return result<tool_call>::success(std::move(parsed));
}

[[nodiscard]] auto parse_message_value(const core::json_value &value)
    -> result<message> {
  if (!value.IsObject()) {
    return result<message>::failure(
        make_error(errc::type_mismatch, "chat message must be an object"));
  }

  message parsed{};
  if (auto field = core::optional_string(value, "role"); field.has_error()) {
    return result<message>::failure(field.error());
  } else {
    parsed.role = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "refusal"); field.has_error()) {
    return result<message>::failure(field.error());
  } else {
    parsed.refusal = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "name"); field.has_error()) {
    return result<message>::failure(field.error());
  } else {
    parsed.name = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "tool_call_id");
      field.has_error()) {
    return result<message>::failure(field.error());
  } else {
    parsed.tool_call_id = std::move(field.value());
  }

  const auto *content = json_member(value, "content");
  if (content != nullptr && !content->IsNull()) {
    if (content->IsString()) {
      parsed.content = core::json_string(*content);
    } else if (content->IsArray()) {
      std::vector<content_part> parts;
      parts.reserve(content->Size());
      for (const auto &entry : content->GetArray()) {
        auto part = parse_content_part_value(entry);
        if (part.has_error()) {
          return result<message>::failure(part.error());
        }
        parts.push_back(std::move(part.value()));
      }
      parsed.content = std::move(parts);
    } else {
      return result<message>::failure(make_error(
          errc::type_mismatch, "chat message content must be a string or array"));
    }
  }

  const auto *tool_calls = json_member(value, "tool_calls");
  if (tool_calls != nullptr && !tool_calls->IsNull()) {
    if (!tool_calls->IsArray()) {
      return result<message>::failure(
          make_error(errc::type_mismatch, "tool_calls must be an array"));
    }
    parsed.tool_calls.reserve(tool_calls->Size());
    for (const auto &entry : tool_calls->GetArray()) {
      auto call = parse_tool_call_value(entry);
      if (call.has_error()) {
        return result<message>::failure(call.error());
      }
      parsed.tool_calls.push_back(std::move(call.value()));
    }
  }

  const auto *function = json_member(value, "function_call");
  if (function != nullptr && !function->IsNull()) {
    if (!function->IsObject()) {
      return result<message>::failure(make_error(
          errc::type_mismatch, "function_call must be an object"));
    }
    function_call parsed_function{};
    if (auto field = core::optional_string(*function, "name"); field.has_error()) {
      return result<message>::failure(field.error());
    } else {
      parsed_function.name = std::move(field.value());
    }
    if (auto field = core::optional_string(*function, "arguments");
        field.has_error()) {
      return result<message>::failure(field.error());
    } else {
      parsed_function.arguments = std::move(field.value());
    }
    parsed.function_call_value = std::move(parsed_function);
  }

  return result<message>::success(std::move(parsed));
}

[[nodiscard]] auto parse_choice_value(const core::json_value &value)
    -> result<choice> {
  if (!value.IsObject()) {
    return result<choice>::failure(
        make_error(errc::type_mismatch, "choice must be an object"));
  }
  choice parsed{};
  if (auto field = core::optional_int64(value, "index"); field.has_error()) {
    return result<choice>::failure(field.error());
  } else {
    parsed.index = field.value().value_or(0);
  }
  if (auto field = core::optional_string(value, "finish_reason");
      field.has_error()) {
    return result<choice>::failure(field.error());
  } else {
    parsed.finish_reason = std::move(field.value());
  }
  if (auto field = parse_optional_core_value_field(value, "logprobs");
      field.has_error()) {
    return result<choice>::failure(field.error());
  } else {
    parsed.logprobs_value = std::move(field.value());
  }

  const auto *message = json_member(value, "message");
  if (message != nullptr && !message->IsNull()) {
    auto parsed_message = parse_message_value(*message);
    if (parsed_message.has_error()) {
      return result<choice>::failure(parsed_message.error());
    }
    parsed.message_value = std::move(parsed_message.value());
  }

  const auto *delta = json_member(value, "delta");
  if (delta != nullptr && !delta->IsNull()) {
    auto parsed_delta = parse_message_value(*delta);
    if (parsed_delta.has_error()) {
      return result<choice>::failure(parsed_delta.error());
    }
    parsed.delta = std::move(parsed_delta.value());
  }

  return result<choice>::success(std::move(parsed));
}

[[nodiscard]] auto parse_usage_value(const core::json_value &value)
    -> result<usage> {
  if (!value.IsObject()) {
    return result<usage>::failure(
        make_error(errc::type_mismatch, "usage must be an object"));
  }

  usage parsed{};
  if (auto field = core::optional_int64(value, "prompt_tokens"); field.has_error()) {
    return result<usage>::failure(field.error());
  } else {
    parsed.prompt_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "completion_tokens");
      field.has_error()) {
    return result<usage>::failure(field.error());
  } else {
    parsed.completion_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "total_tokens"); field.has_error()) {
    return result<usage>::failure(field.error());
  } else {
    parsed.total_tokens = field.value();
  }

  return result<usage>::success(std::move(parsed));
}

[[nodiscard]] auto parse_chat_tool_definition_value(
    const core::json_value &value, const responses::parse_options &options)
    -> result<responses::tool_definition> {
  if (!value.IsObject()) {
    return result<responses::tool_definition>::failure(
        make_error(errc::type_mismatch, "tool definition must be an object"));
  }

  auto type = core::required_string(value, "type");
  if (type.has_error()) {
    return result<responses::tool_definition>::failure(type.error());
  }

  if (type.value() != "function") {
    auto serialized = json_to_string(value);
    if (serialized.has_error()) {
      return result<responses::tool_definition>::failure(serialized.error());
    }
    return responses::parse_tool_definition_json(serialized.value(), options);
  }

  const auto *function = json_member(value, "function");
  const auto &source =
      function != nullptr && function->IsObject() ? *function : value;

  responses::function_tool parsed{};
  if (auto name = core::required_string(source, "name"); name.has_error()) {
    return result<responses::tool_definition>::failure(name.error());
  } else {
    parsed.name = std::move(name.value());
  }
  if (auto description = core::optional_string(source, "description");
      description.has_error()) {
    return result<responses::tool_definition>::failure(description.error());
  } else {
    parsed.description = std::move(description.value());
  }
  if (auto strict = core::optional_bool(source, "strict"); strict.has_error()) {
    return result<responses::tool_definition>::failure(strict.error());
  } else {
    parsed.strict = strict.value();
  }

  const auto *parameters = json_member(source, "parameters");
  if (parameters != nullptr && !parameters->IsNull()) {
    auto serialized = json_to_string(*parameters);
    if (serialized.has_error()) {
      return result<responses::tool_definition>::failure(serialized.error());
    }
    parsed.parameters_json = std::move(serialized.value());
  }

  return result<responses::tool_definition>::success(
      responses::tool_definition{std::move(parsed)});
}

[[nodiscard]] auto parse_request_value(const core::json_value &value,
                                       const responses::parse_options &options)
    -> result<request> {
  if (!value.IsObject()) {
    return result<request>::failure(
        make_error(errc::type_mismatch, "chat request must be an object"));
  }

  request parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<request>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());

  if (auto field = core::optional_string(value, "model"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.model = std::move(field.value());
  }
  if (auto field = parse_optional_core_value_field(value, "audio");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.audio = std::move(field.value());
  }
  if (auto field = core::optional_double(value, "frequency_penalty");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.frequency_penalty = field.value();
  }
  if (auto field = parse_optional_core_value_field(value, "function_call");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.function_call = std::move(field.value());
  }
  const auto *functions = json_member(value, "functions");
  if (functions != nullptr && !functions->IsNull()) {
    if (!functions->IsArray()) {
      return result<request>::failure(
          make_error(errc::type_mismatch, "functions must be an array"));
    }
    parsed.functions.reserve(functions->Size());
    for (const auto &entry : functions->GetArray()) {
      auto converted = core::json_to_value(entry);
      if (converted.has_error()) {
        return result<request>::failure(converted.error());
      }
      parsed.functions.push_back(std::move(converted.value()));
    }
  }
  if (auto field = parse_optional_core_value_field(value, "logit_bias");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.logit_bias = std::move(field.value());
  }
  if (auto field = core::optional_bool(value, "logprobs"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.logprobs = field.value();
  }
  if (auto field = core::optional_int64(value, "max_completion_tokens");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.max_completion_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "max_tokens"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.max_tokens = field.value();
  }
  if (auto field = core::optional_string_array(value, "modalities");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.modalities = std::move(field.value());
  }
  if (auto field = core::optional_int64(value, "n"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.n = field.value();
  }
  if (auto field = core::optional_bool(value, "parallel_tool_calls");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.parallel_tool_calls = field.value();
  }
  if (auto field = parse_optional_core_value_field(value, "prediction");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.prediction = std::move(field.value());
  }
  if (auto field = core::optional_double(value, "presence_penalty");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.presence_penalty = field.value();
  }
  if (auto field = core::optional_string(value, "prompt_cache_key");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.prompt_cache_key = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "prompt_cache_retention");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.prompt_cache_retention = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "reasoning_effort");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.reasoning_effort = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "safety_identifier");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.safety_identifier = std::move(field.value());
  }
  if (auto field = core::optional_int64(value, "seed"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.seed = field.value();
  }
  if (auto field = core::optional_string(value, "service_tier");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.service_tier = std::move(field.value());
  }
  const auto *stop = json_member(value, "stop");
  if (stop != nullptr) {
    auto parsed_stop = parse_stop_sequence_value(*stop);
    if (parsed_stop.has_error()) {
      return result<request>::failure(parsed_stop.error());
    }
    parsed.stop = std::move(parsed_stop.value());
  }
  if (auto field = core::optional_bool(value, "store"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.store = field.value();
  }
  if (auto field = core::optional_double(value, "temperature"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.temperature = field.value();
  }
  if (auto field = core::optional_int64(value, "top_logprobs");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.top_logprobs = field.value();
  }
  if (auto field = core::optional_double(value, "top_p"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.top_p = field.value();
  }
  if (auto field = core::optional_bool(value, "stream"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.stream = field.value();
  }
  const auto *stream_options = json_member(value, "stream_options");
  if (stream_options != nullptr && !stream_options->IsNull()) {
    auto parsed_options = parse_stream_options_value(*stream_options);
    if (parsed_options.has_error()) {
      return result<request>::failure(parsed_options.error());
    }
    parsed.stream_options_value = std::move(parsed_options.value());
  }
  if (auto field = core::optional_string_map(value, "metadata"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.metadata = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "user"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.user = std::move(field.value());
  }
  if (auto field = parse_optional_core_value_field(value, "web_search_options");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.web_search_options = std::move(field.value());
  }

  const auto *messages = json_member(value, "messages");
  if (messages != nullptr && !messages->IsNull()) {
    if (!messages->IsArray()) {
      return result<request>::failure(
          make_error(errc::type_mismatch, "messages must be an array"));
    }
    parsed.messages.reserve(messages->Size());
    for (const auto &entry : messages->GetArray()) {
      auto parsed_message = parse_message_value(entry);
      if (parsed_message.has_error()) {
        return result<request>::failure(parsed_message.error());
      }
      parsed.messages.push_back(std::move(parsed_message.value()));
    }
  }

  const auto *tools = json_member(value, "tools");
  if (tools != nullptr && !tools->IsNull()) {
    if (!tools->IsArray()) {
      return result<request>::failure(
          make_error(errc::type_mismatch, "tools must be an array"));
    }
    parsed.tools.reserve(tools->Size());
    for (const auto &entry : tools->GetArray()) {
      auto parsed_tool = parse_chat_tool_definition_value(entry, options);
      if (parsed_tool.has_error()) {
        return result<request>::failure(parsed_tool.error());
      }
      parsed.tools.push_back(std::move(parsed_tool.value()));
    }
  }

  const auto *tool_choice = json_member(value, "tool_choice");
  if (tool_choice != nullptr && !tool_choice->IsNull()) {
    auto serialized = json_to_string(*tool_choice);
    if (serialized.has_error()) {
      return result<request>::failure(serialized.error());
    }
    auto parsed_choice =
        responses::parse_tool_choice_json(serialized.value(), options);
    if (parsed_choice.has_error()) {
      return result<request>::failure(parsed_choice.error());
    }
    parsed.tool_choice_value = std::move(parsed_choice.value());
  }

  auto response_format = optional_response_format(value);
  if (response_format.has_error()) {
    return result<request>::failure(response_format.error());
  }
  parsed.response_format = std::move(response_format.value());

  return result<request>::success(std::move(parsed));
}

template <typename output_t>
[[nodiscard]] auto parse_completion_like(const core::json_value &value,
                                         const responses::parse_options &)
    -> result<output_t> {
  if (!value.IsObject()) {
    return result<output_t>::failure(
        make_error(errc::type_mismatch, "chat payload must be an object"));
  }

  output_t parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<output_t>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());

  if (auto field = core::optional_string(value, "id"); field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "object"); field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.object = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "model"); field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.model = std::move(field.value());
  }
  if (auto field = core::optional_uint64(value, "created"); field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.created = field.value();
  }
  if (auto field = core::optional_string(value, "system_fingerprint");
      field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.system_fingerprint = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "service_tier"); field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.service_tier = std::move(field.value());
  }

  const auto *choices = json_member(value, "choices");
  if (choices != nullptr && !choices->IsNull()) {
    if (!choices->IsArray()) {
      return result<output_t>::failure(
          make_error(errc::type_mismatch, "choices must be an array"));
    }
    parsed.choices.reserve(choices->Size());
    for (const auto &entry : choices->GetArray()) {
      auto parsed_choice = parse_choice_value(entry);
      if (parsed_choice.has_error()) {
        return result<output_t>::failure(parsed_choice.error());
      }
      parsed.choices.push_back(std::move(parsed_choice.value()));
    }
  }

  const auto *usage = json_member(value, "usage");
  if (usage != nullptr && !usage->IsNull()) {
    auto parsed_usage = parse_usage_value(*usage);
    if (parsed_usage.has_error()) {
      return result<output_t>::failure(parsed_usage.error());
    }
    parsed.usage_value = std::move(parsed_usage.value());
  }

  return result<output_t>::success(std::move(parsed));
}

template <typename parsed_t>
[[nodiscard]] auto parse_single_object(std::string_view json_text,
                                       const responses::parse_options &options,
                                       auto &&parser) -> result<parsed_t> {
  auto document = parse_json(json_text);
  if (document.has_error()) {
    return result<parsed_t>::failure(document.error());
  }
  return parser(document.value(), options);
}

} // namespace

auto parse_request(std::string_view json_text,
                   const responses::parse_options &options)
    -> core::result<request> {
  return parse_single_object<request>(json_text, options, parse_request_value);
}

auto parse_completion(std::string_view json_text,
                      const responses::parse_options &options)
    -> core::result<completion> {
  return parse_single_object<completion>(json_text, options,
                                         parse_completion_like<completion>);
}

auto parse_completion_chunk(std::string_view json_text,
                            const responses::parse_options &options)
    -> core::result<completion_chunk> {
  return parse_single_object<completion_chunk>(
      json_text, options, parse_completion_like<completion_chunk>);
}

} // namespace openai::chat_completions
