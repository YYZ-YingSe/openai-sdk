#include "openai/responses/parser.hpp"

#include <type_traits>
#include <utility>

#include "openai/core/json.hpp"

namespace openai::responses {
namespace {

using core::errc;
using core::json_document;
using core::json_member;
using core::json_string;
using core::json_to_string;
using core::json_value;
using core::make_error;
using core::parse_json;
using core::result;

[[nodiscard]] auto optional_json_blob(const json_value &value,
                                      std::string_view key)
    -> result<std::optional<std::string>> {
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::optional<std::string>>::success(std::nullopt);
  }
  auto serialized = json_to_string(*member);
  if (serialized.has_error()) {
    return result<std::optional<std::string>>::failure(serialized.error());
  }
  return result<std::optional<std::string>>::success(std::move(serialized.value()));
}

[[nodiscard]] auto optional_raw_json(const json_value &value,
                                     std::string_view key)
    -> result<std::optional<raw_json>> {
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::optional<raw_json>>::success(std::nullopt);
  }
  auto parsed = make_raw_json(key, *member);
  if (parsed.has_error()) {
    return result<std::optional<raw_json>>::failure(parsed.error());
  }
  return result<std::optional<raw_json>>::success(std::move(parsed.value()));
}

[[nodiscard]] auto parse_stream_options_value(const json_value &value)
    -> result<stream_options> {
  if (!value.IsObject()) {
    return result<stream_options>::failure(
        make_error(errc::type_mismatch, "stream_options must be an object"));
  }
  stream_options parsed{};
  if (auto field = core::optional_bool(value, "include_obfuscation");
      field.has_error()) {
    return result<stream_options>::failure(field.error());
  } else {
    parsed.include_obfuscation = field.value();
  }
  return result<stream_options>::success(std::move(parsed));
}

[[nodiscard]] auto parse_context_management_value(const json_value &value)
    -> result<context_management_entry> {
  if (!value.IsObject()) {
    return result<context_management_entry>::failure(make_error(
        errc::type_mismatch, "context_management entry must be an object"));
  }
  context_management_entry parsed{};
  if (auto type = core::required_string(value, "type"); type.has_error()) {
    return result<context_management_entry>::failure(type.error());
  } else {
    parsed.type = std::move(type.value());
  }
  if (auto field = core::optional_int64(value, "compact_threshold");
      field.has_error()) {
    return result<context_management_entry>::failure(field.error());
  } else {
    parsed.compact_threshold = field.value();
  }
  return result<context_management_entry>::success(std::move(parsed));
}

[[nodiscard]] auto parse_unknown_variant(const std::string_view type,
                                         const json_value &value,
                                         const parse_options &options,
                                         const char *kind,
                                         auto lookup)
    -> result<raw_json> {
  if (options.registry != nullptr) {
    if (const auto *parser = (options.registry->*lookup)(type);
        parser != nullptr) {
      return (*parser)(type, value, options);
    }
  }
  if (options.strict) {
    return result<raw_json>::failure(make_error(
        errc::not_supported,
        std::string{"unknown "} + kind + " type '" + std::string{type} + "'"));
  }
  return make_raw_json(type, value);
}

[[nodiscard]] auto parse_instruction_payload_value(const json_value &value,
                                                   const parse_options &options)
    -> result<instructions_payload>;

[[nodiscard]] auto parse_message_content_part_value(const json_value &value,
                                                    const parse_options &options)
    -> result<message_content_part>;

[[nodiscard]] auto parse_tool_output_payload_value(const json_value &value,
                                                   const parse_options &options)
    -> result<tool_output_payload>;

template <typename item_t>
[[nodiscard]] auto parse_item_common_fields(const json_value &value,
                                            item_t &parsed)
    -> result<void>;

[[nodiscard]] auto parse_annotation_value(const json_value &value)
    -> result<annotation> {
  if (!value.IsObject()) {
    return result<annotation>::failure(
        make_error(errc::type_mismatch, "annotation must be an object"));
  }

  annotation parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<annotation>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());

  if (auto type = core::optional_string(value, "type"); type.has_error()) {
    return result<annotation>::failure(type.error());
  } else if (type.value().has_value()) {
    parsed.type = std::move(*type.value());
  }
  if (auto field = core::optional_string(value, "title"); field.has_error()) {
    return result<annotation>::failure(field.error());
  } else {
    parsed.title = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "url"); field.has_error()) {
    return result<annotation>::failure(field.error());
  } else {
    parsed.url = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "text"); field.has_error()) {
    return result<annotation>::failure(field.error());
  } else {
    parsed.text = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "file_id"); field.has_error()) {
    return result<annotation>::failure(field.error());
  } else {
    parsed.file_id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "filename"); field.has_error()) {
    return result<annotation>::failure(field.error());
  } else {
    parsed.filename = std::move(field.value());
  }
  if (auto field = core::optional_int64(value, "start_index"); field.has_error()) {
    return result<annotation>::failure(field.error());
  } else {
    parsed.start_index = field.value();
  }
  if (auto field = core::optional_int64(value, "index"); field.has_error()) {
    return result<annotation>::failure(field.error());
  } else {
    parsed.index = field.value();
  }
  if (auto field = core::optional_int64(value, "end_index"); field.has_error()) {
    return result<annotation>::failure(field.error());
  } else {
    parsed.end_index = field.value();
  }

  return result<annotation>::success(std::move(parsed));
}

[[nodiscard]] auto parse_logprob_token_value(const json_value &value)
    -> result<logprob_token> {
  if (!value.IsObject()) {
    return result<logprob_token>::failure(
        make_error(errc::type_mismatch, "logprob token must be an object"));
  }

  logprob_token parsed{};
  if (auto token = core::required_string(value, "token"); token.has_error()) {
    return result<logprob_token>::failure(token.error());
  } else {
    parsed.token = std::move(token.value());
  }
  if (auto logprob = core::optional_double(value, "logprob"); logprob.has_error()) {
    return result<logprob_token>::failure(logprob.error());
  } else {
    parsed.logprob = logprob.value();
  }

  const auto *bytes = json_member(value, "bytes");
  if (bytes != nullptr && !bytes->IsNull()) {
    if (!bytes->IsArray()) {
      return result<logprob_token>::failure(
          make_error(errc::type_mismatch, "bytes must be an array"));
    }
    parsed.bytes.reserve(bytes->Size());
    for (const auto &entry : bytes->GetArray()) {
      if (!entry.IsInt64()) {
        return result<logprob_token>::failure(make_error(
            errc::type_mismatch, "bytes must only contain integers"));
      }
      parsed.bytes.push_back(entry.GetInt64());
    }
  }

  return result<logprob_token>::success(std::move(parsed));
}

[[nodiscard]] auto parse_message_content_part_value(const json_value &value,
                                                    const parse_options &options)
    -> result<message_content_part> {
  if (value.IsString()) {
    return result<message_content_part>::success(
        message_content_part{input_text_content{json_string(value)}});
  }
  if (!value.IsObject()) {
    return result<message_content_part>::failure(make_error(
        errc::type_mismatch, "message content part must be an object"));
  }

  auto type = core::optional_string(value, "type");
  if (type.has_error()) {
    return result<message_content_part>::failure(type.error());
  }

  const auto type_value = type.value().value_or("");
  if (type_value == "input_text") {
    auto text = core::required_string(value, "text");
    if (text.has_error()) {
      return result<message_content_part>::failure(text.error());
    }
    return result<message_content_part>::success(
        message_content_part{input_text_content{std::move(text.value())}});
  }
  if (type_value == "input_image") {
    input_image_content parsed{};
    if (auto field = core::optional_string(value, "image_url");
        field.has_error()) {
      return result<message_content_part>::failure(field.error());
    } else {
      parsed.image_url = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "file_id");
        field.has_error()) {
      return result<message_content_part>::failure(field.error());
    } else {
      parsed.file_id = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "detail"); field.has_error()) {
      return result<message_content_part>::failure(field.error());
    } else {
      parsed.detail = std::move(field.value());
    }
    return result<message_content_part>::success(
        message_content_part{std::move(parsed)});
  }
  if (type_value == "input_file") {
    input_file_content parsed{};
    if (auto field = core::optional_string(value, "file_id");
        field.has_error()) {
      return result<message_content_part>::failure(field.error());
    } else {
      parsed.file_id = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "file_url");
        field.has_error()) {
      return result<message_content_part>::failure(field.error());
    } else {
      parsed.file_url = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "filename");
        field.has_error()) {
      return result<message_content_part>::failure(field.error());
    } else {
      parsed.filename = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "file_data");
        field.has_error()) {
      return result<message_content_part>::failure(field.error());
    } else {
      parsed.file_data = std::move(field.value());
    }
    return result<message_content_part>::success(
        message_content_part{std::move(parsed)});
  }
  if (type_value == "output_text") {
    output_text_content parsed{};
    if (auto text = core::required_string(value, "text"); text.has_error()) {
      return result<message_content_part>::failure(text.error());
    } else {
      parsed.text = std::move(text.value());
    }

    const auto *annotations = json_member(value, "annotations");
    if (annotations != nullptr && !annotations->IsNull()) {
      if (!annotations->IsArray()) {
        return result<message_content_part>::failure(make_error(
            errc::type_mismatch, "annotations must be an array"));
      }
      parsed.annotations.reserve(annotations->Size());
      for (const auto &entry : annotations->GetArray()) {
        auto parsed_annotation = parse_annotation_value(entry);
        if (parsed_annotation.has_error()) {
          return result<message_content_part>::failure(
              parsed_annotation.error());
        }
        parsed.annotations.push_back(std::move(parsed_annotation.value()));
      }
    }

    const auto *logprobs = json_member(value, "logprobs");
    if (logprobs != nullptr && !logprobs->IsNull()) {
      if (!logprobs->IsArray()) {
        return result<message_content_part>::failure(
            make_error(errc::type_mismatch, "logprobs must be an array"));
      }
      parsed.logprobs.reserve(logprobs->Size());
      for (const auto &entry : logprobs->GetArray()) {
        auto parsed_logprob = parse_logprob_token_value(entry);
        if (parsed_logprob.has_error()) {
          return result<message_content_part>::failure(parsed_logprob.error());
        }
        parsed.logprobs.push_back(std::move(parsed_logprob.value()));
      }
    }

    return result<message_content_part>::success(
        message_content_part{std::move(parsed)});
  }
  if (type_value == "refusal") {
    auto refusal = core::required_string(value, "refusal");
    if (refusal.has_error()) {
      return result<message_content_part>::failure(refusal.error());
    }
    return result<message_content_part>::success(
        message_content_part{refusal_content{std::move(refusal.value())}});
  }

  auto raw =
      parse_unknown_variant(type_value, value, options, "content_part",
                            &registry::find_content_part);
  if (raw.has_error()) {
    return result<message_content_part>::failure(raw.error());
  }
  return result<message_content_part>::success(message_content_part{
      std::move(raw.value())});
}

[[nodiscard]] auto parse_tool_output_payload_value(const json_value &value,
                                                   const parse_options &options)
    -> result<tool_output_payload> {
  if (value.IsString()) {
    return result<tool_output_payload>::success(tool_output_payload{
        json_string(value)});
  }
  if (!value.IsArray()) {
    return result<tool_output_payload>::failure(make_error(
        errc::type_mismatch, "tool output must be a string or array"));
  }

  std::vector<message_content_part> parts;
  parts.reserve(value.Size());
  for (const auto &entry : value.GetArray()) {
    auto parsed_part = parse_message_content_part_value(entry, options);
    if (parsed_part.has_error()) {
      return result<tool_output_payload>::failure(parsed_part.error());
    }
    parts.push_back(std::move(parsed_part.value()));
  }
  return result<tool_output_payload>::success(tool_output_payload{
      std::move(parts)});
}

[[nodiscard]] auto parse_message_item_value(const json_value &value,
                                            const parse_options &options)
    -> result<message_item> {
  if (!value.IsObject()) {
    return result<message_item>::failure(
        make_error(errc::type_mismatch, "message item must be an object"));
  }

  message_item parsed{};
  if (auto id = core::optional_string(value, "id"); id.has_error()) {
    return result<message_item>::failure(id.error());
  } else {
    parsed.id = std::move(id.value());
  }
  if (auto status = core::optional_string(value, "status"); status.has_error()) {
    return result<message_item>::failure(status.error());
  } else {
    parsed.status = std::move(status.value());
  }
  if (auto phase = core::optional_string(value, "phase"); phase.has_error()) {
    return result<message_item>::failure(phase.error());
  } else {
    parsed.phase = std::move(phase.value());
  }
  if (auto role = core::required_string(value, "role"); role.has_error()) {
    return result<message_item>::failure(role.error());
  } else {
    parsed.role = std::move(role.value());
  }

  const auto *content = json_member(value, "content");
  if (content == nullptr || content->IsNull()) {
    return result<message_item>::success(std::move(parsed));
  }

  if (content->IsString() || content->IsObject()) {
    auto parsed_part = parse_message_content_part_value(*content, options);
    if (parsed_part.has_error()) {
      return result<message_item>::failure(parsed_part.error());
    }
    parsed.content.push_back(std::move(parsed_part.value()));
    return result<message_item>::success(std::move(parsed));
  }

  if (!content->IsArray()) {
    return result<message_item>::failure(
        make_error(errc::type_mismatch, "message content must be a string, object, or array"));
  }

  parsed.content.reserve(content->Size());
  for (const auto &entry : content->GetArray()) {
    auto parsed_part = parse_message_content_part_value(entry, options);
    if (parsed_part.has_error()) {
      return result<message_item>::failure(parsed_part.error());
    }
    parsed.content.push_back(std::move(parsed_part.value()));
  }

  return result<message_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_reasoning_item_value(const json_value &value)
    -> result<reasoning_item> {
  reasoning_item parsed{};
  if (auto id = core::optional_string(value, "id"); id.has_error()) {
    return result<reasoning_item>::failure(id.error());
  } else {
    parsed.id = std::move(id.value());
  }
  if (auto status = core::optional_string(value, "status"); status.has_error()) {
    return result<reasoning_item>::failure(status.error());
  } else {
    parsed.status = std::move(status.value());
  }
  if (auto field = core::optional_string(value, "encrypted_content");
      field.has_error()) {
    return result<reasoning_item>::failure(field.error());
  } else {
    parsed.encrypted_content = std::move(field.value());
  }

  const auto *summary = json_member(value, "summary");
  if (summary != nullptr && !summary->IsNull()) {
    if (!summary->IsArray()) {
      return result<reasoning_item>::failure(
          make_error(errc::type_mismatch, "summary must be an array"));
    }
    parsed.summary.reserve(summary->Size());
    for (const auto &entry : summary->GetArray()) {
      if (entry.IsString()) {
        parsed.summary.push_back(reasoning_summary_part{json_string(entry)});
        continue;
      }
      if (!entry.IsObject()) {
        return result<reasoning_item>::failure(make_error(
            errc::type_mismatch, "summary entries must be objects or strings"));
      }
      auto text = core::optional_string(entry, "text");
      if (text.has_error()) {
        return result<reasoning_item>::failure(text.error());
      }
      parsed.summary.push_back(
          reasoning_summary_part{text.value().value_or(std::string{})});
    }
  }

  return result<reasoning_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_compaction_item_value(const json_value &value)
    -> result<compaction_item> {
  compaction_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<compaction_item>::failure(common.error());
  }
  if (auto field = optional_json_blob(value, "summary"); field.has_error()) {
    return result<compaction_item>::failure(field.error());
  } else {
    parsed.summary_json = std::move(field.value());
  }
  return result<compaction_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_item_reference_value(const json_value &value)
    -> result<item_reference> {
  item_reference parsed{};
  if (auto field = core::optional_string(value, "id"); field.has_error()) {
    return result<item_reference>::failure(field.error());
  } else {
    parsed.id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "item_id"); field.has_error()) {
    return result<item_reference>::failure(field.error());
  } else {
    parsed.item_id = std::move(field.value());
  }
  return result<item_reference>::success(std::move(parsed));
}

template <typename item_t>
[[nodiscard]] auto parse_item_common_fields(const json_value &value,
                                            item_t &parsed)
    -> result<void> {
  if (auto id = core::optional_string(value, "id"); id.has_error()) {
    return result<void>::failure(id.error());
  } else {
    parsed.id = std::move(id.value());
  }
  if (auto status = core::optional_string(value, "status"); status.has_error()) {
    return result<void>::failure(status.error());
  } else {
    parsed.status = std::move(status.value());
  }
  return result<void>::success();
}

[[nodiscard]] auto parse_function_call_item_value(const json_value &value)
    -> result<function_call_item> {
  function_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<function_call_item>::failure(common.error());
  }
  if (auto call_id = core::optional_string(value, "call_id"); call_id.has_error()) {
    return result<function_call_item>::failure(call_id.error());
  } else {
    parsed.call_id = std::move(call_id.value());
  }
  if (auto name = core::required_string(value, "name"); name.has_error()) {
    return result<function_call_item>::failure(name.error());
  } else {
    parsed.name = std::move(name.value());
  }
  if (auto arguments = core::required_string(value, "arguments");
      arguments.has_error()) {
    return result<function_call_item>::failure(arguments.error());
  } else {
    parsed.arguments = std::move(arguments.value());
  }
  return result<function_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_function_call_output_item_value(
    const json_value &value, const parse_options &options)
    -> result<function_call_output_item> {
  function_call_output_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<function_call_output_item>::failure(common.error());
  }
  if (auto call_id = core::required_string(value, "call_id"); call_id.has_error()) {
    return result<function_call_output_item>::failure(call_id.error());
  } else {
    parsed.call_id = std::move(call_id.value());
  }
  if (const auto *output_value = json_member(value, "output");
      output_value != nullptr && !output_value->IsNull() &&
      output_value->IsString()) {
    parsed.output = json_string(*output_value);
  }
  if (auto output_json = optional_json_blob(value, "output");
      output_json.has_error()) {
    return result<function_call_output_item>::failure(output_json.error());
  } else {
    parsed.output_json = std::move(output_json.value());
  }
  if (const auto *output_value = json_member(value, "output");
      output_value != nullptr && !output_value->IsNull()) {
    auto payload = parse_tool_output_payload_value(*output_value, options);
    if (payload.has_error()) {
      return result<function_call_output_item>::failure(payload.error());
    }
    parsed.output_payload = std::move(payload.value());
  }
  return result<function_call_output_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_function_shell_call_item_value(const json_value &value)
    -> result<function_shell_call_item> {
  function_shell_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<function_shell_call_item>::failure(common.error());
  }
  if (auto field = core::optional_string(value, "call_id"); field.has_error()) {
    return result<function_shell_call_item>::failure(field.error());
  } else {
    parsed.call_id = std::move(field.value());
  }
  if (auto field = optional_json_blob(value, "action"); field.has_error()) {
    return result<function_shell_call_item>::failure(field.error());
  } else {
    parsed.action_json = std::move(field.value());
  }
  if (auto field = optional_json_blob(value, "result"); field.has_error()) {
    return result<function_shell_call_item>::failure(field.error());
  } else {
    parsed.result_json = std::move(field.value());
  }
  return result<function_shell_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_file_search_call_item_value(const json_value &value)
    -> result<file_search_call_item> {
  file_search_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<file_search_call_item>::failure(common.error());
  }
  if (auto queries = core::optional_string_array(value, "queries");
      queries.has_error()) {
    return result<file_search_call_item>::failure(queries.error());
  } else {
    parsed.queries = std::move(queries.value());
  }
  if (auto results = optional_json_blob(value, "results"); results.has_error()) {
    return result<file_search_call_item>::failure(results.error());
  } else {
    parsed.results_json = std::move(results.value());
  }
  return result<file_search_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_tool_search_call_item_value(const json_value &value)
    -> result<tool_search_call_item> {
  tool_search_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<tool_search_call_item>::failure(common.error());
  }
  if (auto field = core::optional_string(value, "call_id"); field.has_error()) {
    return result<tool_search_call_item>::failure(field.error());
  } else {
    parsed.call_id = std::move(field.value());
  }
  if (auto queries = core::optional_string_array(value, "queries");
      queries.has_error()) {
    return result<tool_search_call_item>::failure(queries.error());
  } else {
    parsed.queries = std::move(queries.value());
  }
  if (auto results = optional_json_blob(value, "results"); results.has_error()) {
    return result<tool_search_call_item>::failure(results.error());
  } else {
    parsed.results_json = std::move(results.value());
  }
  return result<tool_search_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_web_search_call_item_value(const json_value &value)
    -> result<web_search_call_item> {
  web_search_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<web_search_call_item>::failure(common.error());
  }
  if (auto action = optional_json_blob(value, "action"); action.has_error()) {
    return result<web_search_call_item>::failure(action.error());
  } else {
    parsed.action_json = std::move(action.value());
  }
  return result<web_search_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_computer_call_item_value(const json_value &value)
    -> result<computer_call_item> {
  computer_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<computer_call_item>::failure(common.error());
  }
  if (auto call_id = core::optional_string(value, "call_id"); call_id.has_error()) {
    return result<computer_call_item>::failure(call_id.error());
  } else {
    parsed.call_id = std::move(call_id.value());
  }
  if (auto action = optional_json_blob(value, "action"); action.has_error()) {
    return result<computer_call_item>::failure(action.error());
  } else {
    parsed.action_json = std::move(action.value());
  }
  if (auto checks = optional_json_blob(value, "pending_safety_checks");
      checks.has_error()) {
    return result<computer_call_item>::failure(checks.error());
  } else {
    parsed.pending_safety_checks_json = std::move(checks.value());
  }
  return result<computer_call_item>::success(std::move(parsed));
}

template <typename output_item_t>
[[nodiscard]] auto parse_call_output_item_value(const json_value &value,
                                                output_item_t &parsed,
                                                const bool require_call_id = true)
    -> result<void> {
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return common;
  }
  const auto *call_id_member = json_member(value, "call_id");
  if (call_id_member != nullptr && !call_id_member->IsNull()) {
    if (!call_id_member->IsString()) {
      return result<void>::failure(
          make_error(errc::type_mismatch, "call_id must be a string"));
    }
    parsed.call_id = json_string(*call_id_member);
  } else if (require_call_id) {
    return result<void>::failure(
        make_error(errc::not_found, "missing required field 'call_id'"));
  }
  if constexpr (requires { parsed.output = std::optional<std::string>{}; }) {
    if (const auto *output_value = json_member(value, "output");
        output_value != nullptr && !output_value->IsNull() &&
        output_value->IsString()) {
      parsed.output = json_string(*output_value);
    }
  }
  if (auto output = optional_json_blob(value, "output"); output.has_error()) {
    return result<void>::failure(output.error());
  } else {
    parsed.output_json = std::move(output.value());
  }
  return result<void>::success();
}

[[nodiscard]] auto parse_local_shell_call_item_value(const json_value &value)
    -> result<local_shell_call_item> {
  local_shell_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<local_shell_call_item>::failure(common.error());
  }
  if (auto call_id = core::optional_string(value, "call_id"); call_id.has_error()) {
    return result<local_shell_call_item>::failure(call_id.error());
  } else {
    parsed.call_id = std::move(call_id.value());
  }
  if (auto action = optional_json_blob(value, "action"); action.has_error()) {
    return result<local_shell_call_item>::failure(action.error());
  } else {
    parsed.action_json = std::move(action.value());
  }
  if (auto output = optional_json_blob(value, "output"); output.has_error()) {
    return result<local_shell_call_item>::failure(output.error());
  } else {
    parsed.output_json = std::move(output.value());
  }
  return result<local_shell_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_shell_call_item_value(const json_value &value)
    -> result<shell_call_item> {
  shell_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<shell_call_item>::failure(common.error());
  }
  if (auto call_id = core::optional_string(value, "call_id"); call_id.has_error()) {
    return result<shell_call_item>::failure(call_id.error());
  } else {
    parsed.call_id = std::move(call_id.value());
  }
  if (auto action = optional_json_blob(value, "action"); action.has_error()) {
    return result<shell_call_item>::failure(action.error());
  } else {
    parsed.action_json = std::move(action.value());
  }
  if (auto output = optional_json_blob(value, "output"); output.has_error()) {
    return result<shell_call_item>::failure(output.error());
  } else {
    parsed.output_json = std::move(output.value());
  }
  return result<shell_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_custom_tool_call_item_value(const json_value &value)
    -> result<custom_tool_call_item> {
  custom_tool_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<custom_tool_call_item>::failure(common.error());
  }
  if (auto call_id = core::optional_string(value, "call_id"); call_id.has_error()) {
    return result<custom_tool_call_item>::failure(call_id.error());
  } else {
    parsed.call_id = std::move(call_id.value());
  }
  if (auto name = core::required_string(value, "name"); name.has_error()) {
    return result<custom_tool_call_item>::failure(name.error());
  } else {
    parsed.name = std::move(name.value());
  }
  if (auto input = core::optional_string(value, "input"); input.has_error()) {
    return result<custom_tool_call_item>::failure(input.error());
  } else {
    parsed.input = std::move(input.value());
  }
  return result<custom_tool_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_custom_tool_call_output_item_value(
    const json_value &value, const parse_options &options)
    -> result<custom_tool_call_output_item> {
  custom_tool_call_output_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<custom_tool_call_output_item>::failure(common.error());
  }
  if (auto call_id = core::required_string(value, "call_id"); call_id.has_error()) {
    return result<custom_tool_call_output_item>::failure(call_id.error());
  } else {
    parsed.call_id = std::move(call_id.value());
  }
  if (const auto *output_value = json_member(value, "output");
      output_value != nullptr && !output_value->IsNull() &&
      output_value->IsString()) {
    parsed.output = json_string(*output_value);
  }
  if (auto output_json = optional_json_blob(value, "output");
      output_json.has_error()) {
    return result<custom_tool_call_output_item>::failure(output_json.error());
  } else {
    parsed.output_json = std::move(output_json.value());
  }
  if (const auto *output_value = json_member(value, "output");
      output_value != nullptr && !output_value->IsNull()) {
    auto payload = parse_tool_output_payload_value(*output_value, options);
    if (payload.has_error()) {
      return result<custom_tool_call_output_item>::failure(payload.error());
    }
    parsed.output_payload = std::move(payload.value());
  }
  return result<custom_tool_call_output_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_image_generation_call_item_value(const json_value &value)
    -> result<image_generation_call_item> {
  image_generation_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<image_generation_call_item>::failure(common.error());
  }
  if (auto output = optional_json_blob(value, "result"); output.has_error()) {
    return result<image_generation_call_item>::failure(output.error());
  } else {
    parsed.result_json = std::move(output.value());
  }
  return result<image_generation_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_code_interpreter_call_item_value(const json_value &value)
    -> result<code_interpreter_call_item> {
  code_interpreter_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<code_interpreter_call_item>::failure(common.error());
  }
  if (auto call_id = core::optional_string(value, "call_id"); call_id.has_error()) {
    return result<code_interpreter_call_item>::failure(call_id.error());
  } else {
    parsed.call_id = std::move(call_id.value());
  }
  if (auto code = core::optional_string(value, "code"); code.has_error()) {
    return result<code_interpreter_call_item>::failure(code.error());
  } else {
    parsed.code = std::move(code.value());
  }
  if (auto outputs = optional_json_blob(value, "outputs"); outputs.has_error()) {
    return result<code_interpreter_call_item>::failure(outputs.error());
  } else {
    parsed.outputs_json = std::move(outputs.value());
  }
  return result<code_interpreter_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_mcp_call_item_value(const json_value &value)
    -> result<mcp_call_item> {
  mcp_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<mcp_call_item>::failure(common.error());
  }
  if (auto name = core::optional_string(value, "name"); name.has_error()) {
    return result<mcp_call_item>::failure(name.error());
  } else {
    parsed.name = std::move(name.value());
  }
  if (auto arguments = core::optional_string(value, "arguments");
      arguments.has_error()) {
    return result<mcp_call_item>::failure(arguments.error());
  } else {
    parsed.arguments = std::move(arguments.value());
  }
  if (auto label = core::optional_string(value, "server_label");
      label.has_error()) {
    return result<mcp_call_item>::failure(label.error());
  } else {
    parsed.server_label = std::move(label.value());
  }
  if (auto field = core::optional_string(value, "approval_request_id");
      field.has_error()) {
    return result<mcp_call_item>::failure(field.error());
  } else {
    parsed.approval_request_id = std::move(field.value());
  }
  if (auto output = optional_json_blob(value, "output"); output.has_error()) {
    return result<mcp_call_item>::failure(output.error());
  } else {
    parsed.output_json = std::move(output.value());
  }
  if (auto error = optional_json_blob(value, "error"); error.has_error()) {
    return result<mcp_call_item>::failure(error.error());
  } else {
    parsed.error_json = std::move(error.value());
  }
  return result<mcp_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_mcp_list_tools_item_value(const json_value &value)
    -> result<mcp_list_tools_item> {
  mcp_list_tools_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<mcp_list_tools_item>::failure(common.error());
  }
  if (auto field = core::optional_string(value, "server_label");
      field.has_error()) {
    return result<mcp_list_tools_item>::failure(field.error());
  } else {
    parsed.server_label = std::move(field.value());
  }
  if (auto field = optional_json_blob(value, "tools"); field.has_error()) {
    return result<mcp_list_tools_item>::failure(field.error());
  } else {
    parsed.tools_json = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "error"); field.has_error()) {
    return result<mcp_list_tools_item>::failure(field.error());
  } else {
    parsed.error = std::move(field.value());
  }
  return result<mcp_list_tools_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_mcp_approval_request_item_value(const json_value &value)
    -> result<mcp_approval_request_item> {
  mcp_approval_request_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<mcp_approval_request_item>::failure(common.error());
  }
  if (auto field = core::optional_string(value, "name"); field.has_error()) {
    return result<mcp_approval_request_item>::failure(field.error());
  } else {
    parsed.name = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "arguments"); field.has_error()) {
    return result<mcp_approval_request_item>::failure(field.error());
  } else {
    parsed.arguments = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "server_label");
      field.has_error()) {
    return result<mcp_approval_request_item>::failure(field.error());
  } else {
    parsed.server_label = std::move(field.value());
  }
  return result<mcp_approval_request_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_mcp_approval_response_item_value(const json_value &value)
    -> result<mcp_approval_response_item> {
  mcp_approval_response_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<mcp_approval_response_item>::failure(common.error());
  }
  if (auto field = core::optional_string(value, "approval_request_id");
      field.has_error()) {
    return result<mcp_approval_response_item>::failure(field.error());
  } else {
    parsed.approval_request_id = std::move(field.value());
  }
  if (auto field = core::optional_bool(value, "approve"); field.has_error()) {
    return result<mcp_approval_response_item>::failure(field.error());
  } else {
    parsed.approve = field.value();
  }
  if (auto field = core::optional_string(value, "reason"); field.has_error()) {
    return result<mcp_approval_response_item>::failure(field.error());
  } else {
    parsed.reason = std::move(field.value());
  }
  return result<mcp_approval_response_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_apply_patch_call_item_value(const json_value &value)
    -> result<apply_patch_call_item> {
  apply_patch_call_item parsed{};
  auto common = parse_item_common_fields(value, parsed);
  if (common.has_error()) {
    return result<apply_patch_call_item>::failure(common.error());
  }
  if (auto call_id = core::optional_string(value, "call_id"); call_id.has_error()) {
    return result<apply_patch_call_item>::failure(call_id.error());
  } else {
    parsed.call_id = std::move(call_id.value());
  }
  if (auto input = core::optional_string(value, "input"); input.has_error()) {
    return result<apply_patch_call_item>::failure(input.error());
  } else {
    parsed.input = std::move(input.value());
  }
  if (auto output = optional_json_blob(value, "output"); output.has_error()) {
    return result<apply_patch_call_item>::failure(output.error());
  } else {
    parsed.output_json = std::move(output.value());
  }
  return result<apply_patch_call_item>::success(std::move(parsed));
}

[[nodiscard]] auto parse_item_value(const json_value &value,
                                    const parse_options &options)
    -> result<item> {
  if (!value.IsObject()) {
    return result<item>::failure(
        make_error(errc::type_mismatch, "response item must be an object"));
  }

  auto type = core::required_string(value, "type");
  if (type.has_error()) {
    return result<item>::failure(type.error());
  }

  const auto &kind = type.value();
  if (kind == "message") {
    auto parsed = parse_message_item_value(value, options);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "reasoning") {
    auto parsed = parse_reasoning_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "compaction") {
    auto parsed = parse_compaction_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "item_reference") {
    auto parsed = parse_item_reference_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "function_call") {
    auto parsed = parse_function_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "function_call_output") {
    auto parsed = parse_function_call_output_item_value(value, options);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "function_shell_call") {
    auto parsed = parse_function_shell_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "function_shell_call_output") {
    function_shell_call_output_item parsed{};
    auto common = parse_call_output_item_value(value, parsed);
    if (common.has_error()) {
      return result<item>::failure(common.error());
    }
    return result<item>::success(item{std::move(parsed)});
  }
  if (kind == "file_search_call") {
    auto parsed = parse_file_search_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "tool_search_call") {
    auto parsed = parse_tool_search_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "tool_search_output") {
    tool_search_output_item parsed{};
    auto common = parse_call_output_item_value(value, parsed);
    if (common.has_error()) {
      return result<item>::failure(common.error());
    }
    return result<item>::success(item{std::move(parsed)});
  }
  if (kind == "web_search_call") {
    auto parsed = parse_web_search_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "computer_call") {
    auto parsed = parse_computer_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "computer_call_output") {
    computer_call_output_item parsed{};
    auto common = parse_call_output_item_value(value, parsed);
    if (common.has_error()) {
      return result<item>::failure(common.error());
    }
    return result<item>::success(item{std::move(parsed)});
  }
  if (kind == "local_shell_call") {
    auto parsed = parse_local_shell_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "local_shell_call_output") {
    local_shell_call_output_item parsed{};
    auto common = parse_call_output_item_value(value, parsed, false);
    if (common.has_error()) {
      return result<item>::failure(common.error());
    }
    return result<item>::success(item{std::move(parsed)});
  }
  if (kind == "shell_call") {
    auto parsed = parse_shell_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "shell_call_output") {
    shell_call_output_item parsed{};
    auto common = parse_call_output_item_value(value, parsed, false);
    if (common.has_error()) {
      return result<item>::failure(common.error());
    }
    return result<item>::success(item{std::move(parsed)});
  }
  if (kind == "custom_tool_call") {
    auto parsed = parse_custom_tool_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "custom_tool_call_output") {
    auto parsed = parse_custom_tool_call_output_item_value(value, options);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "image_generation_call") {
    auto parsed = parse_image_generation_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "code_interpreter_call") {
    auto parsed = parse_code_interpreter_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "code_interpreter_call_output") {
    code_interpreter_call_output_item parsed{};
    auto common = parse_call_output_item_value(value, parsed);
    if (common.has_error()) {
      return result<item>::failure(common.error());
    }
    return result<item>::success(item{std::move(parsed)});
  }
  if (kind == "mcp_call") {
    auto parsed = parse_mcp_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "mcp_list_tools") {
    auto parsed = parse_mcp_list_tools_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "mcp_approval_request") {
    auto parsed = parse_mcp_approval_request_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "mcp_approval_response") {
    auto parsed = parse_mcp_approval_response_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "apply_patch_call") {
    auto parsed = parse_apply_patch_call_item_value(value);
    if (parsed.has_error()) {
      return result<item>::failure(parsed.error());
    }
    return result<item>::success(item{std::move(parsed.value())});
  }
  if (kind == "apply_patch_call_output") {
    apply_patch_call_output_item parsed{};
    auto common = parse_call_output_item_value(value, parsed);
    if (common.has_error()) {
      return result<item>::failure(common.error());
    }
    return result<item>::success(item{std::move(parsed)});
  }

  auto raw =
      parse_unknown_variant(kind, value, options, "item", &registry::find_item);
  if (raw.has_error()) {
    return result<item>::failure(raw.error());
  }
  return result<item>::success(item{std::move(raw.value())});
}

[[nodiscard]] auto parse_reasoning_config_value(const json_value &value)
    -> result<reasoning_config> {
  if (!value.IsObject()) {
    return result<reasoning_config>::failure(
        make_error(errc::type_mismatch, "reasoning must be an object"));
  }
  reasoning_config parsed{};
  if (auto effort = core::optional_string(value, "effort"); effort.has_error()) {
    return result<reasoning_config>::failure(effort.error());
  } else {
    parsed.effort = std::move(effort.value());
  }
  if (auto summary = core::optional_string(value, "generate_summary");
      summary.has_error()) {
    return result<reasoning_config>::failure(summary.error());
  } else {
    parsed.generate_summary = std::move(summary.value());
  }
  if (auto summary = core::optional_string(value, "summary"); summary.has_error()) {
    return result<reasoning_config>::failure(summary.error());
  } else {
    parsed.summary = std::move(summary.value());
  }
  return result<reasoning_config>::success(std::move(parsed));
}

[[nodiscard]] auto parse_text_config_value(const json_value &value)
    -> result<response_text_config> {
  if (!value.IsObject()) {
    return result<response_text_config>::failure(
        make_error(errc::type_mismatch, "text configuration must be an object"));
  }

  const auto *format = json_member(value, "format");
  const auto &selected =
      (format != nullptr && format->IsObject()) ? *format : value;

  response_text_config parsed{};
  if (auto type = core::optional_string(selected, "type"); type.has_error()) {
    return result<response_text_config>::failure(type.error());
  } else if (type.value().has_value()) {
    parsed.type = std::move(*type.value());
  }

  const auto *schema_source = json_member(selected, "json_schema");
  if (schema_source != nullptr && !schema_source->IsObject()) {
    return result<response_text_config>::failure(make_error(
        errc::type_mismatch, "json_schema must be an object"));
  }
  const auto &schema =
      (schema_source != nullptr && schema_source->IsObject()) ? *schema_source
                                                              : selected;

  if (auto name = core::optional_string(schema, "name"); name.has_error()) {
    return result<response_text_config>::failure(name.error());
  } else {
    parsed.schema_name = std::move(name.value());
  }
  if (auto description = core::optional_string(schema, "description");
      description.has_error()) {
    return result<response_text_config>::failure(description.error());
  } else {
    parsed.schema_description = std::move(description.value());
  }
  if (auto strict = core::optional_bool(schema, "strict"); strict.has_error()) {
    return result<response_text_config>::failure(strict.error());
  } else {
    parsed.strict = strict.value();
  }
  if (auto schema_blob = optional_json_blob(schema, "schema");
      schema_blob.has_error()) {
    return result<response_text_config>::failure(schema_blob.error());
  } else {
    parsed.schema_json = std::move(schema_blob.value());
  }
  if (auto verbosity = core::optional_string(selected, "verbosity");
      verbosity.has_error()) {
    return result<response_text_config>::failure(verbosity.error());
  } else {
    parsed.verbosity = std::move(verbosity.value());
  }

  return result<response_text_config>::success(std::move(parsed));
}

[[nodiscard]] auto parse_tool_choice_value(const json_value &value,
                                           const parse_options & /*options*/)
    -> result<tool_choice> {
  tool_choice parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<tool_choice>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());

  if (value.IsString()) {
    parsed.mode = json_string(value);
    return result<tool_choice>::success(std::move(parsed));
  }
  if (!value.IsObject()) {
    return result<tool_choice>::failure(make_error(
        errc::type_mismatch, "tool_choice must be a string or object"));
  }

  if (auto type = core::optional_string(value, "type"); type.has_error()) {
    return result<tool_choice>::failure(type.error());
  } else {
    parsed.type = std::move(type.value());
  }
  if (auto name = core::optional_string(value, "name"); name.has_error()) {
    return result<tool_choice>::failure(name.error());
  } else {
    parsed.name = std::move(name.value());
  }
  if (auto field = core::optional_string(value, "server_label");
      field.has_error()) {
    return result<tool_choice>::failure(field.error());
  } else {
    parsed.server_label = std::move(field.value());
  }
  if (auto field = optional_json_blob(value, "tools"); field.has_error()) {
    return result<tool_choice>::failure(field.error());
  } else {
    parsed.tools_json = std::move(field.value());
  }
  const auto *function = json_member(value, "function");
  if (!parsed.name.has_value() && function != nullptr && function->IsObject()) {
    auto name = core::optional_string(*function, "name");
    if (name.has_error()) {
      return result<tool_choice>::failure(name.error());
    }
    parsed.name = std::move(name.value());
  }
  parsed.mode = parsed.type.value_or("object");
  return result<tool_choice>::success(std::move(parsed));
}

[[nodiscard]] auto parse_tool_definition_value(const json_value &value,
                                               const parse_options &options)
    -> result<tool_definition> {
  if (!value.IsObject()) {
    return result<tool_definition>::failure(
        make_error(errc::type_mismatch, "tool definition must be an object"));
  }

  auto type = core::required_string(value, "type");
  if (type.has_error()) {
    return result<tool_definition>::failure(type.error());
  }

  const auto &kind = type.value();
  if (kind == "function") {
    function_tool parsed{};
    if (auto name = core::required_string(value, "name"); name.has_error()) {
      return result<tool_definition>::failure(name.error());
    } else {
      parsed.name = std::move(name.value());
    }
    if (auto description = core::optional_string(value, "description");
        description.has_error()) {
      return result<tool_definition>::failure(description.error());
    } else {
      parsed.description = std::move(description.value());
    }
    if (auto strict = core::optional_bool(value, "strict"); strict.has_error()) {
      return result<tool_definition>::failure(strict.error());
    } else {
      parsed.strict = strict.value();
    }
    if (auto params = optional_json_blob(value, "parameters"); params.has_error()) {
      return result<tool_definition>::failure(params.error());
    } else {
      parsed.parameters_json = std::move(params.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "file_search") {
    file_search_tool parsed{};
    if (auto ids = core::optional_string_array(value, "vector_store_ids");
        ids.has_error()) {
      return result<tool_definition>::failure(ids.error());
    } else {
      parsed.vector_store_ids = std::move(ids.value());
    }
    if (auto max = core::optional_int64(value, "max_num_results");
        max.has_error()) {
      return result<tool_definition>::failure(max.error());
    } else {
      parsed.max_num_results = max.value();
    }
    if (auto filters = optional_json_blob(value, "filters"); filters.has_error()) {
      return result<tool_definition>::failure(filters.error());
    } else {
      parsed.filters_json = std::move(filters.value());
    }
    if (auto ranking = optional_json_blob(value, "ranking_options");
        ranking.has_error()) {
      return result<tool_definition>::failure(ranking.error());
    } else {
      parsed.ranking_options_json = std::move(ranking.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "tool_search") {
    tool_search_tool parsed{};
    if (auto field = core::optional_string(value, "description");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.description = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "filters"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.filters_json = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "ranking_options");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.ranking_options_json = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "user_location");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.user_location_json = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "config"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.raw_config_json = std::move(field.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "web_search_preview" || kind == "web_search") {
    web_search_tool parsed{};
    if (auto field = core::optional_string(value, "search_context_size");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.search_context_size = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "user_location");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.user_location_json = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "filters"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.filters_json = std::move(field.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "computer_use_preview" || kind == "computer_use") {
    computer_tool parsed{};
    if (auto field = core::optional_string(value, "environment");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.environment = std::move(field.value());
    }
    if (auto field = core::optional_int64(value, "display_width");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.display_width = field.value();
    }
    if (auto field = core::optional_int64(value, "display_height");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.display_height = field.value();
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "namespace") {
    namespace_tool parsed{};
    if (auto field = core::optional_string(value, "name"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.name = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "description");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.description = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "config"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.config_json = std::move(field.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "local_shell") {
    local_shell_tool parsed{};
    if (auto field = core::optional_string(value, "description");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.description = std::move(field.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "shell") {
    shell_tool parsed{};
    if (auto field = core::optional_string(value, "description");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.description = std::move(field.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "function_shell") {
    function_shell_tool parsed{};
    if (auto field = core::optional_string(value, "description");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.description = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "parameters"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.parameters_json = std::move(field.value());
    }
    if (auto field = core::optional_bool(value, "strict"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.strict = field.value();
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "custom") {
    custom_tool parsed{};
    if (auto name = core::required_string(value, "name"); name.has_error()) {
      return result<tool_definition>::failure(name.error());
    } else {
      parsed.name = std::move(name.value());
    }
    if (auto field = core::optional_string(value, "description");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.description = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "format"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.format = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "grammar"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.grammar_json = std::move(field.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "image_generation") {
    image_generation_tool parsed{};
    if (auto field = core::optional_string(value, "model"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.model = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "background");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.background = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "output_format");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.output_format = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "quality");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.quality = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "size"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.size = std::move(field.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "code_interpreter") {
    code_interpreter_tool parsed{};
    if (auto field = optional_json_blob(value, "container"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.container_json = std::move(field.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "mcp") {
    mcp_tool parsed{};
    if (auto field = core::optional_string(value, "server_label");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.server_label = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "server_url");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.server_url = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "connector_id");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.connector_id = std::move(field.value());
    }
    if (auto field = core::optional_string_array(value, "allowed_tools");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.allowed_tools = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "headers"); field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.headers_json = std::move(field.value());
    }
    if (auto field = core::optional_string(value, "require_approval");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.require_approval = std::move(field.value());
    }
    if (auto field = optional_json_blob(value, "authorization");
        field.has_error()) {
      return result<tool_definition>::failure(field.error());
    } else {
      parsed.authorization_json = std::move(field.value());
    }
    return result<tool_definition>::success(tool_definition{std::move(parsed)});
  }
  if (kind == "apply_patch") {
    return result<tool_definition>::success(tool_definition{apply_patch_tool{}});
  }

  auto raw =
      parse_unknown_variant(kind, value, options, "tool", &registry::find_tool);
  if (raw.has_error()) {
    return result<tool_definition>::failure(raw.error());
  }
  return result<tool_definition>::success(tool_definition{std::move(raw.value())});
}

[[nodiscard]] auto parse_input_payload_value(const json_value &value,
                                             const parse_options &options)
    -> result<input_payload> {
  if (value.IsString()) {
    return result<input_payload>::success(input_payload{json_string(value)});
  }
  if (value.IsObject()) {
    auto parsed_item = parse_item_value(value, options);
    if (parsed_item.has_error()) {
      return result<input_payload>::failure(parsed_item.error());
    }
    std::vector<item> items;
    items.push_back(std::move(parsed_item.value()));
    return result<input_payload>::success(input_payload{std::move(items)});
  }
  if (!value.IsArray()) {
    return result<input_payload>::failure(make_error(
        errc::type_mismatch, "input must be a string, object, or array"));
  }

  std::vector<item> items;
  items.reserve(value.Size());
  for (const auto &entry : value.GetArray()) {
    auto parsed_item = parse_item_value(entry, options);
    if (parsed_item.has_error()) {
      return result<input_payload>::failure(parsed_item.error());
    }
    items.push_back(std::move(parsed_item.value()));
  }
  return result<input_payload>::success(input_payload{std::move(items)});
}

[[nodiscard]] auto parse_instruction_payload_value(const json_value &value,
                                                   const parse_options &options)
    -> result<instructions_payload> {
  if (value.IsString()) {
    return result<instructions_payload>::success(
        instructions_payload{json_string(value)});
  }

  auto parsed = parse_input_payload_value(value, options);
  if (parsed.has_error()) {
    return result<instructions_payload>::failure(parsed.error());
  }

  if (std::holds_alternative<std::string>(parsed.value())) {
    return result<instructions_payload>::success(instructions_payload{
        std::get<std::string>(std::move(parsed.value()))});
  }
  return result<instructions_payload>::success(instructions_payload{
      std::get<std::vector<item>>(std::move(parsed.value()))});
}

[[nodiscard]] auto parse_usage_value(const json_value &value) -> result<usage> {
  if (!value.IsObject()) {
    return result<usage>::failure(
        make_error(errc::type_mismatch, "usage must be an object"));
  }
  usage parsed{};
  if (auto field = core::optional_int64(value, "input_tokens"); field.has_error()) {
    return result<usage>::failure(field.error());
  } else {
    parsed.input_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "output_tokens"); field.has_error()) {
    return result<usage>::failure(field.error());
  } else {
    parsed.output_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "total_tokens"); field.has_error()) {
    return result<usage>::failure(field.error());
  } else {
    parsed.total_tokens = field.value();
  }

  const auto *input_details = json_member(value, "input_tokens_details");
  if (input_details != nullptr && input_details->IsObject()) {
    if (auto field = core::optional_int64(*input_details, "cached_tokens");
        field.has_error()) {
      return result<usage>::failure(field.error());
    } else {
      parsed.input_details.cached_tokens = field.value();
    }
  }
  const auto *output_details = json_member(value, "output_tokens_details");
  if (output_details != nullptr && output_details->IsObject()) {
    if (auto field = core::optional_int64(*output_details, "reasoning_tokens");
        field.has_error()) {
      return result<usage>::failure(field.error());
    } else {
      parsed.output_details.reasoning_tokens = field.value();
    }
  }

  return result<usage>::success(std::move(parsed));
}

[[nodiscard]] auto parse_response_error_value(const json_value &value,
                                              const parse_options & /*options*/)
    -> result<response_error> {
  if (!value.IsObject()) {
    return result<response_error>::failure(
        make_error(errc::type_mismatch, "error must be an object"));
  }
  response_error parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<response_error>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());

  if (auto field = core::optional_string(value, "type"); field.has_error()) {
    return result<response_error>::failure(field.error());
  } else {
    parsed.type = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "code"); field.has_error()) {
    return result<response_error>::failure(field.error());
  } else {
    parsed.code = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "message"); field.has_error()) {
    return result<response_error>::failure(field.error());
  } else {
    parsed.message = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "param"); field.has_error()) {
    return result<response_error>::failure(field.error());
  } else {
    parsed.param = std::move(field.value());
  }

  return result<response_error>::success(std::move(parsed));
}

[[nodiscard]] auto parse_response_value(const json_value &value,
                                        const parse_options &options)
    -> result<response> {
  if (!value.IsObject()) {
    return result<response>::failure(
        make_error(errc::type_mismatch, "response must be an object"));
  }

  response parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<response>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());

  if (auto field = core::optional_string(value, "id"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "object"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.object = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "model"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.model = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "status"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.status = std::move(field.value());
  }
  if (auto field = core::optional_uint64(value, "created_at"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.created_at = field.value();
  }
  if (auto field = core::optional_uint64(value, "completed_at");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.completed_at = field.value();
  }
  if (auto field = core::optional_bool(value, "background"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.background = field.value();
  }
  if (auto field = core::optional_bool(value, "stream"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.stream = field.value();
  }
  if (auto field = core::optional_bool(value, "parallel_tool_calls");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.parallel_tool_calls = field.value();
  }
  if (auto field = core::optional_string(value, "previous_response_id");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.previous_response_id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "user"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.user = std::move(field.value());
  }
  if (auto field = core::optional_int64(value, "max_output_tokens");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.max_output_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "max_tool_calls");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.max_tool_calls = field.value();
  }
  if (auto field = core::optional_double(value, "temperature"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.temperature = field.value();
  }
  if (auto field = core::optional_double(value, "top_p"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.top_p = field.value();
  }
  if (auto field = core::optional_double(value, "top_logprobs");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.top_logprobs = field.value();
  }
  if (auto field = core::optional_string(value, "truncation"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.truncation = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "prompt_cache_key");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.prompt_cache_key = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "prompt_cache_retention");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.prompt_cache_retention = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "safety_identifier");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.safety_identifier = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "service_tier");
      field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.service_tier = std::move(field.value());
  }
  if (auto field = core::optional_string_map(value, "metadata"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.metadata = std::move(field.value());
  }
  const auto *stream_options_member = json_member(value, "stream_options");
  if (stream_options_member != nullptr && !stream_options_member->IsNull()) {
    auto parsed_stream_options = parse_stream_options_value(*stream_options_member);
    if (parsed_stream_options.has_error()) {
      return result<response>::failure(parsed_stream_options.error());
    }
    parsed.stream_options_value = std::move(parsed_stream_options.value());
  }
  if (auto field = optional_raw_json(value, "conversation"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.conversation = std::move(field.value());
  }
  if (auto field = optional_raw_json(value, "prompt"); field.has_error()) {
    return result<response>::failure(field.error());
  } else {
    parsed.prompt = std::move(field.value());
  }

  const auto *instructions = json_member(value, "instructions");
  if (instructions != nullptr && !instructions->IsNull()) {
    auto parsed_instructions =
        parse_instruction_payload_value(*instructions, options);
    if (parsed_instructions.has_error()) {
      return result<response>::failure(parsed_instructions.error());
    }
    parsed.instructions = std::move(parsed_instructions.value());
  }

  const auto *output = json_member(value, "output");
  if (output != nullptr && !output->IsNull()) {
    if (!output->IsArray()) {
      return result<response>::failure(
          make_error(errc::type_mismatch, "output must be an array"));
    }
    parsed.output_items.reserve(output->Size());
    for (const auto &entry : output->GetArray()) {
      auto parsed_item = parse_item_value(entry, options);
      if (parsed_item.has_error()) {
        return result<response>::failure(parsed_item.error());
      }
      parsed.output_items.push_back(std::move(parsed_item.value()));
    }
  }

  const auto *tools = json_member(value, "tools");
  if (tools != nullptr && !tools->IsNull()) {
    if (!tools->IsArray()) {
      return result<response>::failure(
          make_error(errc::type_mismatch, "tools must be an array"));
    }
    parsed.tools.reserve(tools->Size());
    for (const auto &entry : tools->GetArray()) {
      auto parsed_tool = parse_tool_definition_value(entry, options);
      if (parsed_tool.has_error()) {
        return result<response>::failure(parsed_tool.error());
      }
      parsed.tools.push_back(std::move(parsed_tool.value()));
    }
  }

  const auto *tool_choice = json_member(value, "tool_choice");
  if (tool_choice != nullptr && !tool_choice->IsNull()) {
    auto parsed_choice = parse_tool_choice_value(*tool_choice, options);
    if (parsed_choice.has_error()) {
      return result<response>::failure(parsed_choice.error());
    }
    parsed.tool_choice_value = std::move(parsed_choice.value());
  }

  const auto *error = json_member(value, "error");
  if (error != nullptr && !error->IsNull()) {
    auto parsed_error = parse_response_error_value(*error, options);
    if (parsed_error.has_error()) {
      return result<response>::failure(parsed_error.error());
    }
    parsed.error = std::move(parsed_error.value());
  }

  const auto *incomplete = json_member(value, "incomplete_details");
  if (incomplete != nullptr && !incomplete->IsNull()) {
    if (!incomplete->IsObject()) {
      return result<response>::failure(make_error(
          errc::type_mismatch, "incomplete_details must be an object"));
    }
    incomplete_details parsed_incomplete{};
    if (auto reason = core::optional_string(*incomplete, "reason");
        reason.has_error()) {
      return result<response>::failure(reason.error());
    } else {
      parsed_incomplete.reason = std::move(reason.value());
    }
    parsed.incomplete = std::move(parsed_incomplete);
  }

  const auto *usage = json_member(value, "usage");
  if (usage != nullptr && !usage->IsNull()) {
    auto parsed_usage = parse_usage_value(*usage);
    if (parsed_usage.has_error()) {
      return result<response>::failure(parsed_usage.error());
    }
    parsed.usage_value = std::move(parsed_usage.value());
  }

  const auto *text = json_member(value, "text");
  if (text != nullptr && !text->IsNull()) {
    auto parsed_text = parse_text_config_value(*text);
    if (parsed_text.has_error()) {
      return result<response>::failure(parsed_text.error());
    }
    parsed.text = std::move(parsed_text.value());
  }

  const auto *reasoning = json_member(value, "reasoning");
  if (reasoning != nullptr && !reasoning->IsNull()) {
    auto parsed_reasoning = parse_reasoning_config_value(*reasoning);
    if (parsed_reasoning.has_error()) {
      return result<response>::failure(parsed_reasoning.error());
    }
    parsed.reasoning = std::move(parsed_reasoning.value());
  }

  return result<response>::success(std::move(parsed));
}

[[nodiscard]] auto parse_request_value(const json_value &value,
                                       const parse_options &options)
    -> result<request> {
  if (!value.IsObject()) {
    return result<request>::failure(
        make_error(errc::type_mismatch, "request must be an object"));
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
  const auto *instructions = json_member(value, "instructions");
  if (instructions != nullptr && !instructions->IsNull()) {
    auto parsed_instructions =
        parse_instruction_payload_value(*instructions, options);
    if (parsed_instructions.has_error()) {
      return result<request>::failure(parsed_instructions.error());
    }
    parsed.instructions = std::move(parsed_instructions.value());
  }
  if (auto field = core::optional_bool(value, "background"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.background = field.value();
  }
  if (auto field = core::optional_bool(value, "parallel_tool_calls");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.parallel_tool_calls = field.value();
  }
  if (auto field = core::optional_bool(value, "store"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.store = field.value();
  }
  if (auto field = core::optional_bool(value, "stream"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.stream = field.value();
  }
  if (auto field = core::optional_string(value, "previous_response_id");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.previous_response_id = std::move(field.value());
  }
  if (auto field = core::optional_int64(value, "max_output_tokens");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.max_output_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "max_tool_calls");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.max_tool_calls = field.value();
  }
  if (auto field = core::optional_double(value, "temperature"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.temperature = field.value();
  }
  if (auto field = core::optional_double(value, "top_p"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.top_p = field.value();
  }
  if (auto field = core::optional_double(value, "top_logprobs");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.top_logprobs = field.value();
  }
  if (auto field = core::optional_string(value, "truncation"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.truncation = std::move(field.value());
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
  if (auto field = core::optional_string(value, "safety_identifier");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.safety_identifier = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "service_tier");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.service_tier = std::move(field.value());
  }
  if (auto field = core::optional_string_array(value, "include");
      field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.include = std::move(field.value());
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
  const auto *stream_options_member = json_member(value, "stream_options");
  if (stream_options_member != nullptr && !stream_options_member->IsNull()) {
    auto parsed_stream_options = parse_stream_options_value(*stream_options_member);
    if (parsed_stream_options.has_error()) {
      return result<request>::failure(parsed_stream_options.error());
    }
    parsed.stream_options_value = std::move(parsed_stream_options.value());
  }
  if (auto field = optional_raw_json(value, "conversation"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.conversation = std::move(field.value());
  }
  if (auto field = optional_raw_json(value, "prompt"); field.has_error()) {
    return result<request>::failure(field.error());
  } else {
    parsed.prompt = std::move(field.value());
  }
  const auto *context_management = json_member(value, "context_management");
  if (context_management != nullptr && !context_management->IsNull()) {
    if (!context_management->IsArray()) {
      return result<request>::failure(make_error(
          errc::type_mismatch, "context_management must be an array"));
    }
    parsed.context_management.reserve(context_management->Size());
    for (const auto &entry : context_management->GetArray()) {
      auto parsed_entry = parse_context_management_value(entry);
      if (parsed_entry.has_error()) {
        return result<request>::failure(parsed_entry.error());
      }
      parsed.context_management.push_back(std::move(parsed_entry.value()));
    }
  }

  const auto *input = json_member(value, "input");
  if (input != nullptr && !input->IsNull()) {
    auto parsed_input = parse_input_payload_value(*input, options);
    if (parsed_input.has_error()) {
      return result<request>::failure(parsed_input.error());
    }
    parsed.input = std::move(parsed_input.value());
  }

  const auto *tools = json_member(value, "tools");
  if (tools != nullptr && !tools->IsNull()) {
    if (!tools->IsArray()) {
      return result<request>::failure(
          make_error(errc::type_mismatch, "tools must be an array"));
    }
    parsed.tools.reserve(tools->Size());
    for (const auto &entry : tools->GetArray()) {
      auto parsed_tool = parse_tool_definition_value(entry, options);
      if (parsed_tool.has_error()) {
        return result<request>::failure(parsed_tool.error());
      }
      parsed.tools.push_back(std::move(parsed_tool.value()));
    }
  }

  const auto *tool_choice = json_member(value, "tool_choice");
  if (tool_choice != nullptr && !tool_choice->IsNull()) {
    auto parsed_choice = parse_tool_choice_value(*tool_choice, options);
    if (parsed_choice.has_error()) {
      return result<request>::failure(parsed_choice.error());
    }
    parsed.tool_choice_value = std::move(parsed_choice.value());
  }

  const auto *reasoning = json_member(value, "reasoning");
  if (reasoning != nullptr && !reasoning->IsNull()) {
    auto parsed_reasoning = parse_reasoning_config_value(*reasoning);
    if (parsed_reasoning.has_error()) {
      return result<request>::failure(parsed_reasoning.error());
    }
    parsed.reasoning = std::move(parsed_reasoning.value());
  }

  const auto *text = json_member(value, "text");
  if (text != nullptr && !text->IsNull()) {
    auto parsed_text = parse_text_config_value(*text);
    if (parsed_text.has_error()) {
      return result<request>::failure(parsed_text.error());
    }
    parsed.text = std::move(parsed_text.value());
  }

  return result<request>::success(std::move(parsed));
}

template <typename parsed_t>
[[nodiscard]] auto parse_single_object(std::string_view json_text,
                                       const parse_options &options,
                                       auto &&parser) -> result<parsed_t> {
  auto document = parse_json(json_text);
  if (document.has_error()) {
    return result<parsed_t>::failure(document.error());
  }
  return parser(document.value(), options);
}

} // namespace

auto parse_request(std::string_view json_text, const parse_options &options)
    -> core::result<request> {
  return parse_single_object<request>(json_text, options, parse_request_value);
}

auto parse_response(std::string_view json_text, const parse_options &options)
    -> core::result<response> {
  return parse_single_object<response>(json_text, options, parse_response_value);
}

auto parse_item_json(std::string_view json_text, const parse_options &options)
    -> core::result<item> {
  return parse_single_object<item>(json_text, options, parse_item_value);
}

auto parse_message_content_part_json(std::string_view json_text,
                                     const parse_options &options)
    -> core::result<message_content_part> {
  return parse_single_object<message_content_part>(
      json_text, options, parse_message_content_part_value);
}

auto parse_tool_definition_json(std::string_view json_text,
                                const parse_options &options)
    -> core::result<tool_definition> {
  return parse_single_object<tool_definition>(json_text, options,
                                              parse_tool_definition_value);
}

auto parse_tool_choice_json(std::string_view json_text,
                            const parse_options &options)
    -> core::result<tool_choice> {
  return parse_single_object<tool_choice>(json_text, options,
                                          parse_tool_choice_value);
}

auto parse_response_error_json(std::string_view json_text,
                               const parse_options &options)
    -> core::result<response_error> {
  return parse_single_object<response_error>(json_text, options,
                                             parse_response_error_value);
}

} // namespace openai::responses
