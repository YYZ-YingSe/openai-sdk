#include "openai/responses/serializer.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "openai/core/error.hpp"
#include "openai/core/json.hpp"

namespace openai::responses {
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

auto raw_json_to_value(const raw_json &value) -> result<core::value> {
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

auto insert_optional_double(object_t &object, std::string_view key,
                            const std::optional<double> &value) -> void {
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

auto insert_json_blob(object_t &object, std::string_view key,
                      const std::optional<std::string> &blob) -> result<void> {
  if (!blob.has_value()) {
    return result<void>::success();
  }
  auto parsed = to_json_value(*blob);
  if (parsed.has_error()) {
    return result<void>::failure(parsed.error());
  }
  insert_value(object, key, std::move(parsed.value()));
  return result<void>::success();
}

auto insert_raw_json(object_t &object, std::string_view key,
                     const std::optional<raw_json> &blob) -> result<void> {
  if (!blob.has_value()) {
    return result<void>::success();
  }
  auto parsed = raw_json_to_value(*blob);
  if (parsed.has_error()) {
    return result<void>::failure(parsed.error());
  }
  insert_value(object, key, std::move(parsed.value()));
  return result<void>::success();
}

auto serialize_annotation_value(const annotation &value) -> result<core::value> {
  if (!value.raw_json.empty()) {
    return to_json_value(value.raw_json);
  }
  object_t object{};
  if (!value.type.empty()) {
    insert_value(object, "type", core::value{value.type});
  }
  insert_optional_string(object, "title", value.title);
  insert_optional_string(object, "url", value.url);
  insert_optional_string(object, "text", value.text);
  insert_optional_string(object, "file_id", value.file_id);
  insert_optional_string(object, "filename", value.filename);
  insert_optional_int64(object, "index", value.index);
  insert_optional_int64(object, "start_index", value.start_index);
  insert_optional_int64(object, "end_index", value.end_index);
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_logprob_token_value(const logprob_token &value)
    -> result<core::value> {
  object_t object{};
  insert_value(object, "token", core::value{value.token});
  insert_optional_double(object, "logprob", value.logprob);
  if (!value.bytes.empty()) {
    array_t bytes{};
    bytes.reserve(value.bytes.size());
    for (const auto entry : value.bytes) {
      bytes.emplace_back(entry);
    }
    insert_value(object, "bytes", core::value{std::move(bytes)});
  }
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_message_content_part_value(const message_content_part &value)
    -> result<core::value> {
  return std::visit(
      []<typename part_t>(const part_t &current) -> result<core::value> {
        using current_t = std::decay_t<part_t>;
        if constexpr (std::is_same_v<current_t, input_text_content>) {
          object_t object{};
          insert_value(object, "type", core::value{"input_text"});
          insert_value(object, "text", core::value{current.text});
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, input_image_content>) {
          object_t object{};
          insert_value(object, "type", core::value{"input_image"});
          insert_optional_string(object, "image_url", current.image_url);
          insert_optional_string(object, "file_id", current.file_id);
          insert_optional_string(object, "detail", current.detail);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, input_file_content>) {
          object_t object{};
          insert_value(object, "type", core::value{"input_file"});
          insert_optional_string(object, "file_id", current.file_id);
          insert_optional_string(object, "file_url", current.file_url);
          insert_optional_string(object, "filename", current.filename);
          insert_optional_string(object, "file_data", current.file_data);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, output_text_content>) {
          object_t object{};
          insert_value(object, "type", core::value{"output_text"});
          insert_value(object, "text", core::value{current.text});
          if (!current.annotations.empty()) {
            array_t annotations{};
            annotations.reserve(current.annotations.size());
            for (const auto &entry : current.annotations) {
              auto serialized = serialize_annotation_value(entry);
              if (serialized.has_error()) {
                return result<core::value>::failure(serialized.error());
              }
              annotations.push_back(std::move(serialized.value()));
            }
            insert_value(object, "annotations",
                         core::value{std::move(annotations)});
          }
          if (!current.logprobs.empty()) {
            array_t logprobs{};
            logprobs.reserve(current.logprobs.size());
            for (const auto &entry : current.logprobs) {
              auto serialized = serialize_logprob_token_value(entry);
              if (serialized.has_error()) {
                return result<core::value>::failure(serialized.error());
              }
              logprobs.push_back(std::move(serialized.value()));
            }
            insert_value(object, "logprobs", core::value{std::move(logprobs)});
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, refusal_content>) {
          object_t object{};
          insert_value(object, "type", core::value{"refusal"});
          insert_value(object, "refusal", core::value{current.refusal});
          return result<core::value>::success(core::value{std::move(object)});
        } else {
          return raw_json_to_value(current);
        }
      },
      value);
}

auto serialize_tool_output_payload_value(const tool_output_payload &value)
    -> result<core::value> {
  return std::visit(
      []<typename payload_t>(const payload_t &current) -> result<core::value> {
        using current_t = std::decay_t<payload_t>;
        if constexpr (std::is_same_v<current_t, std::string>) {
          return result<core::value>::success(core::value{current});
        } else {
          array_t parts{};
          parts.reserve(current.size());
          for (const auto &entry : current) {
            auto serialized = serialize_message_content_part_value(entry);
            if (serialized.has_error()) {
              return result<core::value>::failure(serialized.error());
            }
            parts.push_back(std::move(serialized.value()));
          }
          return result<core::value>::success(core::value{std::move(parts)});
        }
      },
      value);
}

auto insert_call_output_payload(object_t &object,
                                const std::optional<std::string> &output,
                                const std::optional<std::string> &output_json,
                                const std::optional<tool_output_payload> &payload)
    -> result<void> {
  if (output_json.has_value()) {
    return insert_json_blob(object, "output", output_json);
  }
  if (payload.has_value()) {
    auto serialized = serialize_tool_output_payload_value(*payload);
    if (serialized.has_error()) {
      return result<void>::failure(serialized.error());
    }
    insert_value(object, "output", std::move(serialized.value()));
    return result<void>::success();
  }
  if (output.has_value()) {
    insert_value(object, "output", core::value{*output});
  }
  return result<void>::success();
}

auto insert_common_item_fields(object_t &object, const std::optional<std::string> &id,
                               const std::optional<std::string> &status) -> void {
  insert_optional_string(object, "id", id);
  insert_optional_string(object, "status", status);
}

auto serialize_item_value(const item &value) -> result<core::value> {
  return std::visit(
      []<typename item_t>(const item_t &current) -> result<core::value> {
        using current_t = std::decay_t<item_t>;
        if constexpr (std::is_same_v<current_t, raw_json>) {
          return raw_json_to_value(current);
        } else if constexpr (std::is_same_v<current_t, message_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"message"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "phase", current.phase);
          insert_value(object, "role", core::value{current.role});
          if (!current.content.empty()) {
            array_t content{};
            content.reserve(current.content.size());
            for (const auto &entry : current.content) {
              auto serialized = serialize_message_content_part_value(entry);
              if (serialized.has_error()) {
                return result<core::value>::failure(serialized.error());
              }
              content.push_back(std::move(serialized.value()));
            }
            insert_value(object, "content", core::value{std::move(content)});
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, reasoning_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"reasoning"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "encrypted_content",
                                 current.encrypted_content);
          if (!current.summary.empty()) {
            array_t summary{};
            summary.reserve(current.summary.size());
            for (const auto &entry : current.summary) {
              object_t summary_entry{};
              insert_value(summary_entry, "text", core::value{entry.text});
              summary.emplace_back(std::move(summary_entry));
            }
            insert_value(object, "summary", core::value{std::move(summary)});
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, compaction_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"compaction"});
          insert_common_item_fields(object, current.id, current.status);
          if (auto inserted = insert_json_blob(object, "summary",
                                               current.summary_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, item_reference>) {
          object_t object{};
          insert_value(object, "type", core::value{"item_reference"});
          insert_optional_string(object, "id", current.id);
          insert_optional_string(object, "item_id", current.item_id);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, function_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"function_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          insert_value(object, "name", core::value{current.name});
          insert_value(object, "arguments", core::value{current.arguments});
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, function_call_output_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"function_call_output"});
          insert_common_item_fields(object, current.id, current.status);
          insert_value(object, "call_id", core::value{current.call_id});
          if (auto inserted = insert_call_output_payload(object, current.output,
                                                         current.output_json,
                                                         current.output_payload);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, function_shell_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"function_shell_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          if (auto inserted = insert_json_blob(object, "action",
                                               current.action_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "result",
                                               current.result_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, function_shell_call_output_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"function_shell_call_output"});
          insert_common_item_fields(object, current.id, current.status);
          insert_value(object, "call_id", core::value{current.call_id});
          if (auto inserted = insert_json_blob(object, "output",
                                               current.output_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, file_search_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"file_search_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_string_array(object, "queries", current.queries);
          if (auto inserted = insert_json_blob(object, "results",
                                               current.results_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, tool_search_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"tool_search_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          insert_string_array(object, "queries", current.queries);
          if (auto inserted = insert_json_blob(object, "results",
                                               current.results_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, tool_search_output_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"tool_search_output"});
          insert_common_item_fields(object, current.id, current.status);
          insert_value(object, "call_id", core::value{current.call_id});
          if (auto inserted = insert_json_blob(object, "output",
                                               current.output_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, web_search_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"web_search_call"});
          insert_common_item_fields(object, current.id, current.status);
          if (auto inserted = insert_json_blob(object, "action",
                                               current.action_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, computer_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"computer_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          if (auto inserted = insert_json_blob(object, "action",
                                               current.action_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "pending_safety_checks",
                                               current.pending_safety_checks_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, computer_call_output_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"computer_call_output"});
          insert_common_item_fields(object, current.id, current.status);
          insert_value(object, "call_id", core::value{current.call_id});
          if (auto inserted = insert_json_blob(object, "output",
                                               current.output_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, local_shell_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"local_shell_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          if (auto inserted = insert_json_blob(object, "action",
                                               current.action_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "output",
                                               current.output_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, local_shell_call_output_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"local_shell_call_output"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          if (current.output_json.has_value()) {
            if (auto inserted = insert_json_blob(object, "output",
                                                 current.output_json);
                inserted.has_error()) {
              return result<core::value>::failure(inserted.error());
            }
          } else if (current.output.has_value()) {
            insert_value(object, "output", core::value{*current.output});
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, shell_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"shell_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          if (auto inserted = insert_json_blob(object, "action",
                                               current.action_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "output",
                                               current.output_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, shell_call_output_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"shell_call_output"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          if (current.output_json.has_value()) {
            if (auto inserted = insert_json_blob(object, "output",
                                                 current.output_json);
                inserted.has_error()) {
              return result<core::value>::failure(inserted.error());
            }
          } else if (current.output.has_value()) {
            insert_value(object, "output", core::value{*current.output});
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, custom_tool_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"custom_tool_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          insert_value(object, "name", core::value{current.name});
          insert_optional_string(object, "input", current.input);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, custom_tool_call_output_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"custom_tool_call_output"});
          insert_common_item_fields(object, current.id, current.status);
          insert_value(object, "call_id", core::value{current.call_id});
          if (auto inserted = insert_call_output_payload(object, current.output,
                                                         current.output_json,
                                                         current.output_payload);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, image_generation_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"image_generation_call"});
          insert_common_item_fields(object, current.id, current.status);
          if (auto inserted = insert_json_blob(object, "result",
                                               current.result_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, code_interpreter_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"code_interpreter_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          insert_optional_string(object, "code", current.code);
          if (auto inserted = insert_json_blob(object, "outputs",
                                               current.outputs_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, code_interpreter_call_output_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"code_interpreter_call_output"});
          insert_common_item_fields(object, current.id, current.status);
          insert_value(object, "call_id", core::value{current.call_id});
          if (auto inserted = insert_json_blob(object, "output",
                                               current.output_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, mcp_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"mcp_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "name", current.name);
          insert_optional_string(object, "arguments", current.arguments);
          insert_optional_string(object, "server_label", current.server_label);
          insert_optional_string(object, "approval_request_id",
                                 current.approval_request_id);
          if (auto inserted = insert_json_blob(object, "output",
                                               current.output_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "error",
                                               current.error_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, mcp_list_tools_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"mcp_list_tools"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "server_label", current.server_label);
          if (auto inserted = insert_json_blob(object, "tools",
                                               current.tools_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          insert_optional_string(object, "error", current.error);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, mcp_approval_request_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"mcp_approval_request"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "name", current.name);
          insert_optional_string(object, "arguments", current.arguments);
          insert_optional_string(object, "server_label", current.server_label);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, mcp_approval_response_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"mcp_approval_response"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "approval_request_id",
                                 current.approval_request_id);
          insert_optional_bool(object, "approve", current.approve);
          insert_optional_string(object, "reason", current.reason);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, apply_patch_call_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"apply_patch_call"});
          insert_common_item_fields(object, current.id, current.status);
          insert_optional_string(object, "call_id", current.call_id);
          insert_optional_string(object, "input", current.input);
          if (auto inserted = insert_json_blob(object, "output",
                                               current.output_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, apply_patch_call_output_item>) {
          object_t object{};
          insert_value(object, "type", core::value{"apply_patch_call_output"});
          insert_common_item_fields(object, current.id, current.status);
          insert_value(object, "call_id", core::value{current.call_id});
          if (auto inserted = insert_json_blob(object, "output",
                                               current.output_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        }
      },
      value);
}

auto serialize_input_payload_value(const input_payload &value)
    -> result<core::value> {
  return std::visit(
      []<typename payload_t>(const payload_t &current) -> result<core::value> {
        using current_t = std::decay_t<payload_t>;
        if constexpr (std::is_same_v<current_t, std::string>) {
          return result<core::value>::success(core::value{current});
        } else {
          array_t items{};
          items.reserve(current.size());
          for (const auto &entry : current) {
            auto serialized = serialize_item_value(entry);
            if (serialized.has_error()) {
              return result<core::value>::failure(serialized.error());
            }
            items.push_back(std::move(serialized.value()));
          }
          return result<core::value>::success(core::value{std::move(items)});
        }
      },
      value);
}

auto serialize_instructions_payload_value(const instructions_payload &value)
    -> result<core::value> {
  return std::visit(
      []<typename payload_t>(const payload_t &current) -> result<core::value> {
        using current_t = std::decay_t<payload_t>;
        if constexpr (std::is_same_v<current_t, std::string>) {
          return result<core::value>::success(core::value{current});
        } else {
          array_t items{};
          items.reserve(current.size());
          for (const auto &entry : current) {
            auto serialized = serialize_item_value(entry);
            if (serialized.has_error()) {
              return result<core::value>::failure(serialized.error());
            }
            items.push_back(std::move(serialized.value()));
          }
          return result<core::value>::success(core::value{std::move(items)});
        }
      },
      value);
}

auto serialize_reasoning_config_value(const reasoning_config &value)
    -> result<core::value> {
  object_t object{};
  insert_optional_string(object, "effort", value.effort);
  insert_optional_string(object, "generate_summary", value.generate_summary);
  insert_optional_string(object, "summary", value.summary);
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_text_config_value(const response_text_config &value)
    -> result<core::value> {
  object_t text{};
  if (value.verbosity.has_value()) {
    insert_value(text, "verbosity", core::value{*value.verbosity});
  }

  object_t format{};
  if (!value.type.empty()) {
    insert_value(format, "type", core::value{value.type});
  }
  if (value.schema_name.has_value() || value.schema_description.has_value() ||
      value.strict.has_value() || value.schema_json.has_value()) {
    object_t schema{};
    insert_optional_string(schema, "name", value.schema_name);
    insert_optional_string(schema, "description", value.schema_description);
    insert_optional_bool(schema, "strict", value.strict);
    if (auto inserted = insert_json_blob(schema, "schema", value.schema_json);
        inserted.has_error()) {
      return result<core::value>::failure(inserted.error());
    }
    insert_value(format, "json_schema", core::value{std::move(schema)});
  }
  if (!format.empty()) {
    insert_value(text, "format", core::value{std::move(format)});
  }
  return result<core::value>::success(core::value{std::move(text)});
}

auto serialize_tool_choice_value(const tool_choice &value)
    -> result<core::value> {
  if (!value.raw_json.empty()) {
    return to_json_value(value.raw_json);
  }
  if (!value.type.has_value() && !value.name.has_value() &&
      !value.server_label.has_value() && !value.tools_json.has_value()) {
    return result<core::value>::success(core::value{value.mode});
  }
  object_t object{};
  if (value.type.has_value()) {
    insert_value(object, "type", core::value{*value.type});
  } else if (!value.mode.empty() && value.mode != "object") {
    insert_value(object, "type", core::value{value.mode});
  }
  insert_optional_string(object, "name", value.name);
  insert_optional_string(object, "server_label", value.server_label);
  if (auto inserted = insert_json_blob(object, "tools", value.tools_json);
      inserted.has_error()) {
    return result<core::value>::failure(inserted.error());
  }
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_tool_definition_value(const tool_definition &value)
    -> result<core::value> {
  return std::visit(
      []<typename tool_t>(const tool_t &current) -> result<core::value> {
        using current_t = std::decay_t<tool_t>;
        if constexpr (std::is_same_v<current_t, raw_json>) {
          return raw_json_to_value(current);
        } else if constexpr (std::is_same_v<current_t, function_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"function"});
          insert_value(object, "name", core::value{current.name});
          insert_optional_string(object, "description", current.description);
          insert_optional_bool(object, "strict", current.strict);
          if (auto inserted = insert_json_blob(object, "parameters",
                                               current.parameters_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, file_search_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"file_search"});
          insert_string_array(object, "vector_store_ids",
                              current.vector_store_ids);
          insert_optional_int64(object, "max_num_results",
                                current.max_num_results);
          if (auto inserted = insert_json_blob(object, "filters",
                                               current.filters_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "ranking_options",
                                               current.ranking_options_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, tool_search_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"tool_search"});
          insert_optional_string(object, "description", current.description);
          if (auto inserted = insert_json_blob(object, "filters",
                                               current.filters_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "ranking_options",
                                               current.ranking_options_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "user_location",
                                               current.user_location_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "config",
                                               current.raw_config_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, web_search_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"web_search"});
          insert_optional_string(object, "search_context_size",
                                 current.search_context_size);
          if (auto inserted = insert_json_blob(object, "user_location",
                                               current.user_location_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          if (auto inserted = insert_json_blob(object, "filters",
                                               current.filters_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, computer_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"computer_use"});
          insert_optional_string(object, "environment", current.environment);
          insert_optional_int64(object, "display_width",
                                current.display_width);
          insert_optional_int64(object, "display_height",
                                current.display_height);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, namespace_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"namespace"});
          insert_optional_string(object, "name", current.name);
          insert_optional_string(object, "description", current.description);
          if (auto inserted = insert_json_blob(object, "config",
                                               current.config_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, local_shell_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"local_shell"});
          insert_optional_string(object, "description", current.description);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, shell_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"shell"});
          insert_optional_string(object, "description", current.description);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, function_shell_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"function_shell"});
          insert_optional_string(object, "description", current.description);
          if (auto inserted = insert_json_blob(object, "parameters",
                                               current.parameters_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          insert_optional_bool(object, "strict", current.strict);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, custom_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"custom"});
          insert_value(object, "name", core::value{current.name});
          insert_optional_string(object, "description", current.description);
          insert_optional_string(object, "format", current.format);
          if (auto inserted = insert_json_blob(object, "grammar",
                                               current.grammar_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, image_generation_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"image_generation"});
          insert_optional_string(object, "model", current.model);
          insert_optional_string(object, "background", current.background);
          insert_optional_string(object, "output_format",
                                 current.output_format);
          insert_optional_string(object, "quality", current.quality);
          insert_optional_string(object, "size", current.size);
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, code_interpreter_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"code_interpreter"});
          if (auto inserted = insert_json_blob(object, "container",
                                               current.container_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, mcp_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"mcp"});
          insert_optional_string(object, "server_label", current.server_label);
          insert_optional_string(object, "server_url", current.server_url);
          insert_optional_string(object, "connector_id", current.connector_id);
          insert_string_array(object, "allowed_tools", current.allowed_tools);
          if (auto inserted = insert_json_blob(object, "headers",
                                               current.headers_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          insert_optional_string(object, "require_approval",
                                 current.require_approval);
          if (auto inserted = insert_json_blob(object, "authorization",
                                               current.authorization_json);
              inserted.has_error()) {
            return result<core::value>::failure(inserted.error());
          }
          return result<core::value>::success(core::value{std::move(object)});
        } else if constexpr (std::is_same_v<current_t, apply_patch_tool>) {
          object_t object{};
          insert_value(object, "type", core::value{"apply_patch"});
          return result<core::value>::success(core::value{std::move(object)});
        }
      },
      value);
}

auto serialize_usage_value(const usage &value) -> result<core::value> {
  object_t object{};
  insert_optional_int64(object, "input_tokens", value.input_tokens);
  insert_optional_int64(object, "output_tokens", value.output_tokens);
  insert_optional_int64(object, "total_tokens", value.total_tokens);

  object_t input_details{};
  insert_optional_int64(input_details, "cached_tokens",
                        value.input_details.cached_tokens);
  if (!input_details.empty()) {
    insert_value(object, "input_tokens_details",
                 core::value{std::move(input_details)});
  }

  object_t output_details{};
  insert_optional_int64(output_details, "reasoning_tokens",
                        value.output_details.reasoning_tokens);
  if (!output_details.empty()) {
    insert_value(object, "output_tokens_details",
                 core::value{std::move(output_details)});
  }

  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_response_error_value(const response_error &value)
    -> result<core::value> {
  if (!value.raw_json.empty()) {
    return to_json_value(value.raw_json);
  }
  object_t object{};
  insert_optional_string(object, "type", value.type);
  insert_optional_string(object, "code", value.code);
  insert_optional_string(object, "message", value.message);
  insert_optional_string(object, "param", value.param);
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_request_value(const request &value) -> result<core::value> {
  object_t object{};
  insert_optional_string(object, "model", value.model);
  if (value.input.has_value()) {
    auto serialized = serialize_input_payload_value(*value.input);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "input", std::move(serialized.value()));
  }
  if (value.instructions.has_value()) {
    auto serialized = serialize_instructions_payload_value(*value.instructions);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "instructions", std::move(serialized.value()));
  }
  if (!value.tools.empty()) {
    array_t tools{};
    tools.reserve(value.tools.size());
    for (const auto &entry : value.tools) {
      auto serialized = serialize_tool_definition_value(entry);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      tools.push_back(std::move(serialized.value()));
    }
    insert_value(object, "tools", core::value{std::move(tools)});
  }
  if (value.tool_choice_value.has_value()) {
    auto serialized = serialize_tool_choice_value(*value.tool_choice_value);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "tool_choice", std::move(serialized.value()));
  }
  if (value.reasoning.has_value()) {
    auto serialized = serialize_reasoning_config_value(*value.reasoning);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "reasoning", std::move(serialized.value()));
  }
  if (value.text.has_value()) {
    auto serialized = serialize_text_config_value(*value.text);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "text", std::move(serialized.value()));
  }
  insert_optional_bool(object, "background", value.background);
  insert_optional_bool(object, "parallel_tool_calls",
                       value.parallel_tool_calls);
  insert_optional_bool(object, "store", value.store);
  insert_optional_bool(object, "stream", value.stream);
  if (value.stream_options_value.has_value()) {
    object_t stream_options{};
    insert_optional_bool(stream_options, "include_obfuscation",
                         value.stream_options_value->include_obfuscation);
    if (!stream_options.empty()) {
      insert_value(object, "stream_options",
                   core::value{std::move(stream_options)});
    }
  }
  insert_optional_string(object, "previous_response_id",
                         value.previous_response_id);
  insert_optional_int64(object, "max_output_tokens", value.max_output_tokens);
  insert_optional_int64(object, "max_tool_calls", value.max_tool_calls);
  insert_optional_double(object, "temperature", value.temperature);
  insert_optional_double(object, "top_p", value.top_p);
  insert_optional_double(object, "top_logprobs", value.top_logprobs);
  insert_optional_string(object, "truncation", value.truncation);
  insert_optional_string(object, "prompt_cache_key", value.prompt_cache_key);
  insert_optional_string(object, "prompt_cache_retention",
                         value.prompt_cache_retention);
  insert_optional_string(object, "safety_identifier",
                         value.safety_identifier);
  insert_optional_string(object, "service_tier", value.service_tier);
  if (!value.context_management.empty()) {
    array_t entries{};
    entries.reserve(value.context_management.size());
    for (const auto &entry : value.context_management) {
      object_t current{};
      insert_value(current, "type", core::value{entry.type});
      insert_optional_int64(current, "compact_threshold",
                            entry.compact_threshold);
      entries.emplace_back(std::move(current));
    }
    insert_value(object, "context_management", core::value{std::move(entries)});
  }
  if (auto inserted = insert_raw_json(object, "conversation",
                                      value.conversation);
      inserted.has_error()) {
    return result<core::value>::failure(inserted.error());
  }
  if (auto inserted = insert_raw_json(object, "prompt", value.prompt);
      inserted.has_error()) {
    return result<core::value>::failure(inserted.error());
  }
  insert_string_array(object, "include", value.include);
  insert_string_map(object, "metadata", value.metadata);
  insert_optional_string(object, "user", value.user);
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_response_value(const response &value) -> result<core::value> {
  object_t object{};
  insert_optional_string(object, "id", value.id);
  insert_optional_string(object, "object", value.object);
  insert_optional_string(object, "model", value.model);
  insert_optional_string(object, "status", value.status);
  if (auto inserted = insert_optional_uint64(object, "created_at",
                                             value.created_at);
      inserted.has_error()) {
    return result<core::value>::failure(inserted.error());
  }
  if (auto inserted = insert_optional_uint64(object, "completed_at",
                                             value.completed_at);
      inserted.has_error()) {
    return result<core::value>::failure(inserted.error());
  }
  if (value.instructions.has_value()) {
    auto serialized = serialize_instructions_payload_value(*value.instructions);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "instructions", std::move(serialized.value()));
  }
  if (!value.output_items.empty()) {
    array_t output{};
    output.reserve(value.output_items.size());
    for (const auto &entry : value.output_items) {
      auto serialized = serialize_item_value(entry);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      output.push_back(std::move(serialized.value()));
    }
    insert_value(object, "output", core::value{std::move(output)});
  }
  if (!value.tools.empty()) {
    array_t tools{};
    tools.reserve(value.tools.size());
    for (const auto &entry : value.tools) {
      auto serialized = serialize_tool_definition_value(entry);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      tools.push_back(std::move(serialized.value()));
    }
    insert_value(object, "tools", core::value{std::move(tools)});
  }
  if (value.tool_choice_value.has_value()) {
    auto serialized = serialize_tool_choice_value(*value.tool_choice_value);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "tool_choice", std::move(serialized.value()));
  }
  if (value.error.has_value()) {
    auto serialized = serialize_response_error_value(*value.error);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "error", std::move(serialized.value()));
  }
  if (value.incomplete.has_value()) {
    object_t incomplete{};
    insert_optional_string(incomplete, "reason", value.incomplete->reason);
    insert_value(object, "incomplete_details",
                 core::value{std::move(incomplete)});
  }
  if (value.usage_value.has_value()) {
    auto serialized = serialize_usage_value(*value.usage_value);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "usage", std::move(serialized.value()));
  }
  if (value.text.has_value()) {
    auto serialized = serialize_text_config_value(*value.text);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "text", std::move(serialized.value()));
  }
  if (value.reasoning.has_value()) {
    auto serialized = serialize_reasoning_config_value(*value.reasoning);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "reasoning", std::move(serialized.value()));
  }
  insert_optional_bool(object, "background", value.background);
  insert_optional_bool(object, "parallel_tool_calls",
                       value.parallel_tool_calls);
  insert_optional_bool(object, "stream", value.stream);
  if (value.stream_options_value.has_value()) {
    object_t stream_options{};
    insert_optional_bool(stream_options, "include_obfuscation",
                         value.stream_options_value->include_obfuscation);
    if (!stream_options.empty()) {
      insert_value(object, "stream_options",
                   core::value{std::move(stream_options)});
    }
  }
  insert_optional_string(object, "previous_response_id",
                         value.previous_response_id);
  insert_optional_int64(object, "max_output_tokens", value.max_output_tokens);
  insert_optional_int64(object, "max_tool_calls", value.max_tool_calls);
  insert_optional_double(object, "temperature", value.temperature);
  insert_optional_double(object, "top_p", value.top_p);
  insert_optional_double(object, "top_logprobs", value.top_logprobs);
  insert_optional_string(object, "truncation", value.truncation);
  insert_optional_string(object, "prompt_cache_key", value.prompt_cache_key);
  insert_optional_string(object, "prompt_cache_retention",
                         value.prompt_cache_retention);
  insert_optional_string(object, "safety_identifier",
                         value.safety_identifier);
  insert_optional_string(object, "service_tier", value.service_tier);
  if (auto inserted = insert_raw_json(object, "conversation",
                                      value.conversation);
      inserted.has_error()) {
    return result<core::value>::failure(inserted.error());
  }
  if (auto inserted = insert_raw_json(object, "prompt", value.prompt);
      inserted.has_error()) {
    return result<core::value>::failure(inserted.error());
  }
  insert_string_map(object, "metadata", value.metadata);
  insert_optional_string(object, "user", value.user);
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

auto serialize_response(const response &value,
                        const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_response_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_item(const item &value, const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_item_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_message_content_part(const message_content_part &value,
                                    const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_message_content_part_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_tool_definition(const tool_definition &value,
                               const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_tool_definition_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_tool_choice(const tool_choice &value,
                           const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_tool_choice_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_response_error(const response_error &value,
                              const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_response_error_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

} // namespace openai::responses
