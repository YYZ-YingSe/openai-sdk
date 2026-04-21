#include "openai/streaming/serializer.hpp"

#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

#include "openai/core/error.hpp"
#include "openai/core/json.hpp"
#include "openai/responses/serializer.hpp"

namespace openai::streaming {
namespace {

using core::errc;
using core::make_error;
using core::parse_json;
using core::result;
using object_t = core::value::object;

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

auto insert_optional_int64(object_t &object, std::string_view key,
                           const std::optional<std::int64_t> &value) -> void {
  if (value.has_value()) {
    insert_value(object, key, core::value{*value});
  }
}

auto insert_json_string(object_t &object, std::string_view key,
                        const core::result<std::string> &json)
    -> result<void> {
  if (json.has_error()) {
    return result<void>::failure(json.error());
  }
  auto document = parse_json(json.value());
  if (document.has_error()) {
    return result<void>::failure(document.error());
  }
  auto converted = core::json_to_value(document.value());
  if (converted.has_error()) {
    return result<void>::failure(converted.error());
  }
  insert_value(object, key, std::move(converted.value()));
  return result<void>::success();
}

auto uint64_retry_to_int64(std::int64_t value) -> result<core::value> {
  if (value < 0) {
    return result<core::value>::failure(
        make_error(errc::type_mismatch, "retry must be non-negative"));
  }
  return result<core::value>::success(core::value{value});
}

auto serialize_annotation_value(const responses::annotation &value)
    -> result<core::value> {
  if (!value.raw_json.empty()) {
    auto document = parse_json(value.raw_json);
    if (document.has_error()) {
      return result<core::value>::failure(document.error());
    }
    return core::json_to_value(document.value());
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

auto serialize_response_event_value(const response_event &value)
    -> result<core::value> {
  if (value.type == "[DONE]") {
    return result<core::value>::failure(
        make_error(errc::not_supported, "[DONE] is not a JSON event payload"));
  }

  object_t object{};
  insert_value(object, "type", core::value{value.type});
  insert_optional_string(object, "response_id", value.response_id);
  insert_optional_string(object, "item_id", value.item_id);
  insert_optional_string(object, "status", value.status);
  insert_optional_int64(object, "output_index", value.output_index);
  insert_optional_int64(object, "content_index", value.content_index);
  insert_optional_int64(object, "item_index", value.item_index);
  insert_optional_int64(object, "summary_index", value.summary_index);
  insert_optional_int64(object, "sequence_number", value.sequence_number);
  insert_optional_string(object, "delta", value.delta);
  insert_optional_string(object, "text", value.text);
  insert_optional_string(object, "arguments", value.arguments);
  insert_optional_string(object, "code", value.code);
  insert_optional_string(object, "transcript", value.transcript);
  insert_optional_string(object, "partial_image_b64", value.partial_image_b64);

  if (value.annotation.has_value()) {
    auto annotation = serialize_annotation_value(*value.annotation);
    if (annotation.has_error()) {
      return result<core::value>::failure(annotation.error());
    }
    insert_value(object, "annotation", std::move(annotation.value()));
  }
  if (value.item.has_value()) {
    auto inserted =
        insert_json_string(object, "item", responses::serialize_item(*value.item));
    if (inserted.has_error()) {
      return result<core::value>::failure(inserted.error());
    }
  }
  if (value.content_part.has_value()) {
    auto inserted = insert_json_string(
        object, "content_part",
        responses::serialize_message_content_part(*value.content_part));
    if (inserted.has_error()) {
      return result<core::value>::failure(inserted.error());
    }
  }
  if (value.response.has_value()) {
    auto inserted = insert_json_string(
        object, "response", responses::serialize_response(*value.response));
    if (inserted.has_error()) {
      return result<core::value>::failure(inserted.error());
    }
  }
  if (value.error.has_value()) {
    auto inserted = insert_json_string(
        object, "error", responses::serialize_response_error(*value.error));
    if (inserted.has_error()) {
      return result<core::value>::failure(inserted.error());
    }
  }
  for (const auto &[key, entry] : value.extra_fields) {
    insert_value(object, key, entry);
  }
  return result<core::value>::success(core::value{std::move(object)});
}

} // namespace

auto serialize_response_event_json(const response_event &value,
                                   const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_response_event_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_sse_event(const sse_event &value) -> core::result<std::string> {
  std::string text{};
  if (!value.event.empty()) {
    text += "event: ";
    text += value.event;
    text.push_back('\n');
  }
  if (value.id.has_value()) {
    text += "id: ";
    text += *value.id;
    text.push_back('\n');
  }
  if (value.retry.has_value()) {
    auto retry = uint64_retry_to_int64(*value.retry);
    if (retry.has_error()) {
      return core::result<std::string>::failure(retry.error());
    }
    text += "retry: ";
    text += std::to_string(std::get<std::int64_t>(retry.value().storage));
    text.push_back('\n');
  }

  std::size_t start = 0U;
  do {
    const auto end = value.data.find('\n', start);
    const auto line = end == std::string::npos
                          ? std::string_view{value.data}.substr(start)
                          : std::string_view{value.data}.substr(start, end - start);
    text += "data: ";
    text.append(line.data(), line.size());
    text.push_back('\n');
    if (end == std::string::npos) {
      break;
    }
    start = end + 1U;
  } while (start <= value.data.size());

  text.push_back('\n');
  return core::result<std::string>::success(std::move(text));
}

auto serialize_sse_stream(const std::vector<sse_event> &events)
    -> core::result<std::string> {
  std::string text{};
  for (const auto &event : events) {
    auto serialized = serialize_sse_event(event);
    if (serialized.has_error()) {
      return core::result<std::string>::failure(serialized.error());
    }
    text += serialized.value();
  }
  return core::result<std::string>::success(std::move(text));
}

} // namespace openai::streaming
