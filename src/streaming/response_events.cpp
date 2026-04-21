#include "openai/streaming/response_events.hpp"

#include <array>
#include <utility>

#include "openai/core/json.hpp"
#include "openai/responses/parser.hpp"

namespace openai::streaming {
namespace {

using core::errc;
using core::json_member;
using core::json_to_string;
using core::make_error;
using core::parse_json;
using core::result;

[[nodiscard]] auto item_id_from_variant(const responses::item &item)
    -> std::optional<std::string> {
  return std::visit(
      []<typename item_t>(const item_t &current) -> std::optional<std::string> {
        using current_t = std::decay_t<item_t>;
        if constexpr (std::is_same_v<current_t, responses::raw_json>) {
          return std::nullopt;
        } else {
          return current.id;
        }
      },
      item);
}

[[nodiscard]] auto parse_annotation_value(const core::json_value &value)
    -> result<responses::annotation> {
  if (!value.IsObject()) {
    return result<responses::annotation>::failure(make_error(
        errc::type_mismatch, "annotation must be an object"));
  }
  responses::annotation parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<responses::annotation>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());
  if (auto field = core::optional_string(value, "type"); field.has_error()) {
    return result<responses::annotation>::failure(field.error());
  } else if (field.value().has_value()) {
    parsed.type = std::move(*field.value());
  }
  if (auto field = core::optional_string(value, "title"); field.has_error()) {
    return result<responses::annotation>::failure(field.error());
  } else {
    parsed.title = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "url"); field.has_error()) {
    return result<responses::annotation>::failure(field.error());
  } else {
    parsed.url = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "text"); field.has_error()) {
    return result<responses::annotation>::failure(field.error());
  } else {
    parsed.text = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "file_id"); field.has_error()) {
    return result<responses::annotation>::failure(field.error());
  } else {
    parsed.file_id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "filename"); field.has_error()) {
    return result<responses::annotation>::failure(field.error());
  } else {
    parsed.filename = std::move(field.value());
  }
  if (auto field = core::optional_int64(value, "index"); field.has_error()) {
    return result<responses::annotation>::failure(field.error());
  } else {
    parsed.index = field.value();
  }
  if (auto field = core::optional_int64(value, "start_index"); field.has_error()) {
    return result<responses::annotation>::failure(field.error());
  } else {
    parsed.start_index = field.value();
  }
  if (auto field = core::optional_int64(value, "end_index"); field.has_error()) {
    return result<responses::annotation>::failure(field.error());
  } else {
    parsed.end_index = field.value();
  }
  return result<responses::annotation>::success(std::move(parsed));
}

[[nodiscard]] auto is_known_event_field(std::string_view key) noexcept -> bool {
  static constexpr std::array<std::string_view, 19> known_fields{
      "type",          "response_id",    "item_id",     "status",
      "output_index",  "content_index",  "item_index",  "summary_index",
      "sequence_number", "delta",        "text",        "arguments",
      "code",          "transcript",     "partial_image_b64",
      "item",          "output_item",    "part",        "content_part"};
  for (const auto candidate : known_fields) {
    if (candidate == key) {
      return true;
    }
  }
  return key == "response" || key == "error" || key == "annotation";
}

} // namespace

auto parse_response_event_json(std::string_view json_text)
    -> core::result<response_event> {
  auto document = parse_json(json_text);
  if (document.has_error()) {
    return result<response_event>::failure(document.error());
  }
  if (!document.value().IsObject()) {
    return result<response_event>::failure(
        make_error(errc::type_mismatch, "response event must be an object"));
  }

  const auto &root = document.value();
  response_event parsed{};
  auto raw = json_to_string(root);
  if (raw.has_error()) {
    return result<response_event>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());

  if (auto type = core::required_string(root, "type"); type.has_error()) {
    return result<response_event>::failure(type.error());
  } else {
    parsed.type = std::move(type.value());
  }
  if (auto field = core::optional_string(root, "response_id"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.response_id = std::move(field.value());
  }
  if (auto field = core::optional_string(root, "item_id"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.item_id = std::move(field.value());
  }
  if (auto field = core::optional_string(root, "status"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.status = std::move(field.value());
  }
  if (auto field = core::optional_int64(root, "output_index"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.output_index = field.value();
  }
  if (auto field = core::optional_int64(root, "content_index");
      field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.content_index = field.value();
  }
  if (auto field = core::optional_int64(root, "item_index"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.item_index = field.value();
  }
  if (auto field = core::optional_int64(root, "summary_index");
      field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.summary_index = field.value();
  }
  if (auto field = core::optional_int64(root, "sequence_number");
      field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.sequence_number = field.value();
  }
  if (auto field = core::optional_string(root, "delta"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.delta = std::move(field.value());
  }
  if (auto field = core::optional_string(root, "text"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.text = std::move(field.value());
  }
  if (auto field = core::optional_string(root, "arguments"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.arguments = std::move(field.value());
  }
  if (auto field = core::optional_string(root, "code"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.code = std::move(field.value());
  }
  if (auto field = core::optional_string(root, "transcript"); field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.transcript = std::move(field.value());
  }
  if (auto field = core::optional_string(root, "partial_image_b64");
      field.has_error()) {
    return result<response_event>::failure(field.error());
  } else {
    parsed.partial_image_b64 = std::move(field.value());
  }

  const auto *annotation = json_member(root, "annotation");
  if (annotation != nullptr && !annotation->IsNull()) {
    auto parsed_annotation = parse_annotation_value(*annotation);
    if (parsed_annotation.has_error()) {
      return result<response_event>::failure(parsed_annotation.error());
    }
    parsed.annotation = std::move(parsed_annotation.value());
  }

  const auto *item = json_member(root, "item");
  if (item == nullptr) {
    item = json_member(root, "output_item");
  }
  if (item != nullptr && !item->IsNull()) {
    auto serialized = json_to_string(*item);
    if (serialized.has_error()) {
      return result<response_event>::failure(serialized.error());
    }
    auto parsed_item = responses::parse_item_json(serialized.value());
    if (parsed_item.has_error()) {
      return result<response_event>::failure(parsed_item.error());
    }
    parsed.item = std::move(parsed_item.value());
  }

  const auto *part = json_member(root, "part");
  if (part == nullptr) {
    part = json_member(root, "content_part");
  }
  if (part != nullptr && !part->IsNull()) {
    auto serialized = json_to_string(*part);
    if (serialized.has_error()) {
      return result<response_event>::failure(serialized.error());
    }
    auto parsed_part =
        responses::parse_message_content_part_json(serialized.value());
    if (parsed_part.has_error()) {
      return result<response_event>::failure(parsed_part.error());
    }
    parsed.content_part = std::move(parsed_part.value());
  }

  const auto *response = json_member(root, "response");
  if (response != nullptr && !response->IsNull()) {
    auto serialized = json_to_string(*response);
    if (serialized.has_error()) {
      return result<response_event>::failure(serialized.error());
    }
    auto parsed_response = responses::parse_response(serialized.value());
    if (parsed_response.has_error()) {
      return result<response_event>::failure(parsed_response.error());
    }
    parsed.response = std::move(parsed_response.value());
  }

  const auto *error = json_member(root, "error");
  if (error != nullptr && !error->IsNull()) {
    auto serialized = json_to_string(*error);
    if (serialized.has_error()) {
      return result<response_event>::failure(serialized.error());
    }
    auto parsed_error = responses::parse_response_error_json(serialized.value());
    if (parsed_error.has_error()) {
      return result<response_event>::failure(parsed_error.error());
    }
    parsed.error = std::move(parsed_error.value());
  }

  if (!parsed.response_id.has_value() && parsed.response.has_value() &&
      parsed.response->id.has_value()) {
    parsed.response_id = parsed.response->id;
  }
  if (!parsed.item_id.has_value() && parsed.item.has_value()) {
    parsed.item_id = item_id_from_variant(*parsed.item);
  }

  for (auto it = root.MemberBegin(); it != root.MemberEnd(); ++it) {
    const auto key = std::string_view{it->name.GetString(),
                                      static_cast<std::size_t>(
                                          it->name.GetStringLength())};
    if (is_known_event_field(key)) {
      continue;
    }
    auto converted = core::json_to_value(it->value);
    if (converted.has_error()) {
      return result<response_event>::failure(converted.error());
    }
    parsed.extra_fields.emplace(std::string{key}, std::move(converted.value()));
  }

  return result<response_event>::success(std::move(parsed));
}

auto parse_response_event(const sse_event &event)
    -> core::result<response_event> {
  if (event.data == "[DONE]") {
    response_event parsed{};
    parsed.type = "[DONE]";
    parsed.raw_json = "[DONE]";
    return result<response_event>::success(std::move(parsed));
  }
  return parse_response_event_json(event.data);
}

auto parse_response_event_stream(std::string_view stream_text)
    -> core::result<std::vector<response_event>> {
  auto frames = parse_sse_stream(stream_text);
  if (frames.has_error()) {
    return result<std::vector<response_event>>::failure(frames.error());
  }

  std::vector<response_event> events;
  events.reserve(frames.value().size());
  for (const auto &frame : frames.value()) {
    auto parsed_event = parse_response_event(frame);
    if (parsed_event.has_error()) {
      return result<std::vector<response_event>>::failure(parsed_event.error());
    }
    events.push_back(std::move(parsed_event.value()));
  }
  return result<std::vector<response_event>>::success(std::move(events));
}

} // namespace openai::streaming
