#include "openai/compat/parser.hpp"

#include "openai/chat_completions/parser.hpp"
#include "openai/completions/parser.hpp"
#include "openai/core/json.hpp"
#include "openai/embeddings/parser.hpp"
#include "openai/models/parser.hpp"
#include "openai/responses/parser.hpp"
#include "openai/streaming/response_events.hpp"

namespace openai::compat {
namespace {

using core::errc;
using core::json_member;
using core::json_to_string;
using core::make_error;
using core::parse_json;
using core::result;

[[nodiscard]] auto parse_generic_object_value(const core::json_value &value)
    -> result<generic_object> {
  if (!value.IsObject()) {
    return result<generic_object>::failure(make_error(
        errc::type_mismatch, "generic object must be an object"));
  }
  generic_object parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<generic_object>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());
  if (auto field = core::optional_string(value, "id"); field.has_error()) {
    return result<generic_object>::failure(field.error());
  } else {
    parsed.id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "object"); field.has_error()) {
    return result<generic_object>::failure(field.error());
  } else {
    parsed.object = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "type"); field.has_error()) {
    return result<generic_object>::failure(field.error());
  } else {
    parsed.type = std::move(field.value());
  }
  auto payload = core::json_to_value(value);
  if (payload.has_error()) {
    return result<generic_object>::failure(payload.error());
  }
  parsed.data = std::move(payload.value());
  return result<generic_object>::success(std::move(parsed));
}

[[nodiscard]] auto parse_list_envelope_value(const core::json_value &value)
    -> result<list_envelope> {
  if (!value.IsObject()) {
    return result<list_envelope>::failure(
        make_error(errc::type_mismatch, "list envelope must be an object"));
  }
  list_envelope parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<list_envelope>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());
  if (auto field = core::optional_string(value, "object"); field.has_error()) {
    return result<list_envelope>::failure(field.error());
  } else {
    parsed.object = std::move(field.value());
  }
  if (auto field = core::optional_bool(value, "has_more"); field.has_error()) {
    return result<list_envelope>::failure(field.error());
  } else {
    parsed.has_more = field.value();
  }
  if (auto field = core::optional_string(value, "first_id"); field.has_error()) {
    return result<list_envelope>::failure(field.error());
  } else {
    parsed.first_id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "last_id"); field.has_error()) {
    return result<list_envelope>::failure(field.error());
  } else {
    parsed.last_id = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "next"); field.has_error()) {
    return result<list_envelope>::failure(field.error());
  } else {
    parsed.next = std::move(field.value());
  }
  const auto *data = json_member(value, "data");
  if (data == nullptr || !data->IsArray()) {
    return result<list_envelope>::failure(
        make_error(errc::type_mismatch, "list envelope data must be an array"));
  }
  parsed.data.reserve(data->Size());
  for (const auto &entry : data->GetArray()) {
    auto converted = core::json_to_value(entry);
    if (converted.has_error()) {
      return result<list_envelope>::failure(converted.error());
    }
    parsed.data.push_back(std::move(converted.value()));
  }
  return result<list_envelope>::success(std::move(parsed));
}

template <typename output_t>
[[nodiscard]] auto parse_single_object(std::string_view json_text,
                                       auto &&parser) -> result<output_t> {
  auto document = parse_json(json_text);
  if (document.has_error()) {
    return result<output_t>::failure(document.error());
  }
  return parser(document.value());
}

[[nodiscard]] auto parse_error_envelope(std::string_view json_text,
                                        const responses::parse_options &options)
    -> result<error_envelope> {
  auto document = parse_json(json_text);
  if (document.has_error()) {
    return result<error_envelope>::failure(document.error());
  }
  const auto &root = document.value();
  const auto *error = json_member(root, "error");
  if (error == nullptr) {
    return result<error_envelope>::failure(
        make_error(errc::not_found, "missing required field 'error'"));
  }
  auto serialized = json_to_string(*error);
  if (serialized.has_error()) {
    return result<error_envelope>::failure(serialized.error());
  }
  auto parsed_error = responses::parse_response_error_json(serialized.value(), options);
  if (parsed_error.has_error()) {
    return result<error_envelope>::failure(parsed_error.error());
  }
  error_envelope parsed{};
  parsed.error = std::move(parsed_error.value());
  auto raw = json_to_string(root);
  if (raw.has_error()) {
    return result<error_envelope>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());
  return result<error_envelope>::success(std::move(parsed));
}

} // namespace

auto parse_request_json(std::string_view json_text, const operation_hint &hint,
                        const responses::parse_options &options)
    -> core::result<request_document> {
  switch (hint.family) {
  case api_family::responses: {
    auto parsed = responses::parse_request(json_text, options);
    if (parsed.has_error()) {
      return result<request_document>::failure(parsed.error());
    }
    return result<request_document>::success(request_document{std::move(parsed.value())});
  }
  case api_family::chat_completions: {
    auto parsed = chat_completions::parse_request(json_text, options);
    if (parsed.has_error()) {
      return result<request_document>::failure(parsed.error());
    }
    return result<request_document>::success(request_document{std::move(parsed.value())});
  }
  case api_family::completions: {
    auto parsed = completions::parse_request(json_text);
    if (parsed.has_error()) {
      return result<request_document>::failure(parsed.error());
    }
    return result<request_document>::success(request_document{std::move(parsed.value())});
  }
  case api_family::embeddings: {
    auto parsed = embeddings::parse_request(json_text);
    if (parsed.has_error()) {
      return result<request_document>::failure(parsed.error());
    }
    return result<request_document>::success(request_document{std::move(parsed.value())});
  }
  case api_family::models:
  case api_family::unknown:
    break;
  }

  auto parsed = parse_single_object<generic_object>(json_text, parse_generic_object_value);
  if (parsed.has_error()) {
    return result<request_document>::failure(parsed.error());
  }
  return result<request_document>::success(request_document{std::move(parsed.value())});
}

auto parse_response_json(std::string_view json_text, const operation_hint &hint,
                         const responses::parse_options &options)
    -> core::result<response_document> {
  switch (hint.family) {
  case api_family::responses:
    if (hint.streaming) {
      auto parsed = streaming::parse_response_event_json(json_text);
      if (parsed.has_error()) {
        return result<response_document>::failure(parsed.error());
      }
      return result<response_document>::success(response_document{std::move(parsed.value())});
    } else {
      auto parsed = responses::parse_response(json_text, options);
      if (parsed.has_error()) {
        return result<response_document>::failure(parsed.error());
      }
      return result<response_document>::success(response_document{std::move(parsed.value())});
    }
  case api_family::chat_completions:
    if (hint.streaming) {
      auto parsed = chat_completions::parse_completion_chunk(json_text, options);
      if (parsed.has_error()) {
        return result<response_document>::failure(parsed.error());
      }
      return result<response_document>::success(response_document{std::move(parsed.value())});
    } else {
      auto parsed = chat_completions::parse_completion(json_text, options);
      if (parsed.has_error()) {
        return result<response_document>::failure(parsed.error());
      }
      return result<response_document>::success(response_document{std::move(parsed.value())});
    }
  case api_family::completions:
    if (hint.streaming) {
      auto parsed = completions::parse_chunk(json_text);
      if (parsed.has_error()) {
        return result<response_document>::failure(parsed.error());
      }
      return result<response_document>::success(response_document{std::move(parsed.value())});
    } else {
      auto parsed = completions::parse_response(json_text);
      if (parsed.has_error()) {
        return result<response_document>::failure(parsed.error());
      }
      return result<response_document>::success(response_document{std::move(parsed.value())});
    }
  case api_family::embeddings: {
    auto parsed = embeddings::parse_response(json_text);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }
  case api_family::models: {
    if (hint.operation == "list") {
      auto parsed = models::parse_list_response(json_text);
      if (parsed.has_error()) {
        return result<response_document>::failure(parsed.error());
      }
      return result<response_document>::success(response_document{std::move(parsed.value())});
    }
    auto parsed = models::parse_model(json_text);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }
  case api_family::unknown:
    break;
  }

  return parse_document_json(json_text, options);
}

auto parse_document_json(std::string_view json_text,
                         const responses::parse_options &options)
    -> core::result<response_document> {
  auto document = parse_json(json_text);
  if (document.has_error()) {
    return result<response_document>::failure(document.error());
  }
  const auto &root = document.value();
  if (!root.IsObject()) {
    return result<response_document>::failure(
        make_error(errc::type_mismatch, "document must be an object"));
  }

  if (json_member(root, "error") != nullptr) {
    auto parsed = parse_error_envelope(json_text, options);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }

  auto object = core::optional_string(root, "object");
  if (object.has_error()) {
    return result<response_document>::failure(object.error());
  }
  const auto object_value = object.value().value_or("");

  if (object_value == "response") {
    auto parsed = responses::parse_response(json_text, options);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }
  if (object_value == "chat.completion") {
    auto parsed = chat_completions::parse_completion(json_text, options);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }
  if (object_value == "chat.completion.chunk") {
    auto parsed = chat_completions::parse_completion_chunk(json_text, options);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }
  if (object_value == "text_completion") {
    auto parsed = completions::parse_response(json_text);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }
  if (object_value == "model") {
    auto parsed = models::parse_model(json_text);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }
  if (object_value == "list") {
    const auto *data = json_member(root, "data");
    if (data != nullptr && data->IsArray() && !data->Empty() && data->GetArray()[0].IsObject()) {
      auto first_object = core::optional_string(data->GetArray()[0], "object");
      if (first_object.has_error()) {
        return result<response_document>::failure(first_object.error());
      }
      const auto first_object_value = first_object.value().value_or("");
      if (first_object_value == "model") {
        auto parsed = models::parse_list_response(json_text);
        if (parsed.has_value()) {
          return result<response_document>::success(response_document{std::move(parsed.value())});
        }
      }
      if (first_object_value == "embedding") {
        auto parsed = embeddings::parse_response(json_text);
        if (parsed.has_value()) {
          return result<response_document>::success(response_document{std::move(parsed.value())});
        }
      }
    }
    auto parsed = parse_list_envelope_value(root);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }

  if (json_member(root, "type") != nullptr && json_member(root, "response") != nullptr) {
    auto parsed = streaming::parse_response_event_json(json_text);
    if (parsed.has_error()) {
      return result<response_document>::failure(parsed.error());
    }
    return result<response_document>::success(response_document{std::move(parsed.value())});
  }

  if (json_member(root, "choices") != nullptr && json_member(root, "prompt") == nullptr &&
      json_member(root, "messages") == nullptr &&
      (object_value.empty() || object_value == "text_completion")) {
    auto parsed = completions::parse_response(json_text);
    if (parsed.has_value()) {
      return result<response_document>::success(response_document{std::move(parsed.value())});
    }
  }

  auto parsed = parse_generic_object_value(root);
  if (parsed.has_error()) {
    return result<response_document>::failure(parsed.error());
  }
  return result<response_document>::success(response_document{std::move(parsed.value())});
}

} // namespace openai::compat
