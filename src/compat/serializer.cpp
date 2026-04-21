#include "openai/compat/serializer.hpp"

#include <utility>

#include "openai/chat_completions/serializer.hpp"
#include "openai/completions/serializer.hpp"
#include "openai/core/json.hpp"
#include "openai/embeddings/serializer.hpp"
#include "openai/models/serializer.hpp"
#include "openai/responses/serializer.hpp"
#include "openai/streaming/serializer.hpp"

namespace openai::compat {
namespace {

using core::parse_json;
using core::result;
using object_t = core::value::object;
using array_t = core::value::array;

auto parse_json_value(const core::result<std::string> &json)
    -> result<core::value> {
  if (json.has_error()) {
    return result<core::value>::failure(json.error());
  }
  auto document = parse_json(json.value());
  if (document.has_error()) {
    return result<core::value>::failure(document.error());
  }
  return core::json_to_value(document.value());
}

auto serialize_error_envelope_value(const error_envelope &value)
    -> result<core::value> {
  object_t object{};
  auto error = parse_json_value(responses::serialize_response_error(value.error));
  if (error.has_error()) {
    return result<core::value>::failure(error.error());
  }
  object.emplace("error", std::move(error.value()));
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_list_envelope_value(const list_envelope &value)
    -> result<core::value> {
  object_t object{};
  if (value.object.has_value()) {
    object.emplace("object", core::value{*value.object});
  }
  array_t data{};
  data.reserve(value.data.size());
  for (const auto &entry : value.data) {
    data.push_back(entry);
  }
  object.emplace("data", core::value{std::move(data)});
  if (value.has_more.has_value()) {
    object.emplace("has_more", core::value{*value.has_more});
  }
  if (value.first_id.has_value()) {
    object.emplace("first_id", core::value{*value.first_id});
  }
  if (value.last_id.has_value()) {
    object.emplace("last_id", core::value{*value.last_id});
  }
  if (value.next.has_value()) {
    object.emplace("next", core::value{*value.next});
  }
  return result<core::value>::success(core::value{std::move(object)});
}

} // namespace

auto serialize_request_json(const request_document &document,
                            const operation_hint &hint,
                            const core::serialize_options &options)
    -> core::result<std::string> {
  (void)hint;
  return std::visit(
      [&](const auto &current) -> core::result<std::string> {
        using current_t = std::decay_t<decltype(current)>;
        if constexpr (std::is_same_v<current_t, responses::request>) {
          return responses::serialize_request(current, options);
        } else if constexpr (std::is_same_v<current_t,
                                            chat_completions::request>) {
          return chat_completions::serialize_request(current, options);
        } else if constexpr (std::is_same_v<current_t, completions::request>) {
          return completions::serialize_request(current, options);
        } else if constexpr (std::is_same_v<current_t, embeddings::request>) {
          return embeddings::serialize_request(current, options);
        } else {
          return core::serialize_json(current.data, options);
        }
      },
      document);
}

auto serialize_response_json(const response_document &document,
                             const operation_hint &hint,
                             const core::serialize_options &options)
    -> core::result<std::string> {
  (void)hint;
  return std::visit(
      [&](const auto &current) -> core::result<std::string> {
        using current_t = std::decay_t<decltype(current)>;
        if constexpr (std::is_same_v<current_t, responses::response>) {
          return responses::serialize_response(current, options);
        } else if constexpr (std::is_same_v<current_t,
                                            streaming::response_event>) {
          return streaming::serialize_response_event_json(current, options);
        } else if constexpr (std::is_same_v<current_t,
                                            chat_completions::completion>) {
          return chat_completions::serialize_completion(current, options);
        } else if constexpr (std::is_same_v<
                                 current_t,
                                 chat_completions::completion_chunk>) {
          return chat_completions::serialize_completion_chunk(current, options);
        } else if constexpr (std::is_same_v<current_t, completions::response>) {
          return completions::serialize_response(current, options);
        } else if constexpr (std::is_same_v<current_t, completions::chunk>) {
          return completions::serialize_chunk(current, options);
        } else if constexpr (std::is_same_v<current_t, embeddings::response>) {
          return embeddings::serialize_response(current, options);
        } else if constexpr (std::is_same_v<current_t, models::model>) {
          return models::serialize_model(current, options);
        } else if constexpr (std::is_same_v<current_t, models::list_response>) {
          return models::serialize_list_response(current, options);
        } else if constexpr (std::is_same_v<current_t, error_envelope>) {
          auto serialized = serialize_error_envelope_value(current);
          if (serialized.has_error()) {
            return core::result<std::string>::failure(serialized.error());
          }
          return core::serialize_json(serialized.value(), options);
        } else if constexpr (std::is_same_v<current_t, list_envelope>) {
          auto serialized = serialize_list_envelope_value(current);
          if (serialized.has_error()) {
            return core::result<std::string>::failure(serialized.error());
          }
          return core::serialize_json(serialized.value(), options);
        } else {
          return core::serialize_json(current.data, options);
        }
      },
      document);
}

auto serialize_document_json(const document &document,
                             const operation_hint &hint,
                             const core::serialize_options &options)
    -> core::result<std::string> {
  return std::visit(
      [&](const auto &current) -> core::result<std::string> {
        using current_t = std::decay_t<decltype(current)>;
        if constexpr (std::is_same_v<current_t, request_document>) {
          return serialize_request_json(current, hint, options);
        } else {
          return serialize_response_json(current, hint, options);
        }
      },
      document);
}

auto validate_request(const request_document &document,
                      const capabilities::effective_profile &profile,
                      const operation_hint &hint)
    -> capabilities::validation_report {
  return capabilities::validate_request(document, profile, hint);
}

} // namespace openai::compat
