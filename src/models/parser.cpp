#include "openai/models/parser.hpp"

#include "openai/core/json.hpp"

namespace openai::models {
namespace {

using core::errc;
using core::json_member;
using core::json_to_string;
using core::make_error;
using core::parse_json;
using core::result;

[[nodiscard]] auto parse_model_value(const core::json_value &value)
    -> result<model> {
  if (!value.IsObject()) {
    return result<model>::failure(
        make_error(errc::type_mismatch, "model must be an object"));
  }
  model parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<model>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());
  if (auto field = core::required_string(value, "id"); field.has_error()) {
    return result<model>::failure(field.error());
  } else {
    parsed.id = std::move(field.value());
  }
  if (auto field = core::required_string(value, "object"); field.has_error()) {
    return result<model>::failure(field.error());
  } else {
    parsed.object = std::move(field.value());
  }
  if (auto field = core::optional_uint64(value, "created"); field.has_error()) {
    return result<model>::failure(field.error());
  } else {
    parsed.created = field.value().value_or(0);
  }
  if (auto field = core::required_string(value, "owned_by"); field.has_error()) {
    return result<model>::failure(field.error());
  } else {
    parsed.owned_by = std::move(field.value());
  }
  return result<model>::success(std::move(parsed));
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

} // namespace

auto parse_model(std::string_view json_text) -> core::result<model> {
  return parse_single_object<model>(json_text, parse_model_value);
}

auto parse_list_response(std::string_view json_text)
    -> core::result<list_response> {
  return parse_single_object<list_response>(json_text, [](const core::json_value &root) {
    if (!root.IsObject()) {
      return result<list_response>::failure(make_error(
          errc::type_mismatch, "model list response must be an object"));
    }
    list_response parsed{};
    auto raw = json_to_string(root);
    if (raw.has_error()) {
      return result<list_response>::failure(raw.error());
    }
    parsed.raw_json = std::move(raw.value());
    if (auto field = core::required_string(root, "object"); field.has_error()) {
      return result<list_response>::failure(field.error());
    } else {
      parsed.object = std::move(field.value());
    }
    const auto *data = json_member(root, "data");
    if (data == nullptr || !data->IsArray()) {
      return result<list_response>::failure(make_error(
          errc::type_mismatch, "model list data must be an array"));
    }
    parsed.data.reserve(data->Size());
    for (const auto &entry : data->GetArray()) {
      auto parsed_model = parse_model_value(entry);
      if (parsed_model.has_error()) {
        return result<list_response>::failure(parsed_model.error());
      }
      parsed.data.push_back(std::move(parsed_model.value()));
    }
    return result<list_response>::success(std::move(parsed));
  });
}

} // namespace openai::models
