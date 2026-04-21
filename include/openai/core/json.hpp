#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "openai/core/error.hpp"
#include "openai/core/result.hpp"
#include "openai/core/string_view.hpp"
#include "openai/core/value.hpp"

namespace openai::core {

using json_document = rapidjson::Document;
using json_value = rapidjson::Value;

[[nodiscard]] inline auto json_string(const json_value &value) -> std::string {
  return std::string{value.GetString(),
                     static_cast<std::size_t>(value.GetStringLength())};
}

[[nodiscard]] inline auto parse_json(std::string_view text)
    -> result<json_document> {
  json_document document;
  document.Parse(text.data(), text.size());
  if (document.HasParseError()) {
    return result<json_document>::failure(make_error(
        errc::parse_error,
        std::string{rapidjson::GetParseError_En(document.GetParseError())} +
            " at byte " + std::to_string(document.GetErrorOffset())));
  }
  return result<json_document>::success(std::move(document));
}

[[nodiscard]] inline auto json_to_string(const json_value &value)
    -> result<std::string> {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  if (!value.Accept(writer)) {
    return result<std::string>::failure(
        make_error(errc::internal_error, "failed to serialize JSON value"));
  }
  return result<std::string>::success(
      std::string{buffer.GetString(), buffer.GetSize()});
}

[[nodiscard]] inline auto json_to_value(const json_value &value)
    -> result<core::value> {
  if (value.IsNull()) {
    return result<core::value>::success(core::value{nullptr});
  }
  if (value.IsBool()) {
    return result<core::value>::success(core::value{value.GetBool()});
  }
  if (value.IsInt64()) {
    return result<core::value>::success(core::value{value.GetInt64()});
  }
  if (value.IsNumber()) {
    return result<core::value>::success(core::value{value.GetDouble()});
  }
  if (value.IsString()) {
    return result<core::value>::success(core::value{json_string(value)});
  }
  if (value.IsArray()) {
    core::value::array array_values;
    array_values.reserve(value.Size());
    for (const auto &entry : value.GetArray()) {
      auto converted = json_to_value(entry);
      if (converted.has_error()) {
        return result<core::value>::failure(converted.error());
      }
      array_values.push_back(std::move(converted.value()));
    }
    return result<core::value>::success(core::value{std::move(array_values)});
  }
  if (value.IsObject()) {
    core::value::object object_values;
    for (auto it = value.MemberBegin(); it != value.MemberEnd(); ++it) {
      auto converted = json_to_value(it->value);
      if (converted.has_error()) {
        return result<core::value>::failure(converted.error());
      }
      object_values.emplace(json_string(it->name), std::move(converted.value()));
    }
    return result<core::value>::success(core::value{std::move(object_values)});
  }

  return result<core::value>::failure(
      make_error(errc::internal_error, "unsupported JSON value kind"));
}

[[nodiscard]] inline auto json_member(const json_value &value,
                                      std::string_view key) noexcept
    -> const json_value * {
  if (!value.IsObject()) {
    return nullptr;
  }
  const auto it = value.FindMember(rapidjson::StringRef(
      key.data(), static_cast<rapidjson::SizeType>(key.size())));
  if (it == value.MemberEnd()) {
    return nullptr;
  }
  return &it->value;
}

[[nodiscard]] inline auto require_object(const json_value &value,
                                         std::string_view label)
    -> result<const json_value *> {
  if (!value.IsObject()) {
    return result<const json_value *>::failure(make_error(
        errc::type_mismatch, std::string{label} + " must be an object"));
  }
  return result<const json_value *>::success(&value);
}

[[nodiscard]] inline auto require_array(const json_value &value,
                                        std::string_view label)
    -> result<const json_value *> {
  if (!value.IsArray()) {
    return result<const json_value *>::failure(make_error(
        errc::type_mismatch, std::string{label} + " must be an array"));
  }
  return result<const json_value *>::success(&value);
}

[[nodiscard]] inline auto require_member(const json_value &value,
                                         std::string_view key,
                                         std::string_view owner = "object")
    -> result<const json_value *> {
  const auto *member = json_member(value, key);
  if (member == nullptr) {
    return result<const json_value *>::failure(
        make_error(errc::not_found,
                   std::string{owner} + " is missing required field '" +
                       std::string{key} + "'"));
  }
  return result<const json_value *>::success(member);
}

[[nodiscard]] inline auto optional_string(const json_value &value,
                                          std::string_view key)
    -> result<std::optional<std::string>> {
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::optional<std::string>>::success(std::nullopt);
  }
  if (!member->IsString()) {
    return result<std::optional<std::string>>::failure(make_error(
        errc::type_mismatch, std::string{key} + " must be a string"));
  }
  return result<std::optional<std::string>>::success(json_string(*member));
}

[[nodiscard]] inline auto required_string(const json_value &value,
                                          std::string_view key)
    -> result<std::string> {
  auto member = require_member(value, key);
  if (member.has_error()) {
    return result<std::string>::failure(member.error());
  }
  if (!member.value()->IsString()) {
    return result<std::string>::failure(make_error(
        errc::type_mismatch, std::string{key} + " must be a string"));
  }
  return result<std::string>::success(json_string(*member.value()));
}

[[nodiscard]] inline auto optional_bool(const json_value &value,
                                        std::string_view key)
    -> result<std::optional<bool>> {
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::optional<bool>>::success(std::nullopt);
  }
  if (!member->IsBool()) {
    return result<std::optional<bool>>::failure(
        make_error(errc::type_mismatch, std::string{key} + " must be a bool"));
  }
  return result<std::optional<bool>>::success(member->GetBool());
}

[[nodiscard]] inline auto optional_int64(const json_value &value,
                                         std::string_view key)
    -> result<std::optional<std::int64_t>> {
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::optional<std::int64_t>>::success(std::nullopt);
  }
  if (!member->IsInt64()) {
    return result<std::optional<std::int64_t>>::failure(make_error(
        errc::type_mismatch, std::string{key} + " must be an int64"));
  }
  return result<std::optional<std::int64_t>>::success(member->GetInt64());
}

[[nodiscard]] inline auto optional_uint64(const json_value &value,
                                          std::string_view key)
    -> result<std::optional<std::uint64_t>> {
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::optional<std::uint64_t>>::success(std::nullopt);
  }
  if (!member->IsUint64()) {
    return result<std::optional<std::uint64_t>>::failure(make_error(
        errc::type_mismatch, std::string{key} + " must be a uint64"));
  }
  return result<std::optional<std::uint64_t>>::success(member->GetUint64());
}

[[nodiscard]] inline auto optional_double(const json_value &value,
                                          std::string_view key)
    -> result<std::optional<double>> {
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::optional<double>>::success(std::nullopt);
  }
  if (!member->IsNumber()) {
    return result<std::optional<double>>::failure(make_error(
        errc::type_mismatch, std::string{key} + " must be a number"));
  }
  return result<std::optional<double>>::success(member->GetDouble());
}

[[nodiscard]] inline auto optional_string_map(const json_value &value,
                                              std::string_view key)
    -> result<std::map<std::string, std::string>> {
  std::map<std::string, std::string> result_map;
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::map<std::string, std::string>>::success(
        std::move(result_map));
  }
  if (!member->IsObject()) {
    return result<std::map<std::string, std::string>>::failure(make_error(
        errc::type_mismatch, std::string{key} + " must be an object"));
  }
  for (auto it = member->MemberBegin(); it != member->MemberEnd(); ++it) {
    if (!it->value.IsString()) {
      return result<std::map<std::string, std::string>>::failure(make_error(
          errc::type_mismatch,
          std::string{key} + " must only contain string values"));
    }
    result_map.emplace(json_string(it->name), json_string(it->value));
  }
  return result<std::map<std::string, std::string>>::success(
      std::move(result_map));
}

[[nodiscard]] inline auto optional_string_array(const json_value &value,
                                                std::string_view key)
    -> result<std::vector<std::string>> {
  std::vector<std::string> values;
  const auto *member = json_member(value, key);
  if (member == nullptr || member->IsNull()) {
    return result<std::vector<std::string>>::success(std::move(values));
  }
  if (!member->IsArray()) {
    return result<std::vector<std::string>>::failure(make_error(
        errc::type_mismatch, std::string{key} + " must be an array"));
  }
  values.reserve(member->Size());
  for (const auto &entry : member->GetArray()) {
    if (!entry.IsString()) {
      return result<std::vector<std::string>>::failure(make_error(
          errc::type_mismatch,
          std::string{key} + " must only contain string values"));
    }
    values.push_back(json_string(entry));
  }
  return result<std::vector<std::string>>::success(std::move(values));
}

} // namespace openai::core
