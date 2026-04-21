#include "openai/embeddings/serializer.hpp"

#include <type_traits>
#include <utility>

namespace openai::embeddings {
namespace {

using core::result;
using object_t = core::value::object;
using array_t = core::value::array;

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

auto serialize_input_value(const embedding_input &value)
    -> result<core::value> {
  return std::visit(
      []<typename input_t>(const input_t &current) -> result<core::value> {
        using current_t = std::decay_t<input_t>;
        if constexpr (std::is_same_v<current_t, std::string>) {
          return result<core::value>::success(core::value{current});
        } else if constexpr (std::is_same_v<current_t, std::vector<std::string>>) {
          array_t values{};
          values.reserve(current.size());
          for (const auto &entry : current) {
            values.emplace_back(entry);
          }
          return result<core::value>::success(core::value{std::move(values)});
        } else if constexpr (std::is_same_v<current_t, std::vector<std::int64_t>>) {
          array_t values{};
          values.reserve(current.size());
          for (const auto entry : current) {
            values.emplace_back(entry);
          }
          return result<core::value>::success(core::value{std::move(values)});
        } else {
          array_t batches{};
          batches.reserve(current.size());
          for (const auto &batch : current) {
            array_t values{};
            values.reserve(batch.size());
            for (const auto entry : batch) {
              values.emplace_back(entry);
            }
            batches.emplace_back(std::move(values));
          }
          return result<core::value>::success(core::value{std::move(batches)});
        }
      },
      value);
}

auto serialize_embedding_value(const embedding &value) -> result<core::value> {
  object_t object{};
  insert_value(object, "object", core::value{value.object});
  insert_value(object, "index", core::value{value.index});
  std::visit(
      [&object](const auto &current) {
        using current_t = std::decay_t<decltype(current)>;
        if constexpr (std::is_same_v<current_t, std::string>) {
          insert_value(object, "embedding", core::value{current});
        } else {
          array_t values{};
          values.reserve(current.size());
          for (const auto entry : current) {
            values.emplace_back(entry);
          }
          insert_value(object, "embedding", core::value{std::move(values)});
        }
      },
      value.values);
  return result<core::value>::success(core::value{std::move(object)});
}

} // namespace

auto serialize_request(const request &value,
                       const core::serialize_options &options)
    -> core::result<std::string> {
  object_t object{};
  auto input = serialize_input_value(value.input);
  if (input.has_error()) {
    return core::result<std::string>::failure(input.error());
  }
  insert_value(object, "input", std::move(input.value()));
  insert_value(object, "model", core::value{value.model});
  insert_optional_int64(object, "dimensions", value.dimensions);
  insert_optional_string(object, "encoding_format", value.encoding_format);
  insert_optional_string(object, "user", value.user);
  return core::serialize_json(core::value{std::move(object)}, options);
}

auto serialize_response(const response &value,
                        const core::serialize_options &options)
    -> core::result<std::string> {
  object_t object{};
  insert_value(object, "object", core::value{value.object});
  insert_value(object, "model", core::value{value.model});
  array_t data{};
  data.reserve(value.data.size());
  for (const auto &entry : value.data) {
    auto serialized = serialize_embedding_value(entry);
    if (serialized.has_error()) {
      return core::result<std::string>::failure(serialized.error());
    }
    data.push_back(std::move(serialized.value()));
  }
  insert_value(object, "data", core::value{std::move(data)});
  object_t usage{};
  insert_value(usage, "prompt_tokens", core::value{value.usage_value.prompt_tokens});
  insert_value(usage, "total_tokens", core::value{value.usage_value.total_tokens});
  insert_value(object, "usage", core::value{std::move(usage)});
  return core::serialize_json(core::value{std::move(object)}, options);
}

} // namespace openai::embeddings
