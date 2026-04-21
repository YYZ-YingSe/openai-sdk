#include "openai/models/serializer.hpp"

#include <limits>
#include <utility>

#include "openai/core/error.hpp"

namespace openai::models {
namespace {

using core::errc;
using core::make_error;
using core::result;
using object_t = core::value::object;
using array_t = core::value::array;

auto insert_value(object_t &object, std::string_view key, core::value value)
    -> void {
  object.emplace(std::string{key}, std::move(value));
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

auto serialize_model_value(const model &value) -> result<core::value> {
  object_t object{};
  insert_value(object, "id", core::value{value.id});
  insert_value(object, "object", core::value{value.object});
  auto created = uint64_to_value(value.created);
  if (created.has_error()) {
    return result<core::value>::failure(created.error());
  }
  insert_value(object, "created", std::move(created.value()));
  insert_value(object, "owned_by", core::value{value.owned_by});
  return result<core::value>::success(core::value{std::move(object)});
}

} // namespace

auto serialize_model(const model &value, const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_model_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_list_response(const list_response &value,
                             const core::serialize_options &options)
    -> core::result<std::string> {
  object_t object{};
  insert_value(object, "object", core::value{value.object});
  array_t data{};
  data.reserve(value.data.size());
  for (const auto &entry : value.data) {
    auto serialized = serialize_model_value(entry);
    if (serialized.has_error()) {
      return core::result<std::string>::failure(serialized.error());
    }
    data.push_back(std::move(serialized.value()));
  }
  insert_value(object, "data", core::value{std::move(data)});
  return core::serialize_json(core::value{std::move(object)}, options);
}

} // namespace openai::models
