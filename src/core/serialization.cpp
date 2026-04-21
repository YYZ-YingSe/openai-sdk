#include "openai/core/serialization.hpp"

#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "openai/core/error.hpp"

namespace openai::core {
namespace {

template <typename writer_t>
auto write_value(writer_t &writer, const value &current) -> bool {
  if (current.is_null()) {
    return writer.Null();
  }
  if (current.is_bool()) {
    return writer.Bool(current.as_bool());
  }
  if (current.is_int()) {
    return writer.Int64(current.as_int());
  }
  if (current.is_double()) {
    return writer.Double(current.as_double());
  }
  if (current.is_string()) {
    const auto &text = current.as_string();
    return writer.String(text.data(),
                         static_cast<rapidjson::SizeType>(text.size()));
  }
  if (current.is_array()) {
    if (!writer.StartArray()) {
      return false;
    }
    for (const auto &entry : current.as_array()) {
      if (!write_value(writer, entry)) {
        return false;
      }
    }
    return writer.EndArray();
  }
  if (!writer.StartObject()) {
    return false;
  }
  for (const auto &[key, entry] : current.as_object()) {
    if (!writer.Key(key.data(), static_cast<rapidjson::SizeType>(key.size()))) {
      return false;
    }
    if (!write_value(writer, entry)) {
      return false;
    }
  }
  return writer.EndObject();
}

template <typename writer_t>
auto serialize_with_writer(const value &current) -> result<std::string> {
  rapidjson::StringBuffer buffer;
  writer_t writer(buffer);
  if (!write_value(writer, current)) {
    return result<std::string>::failure(
        make_error(errc::internal_error, "failed to serialize JSON value"));
  }
  return result<std::string>::success(
      std::string{buffer.GetString(), buffer.GetSize()});
}

} // namespace

auto serialize_json(const value &current, const serialize_options &options)
    -> result<std::string> {
  if (options.pretty) {
    return serialize_with_writer<rapidjson::PrettyWriter<rapidjson::StringBuffer>>(
        current);
  }
  return serialize_with_writer<rapidjson::Writer<rapidjson::StringBuffer>>(
      current);
}

} // namespace openai::core
