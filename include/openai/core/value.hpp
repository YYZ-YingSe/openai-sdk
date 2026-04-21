#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace openai::core {

struct value {
  using array = std::vector<value>;
  using object = std::map<std::string, value, std::less<>>;
  using storage_type =
      std::variant<std::nullptr_t, bool, std::int64_t, double, std::string,
                   array, object>;

  value() : storage(nullptr) {}
  value(std::nullptr_t) : storage(nullptr) {}
  value(bool current) : storage(current) {}
  value(std::int64_t current) : storage(current) {}
  value(int current) : storage(static_cast<std::int64_t>(current)) {}
  value(double current) : storage(current) {}
  value(std::string current) : storage(std::move(current)) {}
  value(const char *current) : storage(std::string{current}) {}
  value(array current) : storage(std::move(current)) {}
  value(object current) : storage(std::move(current)) {}

  [[nodiscard]] auto is_null() const noexcept -> bool {
    return std::holds_alternative<std::nullptr_t>(storage);
  }

  [[nodiscard]] auto is_bool() const noexcept -> bool {
    return std::holds_alternative<bool>(storage);
  }

  [[nodiscard]] auto is_int() const noexcept -> bool {
    return std::holds_alternative<std::int64_t>(storage);
  }

  [[nodiscard]] auto is_double() const noexcept -> bool {
    return std::holds_alternative<double>(storage);
  }

  [[nodiscard]] auto is_string() const noexcept -> bool {
    return std::holds_alternative<std::string>(storage);
  }

  [[nodiscard]] auto is_array() const noexcept -> bool {
    return std::holds_alternative<array>(storage);
  }

  [[nodiscard]] auto is_object() const noexcept -> bool {
    return std::holds_alternative<object>(storage);
  }

  [[nodiscard]] auto as_bool() const -> const bool & {
    return std::get<bool>(storage);
  }

  [[nodiscard]] auto as_int() const -> const std::int64_t & {
    return std::get<std::int64_t>(storage);
  }

  [[nodiscard]] auto as_double() const -> const double & {
    return std::get<double>(storage);
  }

  [[nodiscard]] auto as_string() const -> const std::string & {
    return std::get<std::string>(storage);
  }

  [[nodiscard]] auto as_array() const -> const array & {
    return std::get<array>(storage);
  }

  [[nodiscard]] auto as_array() -> array & { return std::get<array>(storage); }

  [[nodiscard]] auto as_object() const -> const object & {
    return std::get<object>(storage);
  }

  [[nodiscard]] auto as_object() -> object & { return std::get<object>(storage); }

  storage_type storage;
};

[[nodiscard]] inline auto object_find(const value::object &current,
                                      std::string_view key) noexcept
    -> const value * {
  const auto it = current.find(key);
  if (it == current.end()) {
    return nullptr;
  }
  return &it->second;
}

} // namespace openai::core
