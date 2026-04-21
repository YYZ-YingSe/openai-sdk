#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include "openai/core/error.hpp"

namespace openai::core {

template <typename value_t, typename error_t = error_code> class result {
public:
  using value_type = value_t;
  using error_type = error_t;

  result(const value_t &value) : storage_(value) {}
  result(value_t &&value) : storage_(std::move(value)) {}
  result(const error_t &error) : storage_(error) {}
  result(error_t &&error) : storage_(std::move(error)) {}

  [[nodiscard]] static auto success(value_t value) -> result {
    return result(std::move(value));
  }

  [[nodiscard]] static auto failure(error_t error) -> result {
    return result(std::move(error));
  }

  [[nodiscard]] auto has_value() const noexcept -> bool {
    return std::holds_alternative<value_t>(storage_);
  }

  [[nodiscard]] auto has_error() const noexcept -> bool { return !has_value(); }

  [[nodiscard]] auto value() & -> value_t & {
    return std::get<value_t>(storage_);
  }

  [[nodiscard]] auto value() const & -> const value_t & {
    return std::get<value_t>(storage_);
  }

  [[nodiscard]] auto value() && -> value_t && {
    return std::get<value_t>(std::move(storage_));
  }

  [[nodiscard]] auto error() & -> error_t & {
    return std::get<error_t>(storage_);
  }

  [[nodiscard]] auto error() const & -> const error_t & {
    return std::get<error_t>(storage_);
  }

private:
  std::variant<value_t, error_t> storage_;
};

template <typename error_t> class result<void, error_t> {
public:
  using value_type = void;
  using error_type = error_t;

  result() = default;
  result(const error_t &error) : error_(error), has_error_(true) {}
  result(error_t &&error) : error_(std::move(error)), has_error_(true) {}

  [[nodiscard]] static auto success() -> result { return result(); }

  [[nodiscard]] static auto failure(error_t error) -> result {
    return result(std::move(error));
  }

  [[nodiscard]] auto has_value() const noexcept -> bool { return !has_error_; }

  [[nodiscard]] auto has_error() const noexcept -> bool { return has_error_; }

  [[nodiscard]] auto error() & -> error_t & {
    if (!has_error_) {
      throw std::logic_error("result<void> does not contain an error");
    }
    return error_;
  }

  [[nodiscard]] auto error() const & -> const error_t & {
    if (!has_error_) {
      throw std::logic_error("result<void> does not contain an error");
    }
    return error_;
  }

private:
  error_t error_{};
  bool has_error_{false};
};

} // namespace openai::core
