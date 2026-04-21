#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace openai::core {

enum class errc : std::uint16_t {
  ok = 0U,
  invalid_argument,
  parse_error,
  type_mismatch,
  not_found,
  not_supported,
  internal_error,
};

[[nodiscard]] inline constexpr auto to_string(const errc code) noexcept
    -> std::string_view {
  switch (code) {
  case errc::ok:
    return "ok";
  case errc::invalid_argument:
    return "invalid_argument";
  case errc::parse_error:
    return "parse_error";
  case errc::type_mismatch:
    return "type_mismatch";
  case errc::not_found:
    return "not_found";
  case errc::not_supported:
    return "not_supported";
  case errc::internal_error:
    return "internal_error";
  }

  return "internal_error";
}

class error_code {
public:
  error_code() = default;

  constexpr explicit error_code(const errc code) noexcept : code_(code) {}

  error_code(errc code, std::string message) noexcept
      : code_(code), message_(std::move(message)) {}

  [[nodiscard]] constexpr auto code() const noexcept -> errc { return code_; }

  [[nodiscard]] constexpr auto failed() const noexcept -> bool {
    return code_ != errc::ok;
  }

  [[nodiscard]] constexpr auto ok() const noexcept -> bool {
    return code_ == errc::ok;
  }

  [[nodiscard]] auto message() const noexcept -> const std::string & {
    return message_;
  }

  [[nodiscard]] auto describe() const -> std::string {
    if (message_.empty()) {
      return std::string{to_string(code_)};
    }
    return std::string{to_string(code_)} + ": " + message_;
  }

private:
  errc code_{errc::ok};
  std::string message_{};
};

[[nodiscard]] inline auto make_error(errc code, std::string message = {})
    -> error_code {
  return error_code(code, std::move(message));
}

template <typename char_t, typename traits_t>
auto operator<<(std::basic_ostream<char_t, traits_t> &stream,
                const error_code &error)
    -> std::basic_ostream<char_t, traits_t> & {
  stream << error.describe();
  return stream;
}

} // namespace openai::core
