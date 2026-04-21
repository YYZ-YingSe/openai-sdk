#pragma once

#include <exception>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace openai::test {

using test_function = std::function<void()>;

struct test_case {
  std::string name;
  test_function function;
};

inline auto registry() -> std::vector<test_case> & {
  static std::vector<test_case> all_tests;
  return all_tests;
}

class registrar {
public:
  registrar(std::string name, test_function function) {
    registry().push_back(
        test_case{std::move(name), std::move(function)});
  }
};

[[noreturn]] inline auto fail(std::string_view expression,
                              std::string_view file, int line,
                              std::string_view message = {}) -> void {
  std::ostringstream stream;
  stream << file << ":" << line << " assertion failed: " << expression;
  if (!message.empty()) {
    stream << " (" << message << ")";
  }
  throw std::runtime_error(stream.str());
}

} // namespace openai::test

#define OAI_CONCAT_IMPL(a, b) a##b
#define OAI_CONCAT(a, b) OAI_CONCAT_IMPL(a, b)

#define OAI_TEST(name)                                                         \
  static void OAI_CONCAT(openai_test_fn_, __LINE__)();                         \
  static ::openai::test::registrar OAI_CONCAT(openai_test_reg_, __LINE__)(     \
      name, OAI_CONCAT(openai_test_fn_, __LINE__));                            \
  static void OAI_CONCAT(openai_test_fn_, __LINE__)()

#define OAI_REQUIRE(expr)                                                      \
  do {                                                                         \
    if (!(expr)) {                                                             \
      ::openai::test::fail(#expr, __FILE__, __LINE__);                         \
    }                                                                          \
  } while (false)

#define OAI_REQUIRE_MSG(expr, message)                                         \
  do {                                                                         \
    if (!(expr)) {                                                             \
      ::openai::test::fail(#expr, __FILE__, __LINE__, message);                \
    }                                                                          \
  } while (false)
