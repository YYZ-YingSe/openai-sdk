#include <exception>
#include <iostream>

#include "test_support.hpp"

auto main() -> int {
  int failures = 0;
  for (const auto &test : openai::test::registry()) {
    try {
      test.function();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
    } catch (...) {
      ++failures;
      std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
    }
  }

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
  }
  return failures == 0 ? 0 : 1;
}
