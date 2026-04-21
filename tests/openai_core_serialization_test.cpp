#include <string>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("serialize core::value scalars arrays and objects deterministically") {
  using openai::core::value;

  value::object root{
      {"input", value{value::array{value{"hello"}, value{"world"}}}},
      {"max_tokens", value{64}},
      {"model", value{"gpt-4.1"}},
      {"stream", value{true}},
  };

  auto json = openai::core::serialize_json(value{std::move(root)});
  OAI_REQUIRE(json.has_value());
  OAI_REQUIRE(json.value() ==
              "{\"input\":[\"hello\",\"world\"],\"max_tokens\":64,"
              "\"model\":\"gpt-4.1\",\"stream\":true}");
}

} // namespace
