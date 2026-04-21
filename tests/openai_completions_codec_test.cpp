#include <string>
#include <variant>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("serialize legacy completions request and response") {
  openai::completions::request request{};
  request.model = "gpt-3.5-turbo-instruct";
  request.prompt = std::string{"hello"};
  request.stream = true;

  auto request_json = openai::completions::serialize_request(request);
  OAI_REQUIRE(request_json.has_value());
  OAI_REQUIRE(request_json.value().find("\"prompt\":\"hello\"") !=
              std::string::npos);

  auto reparsed_request =
      openai::completions::parse_request(request_json.value());
  OAI_REQUIRE(reparsed_request.has_value());
  OAI_REQUIRE(reparsed_request.value().model == request.model);

  openai::completions::response response{};
  response.id = "cmpl_1";
  response.object = "text_completion";
  response.created = 1744281600;
  response.model = "gpt-3.5-turbo-instruct";

  openai::completions::choice choice{};
  choice.index = 0;
  choice.text = "Hello";
  choice.finish_reason = "stop";
  response.choices.push_back(std::move(choice));

  openai::completions::usage usage{};
  usage.completion_tokens = 1;
  usage.prompt_tokens = 1;
  usage.total_tokens = 2;
  response.usage_value = usage;

  auto response_json = openai::completions::serialize_response(response);
  OAI_REQUIRE(response_json.has_value());
  auto reparsed_response =
      openai::completions::parse_response(response_json.value());
  OAI_REQUIRE(reparsed_response.has_value());
  OAI_REQUIRE(reparsed_response.value().choices.size() == 1U);
  OAI_REQUIRE(reparsed_response.value().usage_value.has_value());
}

OAI_TEST("serialize legacy completions chunk with null finish_reason") {
  openai::completions::chunk chunk{};
  chunk.id = "cmpl_1";
  chunk.object = "text_completion";
  chunk.created = 1744281600;
  chunk.model = "gpt-3.5-turbo-instruct";

  openai::completions::choice choice{};
  choice.index = 0;
  choice.text = "Hel";
  chunk.choices.push_back(std::move(choice));

  auto chunk_json = openai::completions::serialize_chunk(chunk);
  OAI_REQUIRE(chunk_json.has_value());
  OAI_REQUIRE(chunk_json.value().find("\"finish_reason\":null") !=
              std::string::npos);

  auto reparsed_chunk = openai::completions::parse_chunk(chunk_json.value());
  OAI_REQUIRE(reparsed_chunk.has_value());
  OAI_REQUIRE(reparsed_chunk.value().choices.size() == 1U);
  OAI_REQUIRE(!reparsed_chunk.value().choices[0].finish_reason.has_value());
}

} // namespace
