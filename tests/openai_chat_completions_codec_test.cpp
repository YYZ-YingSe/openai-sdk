#include <string>
#include <variant>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("serialize chat completions request with messages and tools") {
  openai::chat_completions::request request{};
  request.model = "gpt-4o-mini";
  request.stream = true;

  openai::chat_completions::message message{};
  message.role = "user";
  message.content = std::string{"hello"};
  request.messages.push_back(std::move(message));

  openai::responses::function_tool tool{};
  tool.name = "lookup_weather";
  tool.parameters_json = R"json({"type":"object"})json";
  request.tools.push_back(tool);

  request.parallel_tool_calls = true;

  auto json = openai::chat_completions::serialize_request(request);
  OAI_REQUIRE(json.has_value());
  OAI_REQUIRE(json.value().find("\"messages\"") != std::string::npos);
  OAI_REQUIRE(json.value().find("\"stream\":true") != std::string::npos);

  auto reparsed = openai::chat_completions::parse_request(json.value());
  OAI_REQUIRE(reparsed.has_value());
  OAI_REQUIRE(reparsed.value().model.has_value());
  OAI_REQUIRE(reparsed.value().model.value() == "gpt-4o-mini");
  OAI_REQUIRE(reparsed.value().messages.size() == 1U);
}

OAI_TEST("serialize chat completion response and chunk") {
  openai::chat_completions::completion completion{};
  completion.id = "chatcmpl_1";
  completion.object = "chat.completion";
  completion.model = "gpt-4o-mini";
  completion.created = 1744281600;

  openai::chat_completions::choice choice{};
  choice.index = 0;
  choice.finish_reason = "stop";
  openai::chat_completions::message message{};
  message.role = "assistant";
  message.content = std::string{"Hello"};
  choice.message_value = std::move(message);
  completion.choices.push_back(std::move(choice));

  auto completion_json =
      openai::chat_completions::serialize_completion(completion);
  OAI_REQUIRE(completion_json.has_value());
  auto reparsed_completion =
      openai::chat_completions::parse_completion(completion_json.value());
  OAI_REQUIRE(reparsed_completion.has_value());
  OAI_REQUIRE(reparsed_completion.value().choices.size() == 1U);

  openai::chat_completions::completion_chunk chunk{};
  chunk.id = "chatcmpl_1";
  chunk.object = "chat.completion.chunk";
  chunk.model = "gpt-4o-mini";
  chunk.created = 1744281601;

  openai::chat_completions::choice delta_choice{};
  delta_choice.index = 0;
  openai::chat_completions::message delta{};
  delta.role = "assistant";
  delta.content = std::string{"Hel"};
  delta_choice.delta = std::move(delta);
  chunk.choices.push_back(std::move(delta_choice));

  auto chunk_json = openai::chat_completions::serialize_completion_chunk(chunk);
  OAI_REQUIRE(chunk_json.has_value());
  OAI_REQUIRE(chunk_json.value().find("\"finish_reason\":null") !=
              std::string::npos);
  auto reparsed_chunk =
      openai::chat_completions::parse_completion_chunk(chunk_json.value());
  OAI_REQUIRE(reparsed_chunk.has_value());
  OAI_REQUIRE(reparsed_chunk.value().choices.size() == 1U);
  OAI_REQUIRE(reparsed_chunk.value().choices[0].delta.has_value());
}

} // namespace
