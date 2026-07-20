#include <string>
#include <variant>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("serialize responses request with tools and structured output") {
  openai::responses::request request{};
  request.model = "gpt-4.1";
  request.parallel_tool_calls = true;
  request.metadata = {{"trace_id", "abc-123"}};
  request.input = std::string{"hello"};

  openai::responses::function_tool tool{};
  tool.name = "lookup_weather";
  tool.description = "Look up weather";
  tool.parameters_json =
      R"json({"type":"object","properties":{"city":{"type":"string"}}})json";
  request.tools.push_back(tool);

  openai::responses::response_text_config text{};
  text.type = "json_schema";
  text.schema_name = "weather_answer";
  text.strict = true;
  text.schema_json =
      R"json({"type":"object","properties":{"summary":{"type":"string"}}})json";
  request.text = text;

  auto json = openai::responses::serialize_request(request);
  OAI_REQUIRE(json.has_value());
  OAI_REQUIRE(json.value().find("\"model\":\"gpt-4.1\"") != std::string::npos);
  OAI_REQUIRE(json.value().find("\"lookup_weather\"") != std::string::npos);
  OAI_REQUIRE(json.value().find("\"json_schema\"") != std::string::npos);

  auto reparsed = openai::responses::parse_request(json.value());
  OAI_REQUIRE(reparsed.has_value());
  OAI_REQUIRE(reparsed.value().model.has_value());
  OAI_REQUIRE(reparsed.value().model.value() == "gpt-4.1");
  OAI_REQUIRE(reparsed.value().tools.size() == 1U);
}

OAI_TEST("responses custom grammar tool uses nested format contract") {
  openai::responses::request request{};
  request.model = "gpt-5";
  request.input = std::string{"Update the file."};

  openai::responses::custom_tool tool{};
  tool.name = "apply_patch";
  tool.description = "Apply a context-verified patch";
  tool.format = openai::responses::custom_tool_grammar_format{
      .syntax = "lark",
      .definition = "start: patch\npatch: /.+/",
  };
  request.tools.push_back(std::move(tool));

  auto json = openai::responses::serialize_request(request);
  OAI_REQUIRE(json.has_value());
  OAI_REQUIRE(json.value().find("\"format\":{") != std::string::npos);
  OAI_REQUIRE(json.value().find("\"type\":\"grammar\"") !=
              std::string::npos);
  OAI_REQUIRE(json.value().find("\"syntax\":\"lark\"") !=
              std::string::npos);
  OAI_REQUIRE(json.value().find("\"definition\":") != std::string::npos);
  OAI_REQUIRE(json.value().find("\"grammar\":") == std::string::npos);

  auto reparsed = openai::responses::parse_request(json.value());
  OAI_REQUIRE(reparsed.has_value());
  OAI_REQUIRE(reparsed.value().tools.size() == 1U);
  const auto *reparsed_tool =
      std::get_if<openai::responses::custom_tool>(&reparsed.value().tools[0]);
  OAI_REQUIRE(reparsed_tool != nullptr);
  OAI_REQUIRE(reparsed_tool->format.has_value());
  OAI_REQUIRE(reparsed_tool->format->syntax == "lark");
  OAI_REQUIRE(reparsed_tool->format->definition ==
              "start: patch\npatch: /.+/");
}

OAI_TEST("responses custom tool rejects the legacy sibling grammar shape") {
  const auto parsed = openai::responses::parse_request(R"json({
    "model": "gpt-5",
    "input": "Update the file.",
    "tools": [{
      "type": "custom",
      "name": "apply_patch",
      "format": "grammar",
      "grammar": {"syntax": "lark", "definition": "start: patch"}
    }]
  })json");

  OAI_REQUIRE(parsed.has_error());
  OAI_REQUIRE(parsed.error().code() == openai::core::errc::type_mismatch);
}

OAI_TEST("responses request and response round-trip canonically") {
  const std::string request_json =
      R"json({"model":"gpt-4.1","input":"hello","parallel_tool_calls":true})json";
  auto parsed_request = openai::responses::parse_request(request_json);
  OAI_REQUIRE(parsed_request.has_value());
  auto serialized_request =
      openai::responses::serialize_request(parsed_request.value());
  OAI_REQUIRE(serialized_request.has_value());
  auto reparsed_request =
      openai::responses::parse_request(serialized_request.value());
  OAI_REQUIRE(reparsed_request.has_value());
  OAI_REQUIRE(reparsed_request.value().model == parsed_request.value().model);
  OAI_REQUIRE(reparsed_request.value().parallel_tool_calls ==
              parsed_request.value().parallel_tool_calls);

  const std::string response_json = R"json(
{
  "id":"resp_123",
  "object":"response",
  "model":"gpt-4.1",
  "status":"completed",
  "created_at":1744281600,
  "output":[
    {
      "type":"message",
      "role":"assistant",
      "content":[
        {"type":"output_text","text":"Hello world","annotations":[]}
      ]
    }
  ],
  "usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}
}
)json";
  auto parsed_response = openai::responses::parse_response(response_json);
  OAI_REQUIRE(parsed_response.has_value());
  auto serialized_response =
      openai::responses::serialize_response(parsed_response.value());
  OAI_REQUIRE(serialized_response.has_value());
  auto reparsed_response =
      openai::responses::parse_response(serialized_response.value());
  OAI_REQUIRE(reparsed_response.has_value());
  OAI_REQUIRE(openai::responses::collect_output_text(reparsed_response.value()) ==
              "Hello world");
  OAI_REQUIRE(reparsed_response.value().usage_value.has_value());
  OAI_REQUIRE(reparsed_response.value().usage_value->total_tokens.has_value());
  OAI_REQUIRE(reparsed_response.value().usage_value->total_tokens.value() == 5);
}

} // namespace
