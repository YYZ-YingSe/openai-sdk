#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fixture_io.hpp"
#include "openai/openai.hpp"
#include "schema_validation.hpp"
#include "test_support.hpp"

namespace {

auto require_valid_schema(std::string_view schema_pointer, std::string_view json,
                          std::string_view label) -> void {
  const auto result = openai::test::validate_contract_schema(schema_pointer, json);
  OAI_REQUIRE_MSG(result.valid,
                  std::string{label} + ": " + result.message);
}

auto require_invalid_schema(std::string_view schema_pointer, std::string_view json,
                            std::string_view label) -> void {
  const auto result = openai::test::validate_contract_schema(schema_pointer, json);
  OAI_REQUIRE_MSG(!result.valid,
                  std::string{label} +
                      ": schema unexpectedly accepted malformed payload");
}

auto require_valid_fixture(std::string_view schema_pointer,
                           std::string_view fixture_path) -> void {
  require_valid_schema(schema_pointer, openai::test::read_fixture(fixture_path),
                       fixture_path);
}

OAI_TEST("golden fixtures satisfy retained contract schemas") {
  require_valid_fixture("/definitions/responses_request_document",
                        "responses/request-basic.json");
  require_valid_fixture("/definitions/responses_response_document",
                        "responses/response-basic.json");
  require_valid_fixture("/definitions/responses_response_document",
                        "responses/response-unknown-item.json");
  require_valid_fixture("/definitions/chat_request_document",
                        "chat_completions/request-basic.json");
  require_valid_fixture("/definitions/chat_completion_document",
                        "chat_completions/response-basic.json");
  require_valid_fixture("/definitions/completions_request_document",
                        "completions/request-basic.json");
  require_valid_fixture("/definitions/completions_response_document",
                        "completions/response-basic.json");
  require_valid_fixture("/definitions/embeddings_request_document",
                        "embeddings/request-basic.json");
  require_valid_fixture("/definitions/models_list_response_document",
                        "models/list-basic.json");

  const auto stream_text =
      openai::test::read_fixture("streaming/response-created.sse");
  auto parsed = openai::streaming::parse_response_event_stream(stream_text);
  OAI_REQUIRE(parsed.has_value());
  OAI_REQUIRE(parsed.value().size() == 2U);

  auto created_json =
      openai::streaming::serialize_response_event_json(parsed.value()[0]);
  OAI_REQUIRE(created_json.has_value());
  require_valid_schema("/definitions/streaming_response_event_document",
                       created_json.value(), "streaming/response-created.sse");
  OAI_REQUIRE(parsed.value()[1].type == "[DONE]");
}

OAI_TEST("serialized outputs satisfy retained contract schemas") {
  openai::responses::request responses_request{};
  responses_request.model = "gpt-4.1";
  responses_request.instructions = std::string{"Answer briefly."};
  openai::responses::message_item input_message{};
  input_message.role = "user";
  input_message.content.push_back(
      openai::responses::input_text_content{"hello"});
  input_message.content.push_back(openai::responses::input_image_content{
      .image_url = "https://example.com/demo.png",
      .detail = "high",
  });
  std::vector<openai::responses::item> input_items{};
  input_items.emplace_back(std::move(input_message));
  responses_request.input = std::move(input_items);
  openai::responses::function_tool responses_function_tool{};
  responses_function_tool.name = "lookup_weather";
  responses_function_tool.description = "Look up the weather";
  responses_function_tool.strict = true;
  responses_function_tool.parameters_json =
      R"json({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})json";
  responses_request.tools.push_back(std::move(responses_function_tool));
  openai::responses::custom_tool responses_custom_tool{};
  responses_custom_tool.name = "apply_patch";
  responses_custom_tool.description = "Apply a context-verified patch";
  responses_custom_tool.format =
      openai::responses::custom_tool_grammar_format{
          .syntax = "lark",
          .definition = "start: patch\npatch: /.+/",
      };
  responses_request.tools.push_back(std::move(responses_custom_tool));
  responses_request.tools.push_back(openai::responses::apply_patch_tool{});
  openai::responses::tool_choice responses_tool_choice{};
  responses_tool_choice.type = "function";
  responses_tool_choice.name = "lookup_weather";
  responses_request.tool_choice_value = std::move(responses_tool_choice);
  openai::responses::reasoning_config responses_reasoning{};
  responses_reasoning.effort = "medium";
  responses_request.reasoning = std::move(responses_reasoning);
  openai::responses::response_text_config responses_text{};
  responses_text.type = "json_schema";
  responses_text.schema_name = "weather_answer";
  responses_text.strict = true;
  responses_text.schema_json =
      R"json({"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"]})json";
  responses_request.text = std::move(responses_text);
  responses_request.background = false;
  responses_request.parallel_tool_calls = true;
  responses_request.store = true;
  responses_request.stream = true;
  responses_request.stream_options_value =
      openai::responses::stream_options{.include_obfuscation = true};
  responses_request.previous_response_id = "resp_prev";
  responses_request.max_output_tokens = 128;
  responses_request.max_tool_calls = 4;
  responses_request.temperature = 0.4;
  responses_request.top_p = 0.95;
  responses_request.top_logprobs = 2.0;
  responses_request.truncation = "auto";
  responses_request.prompt_cache_key = "weather-key";
  responses_request.prompt_cache_retention = "session";
  responses_request.safety_identifier = "sid_1";
  responses_request.service_tier = "default";
  responses_request.context_management.push_back(
      openai::responses::context_management_entry{
          .type = "auto",
          .compact_threshold = 12,
      });
  responses_request.conversation = openai::responses::raw_json{
      .type = "conversation",
      .data = openai::core::value::object{
          {"id", openai::core::value{"conv_1"}},
      },
  };
  responses_request.prompt = openai::responses::raw_json{
      .type = "prompt",
      .data = openai::core::value::object{
          {"id", openai::core::value{"prompt_1"}},
      },
  };
  responses_request.include = {"output_text"};
  responses_request.metadata = {{"trace_id", "trace-123"}};
  responses_request.user = "sdk-user";

  auto responses_request_json =
      openai::responses::serialize_request(responses_request);
  OAI_REQUIRE(responses_request_json.has_value());
  require_valid_schema("/definitions/responses_request_document",
                       responses_request_json.value(),
                       "serialized responses request");

  openai::responses::response responses_response{};
  responses_response.id = "resp_123";
  responses_response.object = "response";
  responses_response.model = "gpt-4.1";
  responses_response.status = "completed";
  responses_response.created_at = 1744281600;
  responses_response.completed_at = 1744281605;
  responses_response.instructions = std::string{"Answer briefly."};

  openai::responses::message_item assistant_message{};
  assistant_message.id = "msg_1";
  assistant_message.status = "completed";
  assistant_message.role = "assistant";
  openai::responses::output_text_content output_text{};
  output_text.text = "It is sunny.";
  output_text.annotations.push_back(openai::responses::annotation{
      .type = "url_citation",
      .title = "Forecast",
      .url = "https://example.com/forecast",
  });
  output_text.logprobs.push_back(openai::responses::logprob_token{
      .token = "sunny",
      .logprob = -0.1,
      .bytes = {115, 117, 110, 110, 121},
  });
  assistant_message.content.emplace_back(std::move(output_text));

  openai::responses::function_call_item call_item{};
  call_item.id = "fc_1";
  call_item.status = "completed";
  call_item.call_id = "call_1";
  call_item.name = "lookup_weather";
  call_item.arguments = R"json({"city":"Hangzhou"})json";

  openai::responses::function_call_output_item call_output{};
  call_output.id = "fco_1";
  call_output.status = "completed";
  call_output.call_id = "call_1";
  std::vector<openai::responses::message_content_part> tool_payload{};
  tool_payload.emplace_back(
      openai::responses::output_text_content{.text = "22C and sunny"});
  call_output.output_payload = std::move(tool_payload);

  openai::responses::mcp_call_item mcp_item{};
  mcp_item.id = "mcp_1";
  mcp_item.status = "completed";
  mcp_item.name = "list_docs";
  mcp_item.server_label = "docs";
  mcp_item.output_json = R"json({"results":[{"id":"doc_1"}]})json";

  responses_response.output_items.emplace_back(std::move(assistant_message));
  responses_response.output_items.emplace_back(std::move(call_item));
  responses_response.output_items.emplace_back(std::move(call_output));
  responses_response.output_items.emplace_back(std::move(mcp_item));
  responses_response.tools.push_back(openai::responses::apply_patch_tool{});
  openai::responses::tool_choice response_tool_choice{};
  response_tool_choice.type = "function";
  response_tool_choice.name = "lookup_weather";
  responses_response.tool_choice_value = std::move(response_tool_choice);
  responses_response.usage_value = openai::responses::usage{
      .input_tokens = 10,
      .output_tokens = 6,
      .total_tokens = 16,
      .input_details = openai::responses::usage_details{.cached_tokens = 2},
      .output_details =
          openai::responses::usage_details{.reasoning_tokens = 1},
  };
  openai::responses::response_text_config response_text_config{};
  response_text_config.type = "json_schema";
  response_text_config.schema_name = "weather_answer";
  response_text_config.strict = true;
  response_text_config.schema_json =
      R"json({"type":"object","properties":{"summary":{"type":"string"}}})json";
  responses_response.text = std::move(response_text_config);
  openai::responses::reasoning_config response_reasoning{};
  response_reasoning.summary = "Condensed reasoning";
  responses_response.reasoning = std::move(response_reasoning);
  responses_response.parallel_tool_calls = true;
  responses_response.stream = true;
  responses_response.stream_options_value =
      openai::responses::stream_options{.include_obfuscation = true};
  responses_response.max_output_tokens = 128;
  responses_response.max_tool_calls = 4;
  responses_response.temperature = 0.4;
  responses_response.top_p = 0.95;
  responses_response.top_logprobs = 2.0;
  responses_response.truncation = "auto";
  responses_response.prompt_cache_key = "weather-key";
  responses_response.prompt_cache_retention = "session";
  responses_response.safety_identifier = "sid_1";
  responses_response.service_tier = "default";
  responses_response.metadata = {{"trace_id", "trace-123"}};
  responses_response.user = "sdk-user";

  auto responses_response_json =
      openai::responses::serialize_response(responses_response);
  OAI_REQUIRE(responses_response_json.has_value());
  require_valid_schema("/definitions/responses_response_document",
                       responses_response_json.value(),
                       "serialized responses response");

  openai::chat_completions::request chat_request{};
  chat_request.model = "gpt-4o-mini";
  openai::chat_completions::message chat_message{};
  chat_message.role = "user";
  std::vector<openai::chat_completions::content_part> chat_parts{};
  chat_parts.emplace_back(
      openai::chat_completions::text_content_part{.text = "Describe this."});
  chat_parts.emplace_back(openai::chat_completions::image_content_part{
      .url = "https://example.com/city.png",
      .detail = "high",
  });
  chat_parts.emplace_back(openai::chat_completions::audio_content_part{
      .data = "YWJj",
      .format = "wav",
  });
  chat_message.content = std::move(chat_parts);
  chat_request.messages.push_back(std::move(chat_message));
  openai::responses::function_tool chat_tool{};
  chat_tool.name = "lookup_weather";
  chat_tool.parameters_json = R"json({"type":"object"})json";
  chat_request.tools.push_back(std::move(chat_tool));
  openai::responses::tool_choice chat_tool_choice{};
  chat_tool_choice.type = "function";
  chat_tool_choice.name = "lookup_weather";
  chat_request.tool_choice_value = std::move(chat_tool_choice);
  openai::responses::response_text_config chat_response_format{};
  chat_response_format.type = "json_schema";
  chat_response_format.schema_name = "answer";
  chat_response_format.schema_json =
      R"json({"type":"object","properties":{"summary":{"type":"string"}}})json";
  chat_request.response_format = std::move(chat_response_format);
  chat_request.audio = openai::core::value::object{
      {"voice", openai::core::value{"alloy"}},
  };
  chat_request.function_call = openai::core::value::object{
      {"name", openai::core::value{"lookup_weather"}},
  };
  chat_request.functions.push_back(openai::core::value::object{
      {"name", openai::core::value{"lookup_weather"}},
  });
  chat_request.frequency_penalty = 0.1;
  chat_request.logit_bias = openai::core::value::object{
      {"42", openai::core::value{1}},
  };
  chat_request.logprobs = true;
  chat_request.max_completion_tokens = 64;
  chat_request.max_tokens = 128;
  chat_request.modalities = {"text"};
  chat_request.n = 1;
  chat_request.parallel_tool_calls = true;
  chat_request.prediction = openai::core::value::object{
      {"type", openai::core::value{"content"}},
  };
  chat_request.presence_penalty = 0.2;
  chat_request.prompt_cache_key = "chat-key";
  chat_request.prompt_cache_retention = "session";
  chat_request.reasoning_effort = "medium";
  chat_request.safety_identifier = "sid_2";
  chat_request.seed = 7;
  chat_request.service_tier = "default";
  chat_request.stop = std::vector<std::string>{"END"};
  chat_request.store = true;
  chat_request.temperature = 0.3;
  chat_request.top_logprobs = 2;
  chat_request.top_p = 0.8;
  chat_request.stream = true;
  chat_request.stream_options_value =
      openai::chat_completions::stream_options{
          .include_obfuscation = true,
          .include_usage = true,
      };
  chat_request.metadata = {{"trace_id", "trace-chat"}};
  chat_request.user = "sdk-user";
  chat_request.web_search_options = openai::core::value::object{
      {"search_context_size", openai::core::value{"medium"}},
  };

  auto chat_request_json =
      openai::chat_completions::serialize_request(chat_request);
  OAI_REQUIRE(chat_request_json.has_value());
  require_valid_schema("/definitions/chat_request_document",
                       chat_request_json.value(), "serialized chat request");

  openai::chat_completions::completion chat_completion{};
  chat_completion.id = "chatcmpl_1";
  chat_completion.object = "chat.completion";
  chat_completion.model = "gpt-4o-mini";
  chat_completion.created = 1744281600;
  chat_completion.service_tier = "default";
  chat_completion.system_fingerprint = "fp_123";
  openai::chat_completions::choice chat_choice{};
  chat_choice.index = 0;
  chat_choice.finish_reason = "stop";
  chat_choice.logprobs_value = openai::core::value::object{
      {"content", openai::core::value::array{}},
  };
  openai::chat_completions::message assistant_completion_message{};
  assistant_completion_message.role = "assistant";
  assistant_completion_message.content = std::string{"Hello there"};
  openai::chat_completions::tool_call completion_tool_call{};
  completion_tool_call.id = "tool_1";
  completion_tool_call.type = "function";
  completion_tool_call.name = "lookup_weather";
  completion_tool_call.arguments = R"json({"city":"Hangzhou"})json";
  assistant_completion_message.tool_calls.push_back(
      std::move(completion_tool_call));
  assistant_completion_message.function_call_value =
      openai::chat_completions::function_call{
          .name = "lookup_weather",
          .arguments = R"json({"city":"Hangzhou"})json",
      };
  chat_choice.message_value = std::move(assistant_completion_message);
  chat_completion.choices.push_back(std::move(chat_choice));
  chat_completion.usage_value = openai::chat_completions::usage{
      .prompt_tokens = 10,
      .completion_tokens = 4,
      .total_tokens = 14,
  };

  auto chat_completion_json =
      openai::chat_completions::serialize_completion(chat_completion);
  OAI_REQUIRE(chat_completion_json.has_value());
  require_valid_schema("/definitions/chat_completion_document",
                       chat_completion_json.value(),
                       "serialized chat completion");

  openai::chat_completions::completion_chunk chat_chunk{};
  chat_chunk.id = "chatcmpl_1";
  chat_chunk.object = "chat.completion.chunk";
  chat_chunk.model = "gpt-4o-mini";
  chat_chunk.created = 1744281601;
  openai::chat_completions::choice chat_delta_choice{};
  chat_delta_choice.index = 0;
  openai::chat_completions::message delta_message{};
  delta_message.role = "assistant";
  delta_message.content = std::string{"Hel"};
  chat_delta_choice.delta = std::move(delta_message);
  chat_chunk.choices.push_back(std::move(chat_delta_choice));

  auto chat_chunk_json =
      openai::chat_completions::serialize_completion_chunk(chat_chunk);
  OAI_REQUIRE(chat_chunk_json.has_value());
  require_valid_schema("/definitions/chat_completion_chunk_document",
                       chat_chunk_json.value(), "serialized chat chunk");

  openai::completions::request completions_request{};
  completions_request.model = "gpt-3.5-turbo-instruct";
  completions_request.prompt =
      std::vector<std::vector<std::int64_t>>{{1, 2, 3}, {4, 5}};
  completions_request.best_of = 2;
  completions_request.echo = false;
  completions_request.frequency_penalty = 0.2;
  completions_request.logit_bias = openai::core::value::object{
      {"42", openai::core::value{1}},
  };
  completions_request.logprobs = 2;
  completions_request.max_tokens = 32;
  completions_request.n = 1;
  completions_request.presence_penalty = 0.1;
  completions_request.seed = 5;
  completions_request.stop = std::vector<std::string>{"END"};
  completions_request.stream = true;
  completions_request.stream_options_value =
      openai::completions::stream_options{
          .include_obfuscation = true,
          .include_usage = true,
      };
  completions_request.suffix = "!";
  completions_request.temperature = 0.6;
  completions_request.top_p = 0.9;
  completions_request.user = "sdk-user";

  auto completions_request_json =
      openai::completions::serialize_request(completions_request);
  OAI_REQUIRE(completions_request_json.has_value());
  require_valid_schema("/definitions/completions_request_document",
                       completions_request_json.value(),
                       "serialized completions request");

  openai::completions::response completions_response{};
  completions_response.id = "cmpl_1";
  completions_response.object = "text_completion";
  completions_response.created = 1744281600;
  completions_response.model = "gpt-3.5-turbo-instruct";
  openai::completions::choice completions_choice{};
  completions_choice.index = 0;
  completions_choice.text = "Hello";
  completions_choice.finish_reason = "stop";
  completions_choice.logprobs_value = openai::completions::logprobs{
      .text_offset = std::vector<std::int64_t>{0},
      .token_logprobs = std::vector<double>{-0.2},
      .tokens = std::vector<std::string>{"Hello"},
      .top_logprobs = std::vector<std::map<std::string, double>>{
          {{"Hello", -0.2}},
      },
  };
  completions_response.choices.push_back(std::move(completions_choice));
  completions_response.usage_value = openai::completions::usage{
      .completion_tokens = 1,
      .prompt_tokens = 2,
      .total_tokens = 3,
      .completion_tokens_details_value =
          openai::completions::completion_tokens_details{
              .accepted_prediction_tokens = 1,
              .reasoning_tokens = 0,
          },
      .prompt_tokens_details_value =
          openai::completions::prompt_tokens_details{
              .cached_tokens = 1,
          },
  };

  auto completions_response_json =
      openai::completions::serialize_response(completions_response);
  OAI_REQUIRE(completions_response_json.has_value());
  require_valid_schema("/definitions/completions_response_document",
                       completions_response_json.value(),
                       "serialized completions response");

  openai::completions::chunk completions_chunk{};
  completions_chunk.id = "cmpl_1";
  completions_chunk.object = "text_completion";
  completions_chunk.created = 1744281601;
  completions_chunk.model = "gpt-3.5-turbo-instruct";
  openai::completions::choice chunk_choice{};
  chunk_choice.index = 0;
  chunk_choice.text = "Hel";
  completions_chunk.choices.push_back(std::move(chunk_choice));

  auto completions_chunk_json =
      openai::completions::serialize_chunk(completions_chunk);
  OAI_REQUIRE(completions_chunk_json.has_value());
  require_valid_schema("/definitions/completions_chunk_document",
                       completions_chunk_json.value(),
                       "serialized completions chunk");

  openai::embeddings::request embeddings_request{};
  embeddings_request.model = "text-embedding-3-large";
  embeddings_request.input =
      std::vector<std::vector<std::int64_t>>{{1, 2, 3}, {4, 5, 6}};
  embeddings_request.dimensions = 1024;
  embeddings_request.encoding_format = "float";
  embeddings_request.user = "sdk-user";

  auto embeddings_request_json =
      openai::embeddings::serialize_request(embeddings_request);
  OAI_REQUIRE(embeddings_request_json.has_value());
  require_valid_schema("/definitions/embeddings_request_document",
                       embeddings_request_json.value(),
                       "serialized embeddings request");

  openai::embeddings::response embeddings_response{};
  embeddings_response.object = "list";
  embeddings_response.model = "text-embedding-3-large";
  embeddings_response.usage_value.prompt_tokens = 6;
  embeddings_response.usage_value.total_tokens = 6;
  embeddings_response.data.push_back(openai::embeddings::embedding{
      .object = "embedding",
      .index = 0,
      .values = std::string{"base64-encoded-vector"},
  });

  auto embeddings_response_json =
      openai::embeddings::serialize_response(embeddings_response);
  OAI_REQUIRE(embeddings_response_json.has_value());
  require_valid_schema("/definitions/embeddings_response_document",
                       embeddings_response_json.value(),
                       "serialized embeddings response");

  openai::models::model model{};
  model.id = "gpt-4.1";
  model.object = "model";
  model.created = 1744281600;
  model.owned_by = "openai";
  auto model_json = openai::models::serialize_model(model);
  OAI_REQUIRE(model_json.has_value());
  require_valid_schema("/definitions/model_document", model_json.value(),
                       "serialized model");

  openai::models::list_response models_list{};
  models_list.object = "list";
  models_list.data.push_back(model);
  auto models_list_json = openai::models::serialize_list_response(models_list);
  OAI_REQUIRE(models_list_json.has_value());
  require_valid_schema("/definitions/models_list_response_document",
                       models_list_json.value(), "serialized models list");

  openai::streaming::response_event event{};
  event.type = "response.output_text.delta";
  event.response_id = "resp_123";
  event.item_id = "msg_1";
  event.status = "in_progress";
  event.output_index = 0;
  event.content_index = 0;
  event.item_index = 0;
  event.summary_index = 0;
  event.sequence_number = 1;
  event.delta = "Hel";
  event.text = "Hello";
  event.arguments = R"json({"city":"Hangzhou"})json";
  event.code = "print('hi')";
  event.transcript = "Hello";
  event.partial_image_b64 = "YWJj";
  event.annotation = openai::responses::annotation{
      .type = "url_citation",
      .title = "Forecast",
      .url = "https://example.com/forecast",
  };
  openai::responses::message_item event_item{};
  event_item.role = "assistant";
  event_item.content.emplace_back(
      openai::responses::output_text_content{.text = "Hello"});
  event.item = std::move(event_item);
  event.content_part =
      openai::responses::output_text_content{.text = "Hello"};
  event.response = openai::responses::response{
      .id = "resp_123",
      .object = "response",
      .status = "in_progress",
  };
  event.error = openai::responses::response_error{
      .type = "rate_limit_error",
      .message = "slow down",
  };
  event.extra_fields.emplace("provider",
                             openai::core::value{"demo-provider"});

  auto event_json = openai::streaming::serialize_response_event_json(event);
  OAI_REQUIRE(event_json.has_value());
  require_valid_schema("/definitions/streaming_response_event_document",
                       event_json.value(), "serialized response event");

  openai::compat::error_envelope compat_error{};
  compat_error.error.type = "invalid_request_error";
  compat_error.error.code = "bad_request";
  compat_error.error.message = "oops";
  auto compat_error_json =
      openai::compat::serialize_response_json(
          openai::compat::response_document{compat_error});
  OAI_REQUIRE(compat_error_json.has_value());
  require_valid_schema("/definitions/compat_error_envelope_document",
                       compat_error_json.value(), "serialized compat error");

  openai::compat::list_envelope compat_list{};
  compat_list.object = "list";
  compat_list.has_more = true;
  compat_list.first_id = "item_1";
  compat_list.last_id = "item_2";
  compat_list.next = "cursor_2";
  compat_list.data.push_back(openai::core::value::object{
      {"id", openai::core::value{"item_1"}},
      {"object", openai::core::value{"model"}},
  });
  auto compat_list_json =
      openai::compat::serialize_response_json(
          openai::compat::response_document{compat_list});
  OAI_REQUIRE(compat_list_json.has_value());
  require_valid_schema("/definitions/compat_list_envelope_document",
                       compat_list_json.value(), "serialized compat list");

  openai::compat::generic_object generic{};
  generic.data = openai::core::value::object{
      {"id", openai::core::value{"obj_1"}},
      {"object", openai::core::value{"provider.object"}},
      {"type", openai::core::value{"vendor_payload"}},
      {"payload", openai::core::value::object{
                      {"answer", openai::core::value{42}},
                  }},
  };
  auto generic_json =
      openai::compat::serialize_request_json(
          openai::compat::request_document{generic});
  OAI_REQUIRE(generic_json.has_value());
  require_valid_schema("/definitions/json_object", generic_json.value(),
                       "serialized generic compat object");
}

OAI_TEST("schemas and parsers reject malformed payloads") {
  const std::string bad_responses_request =
      R"json({"model":"gpt-4.1","stream_options":"oops"})json";
  require_invalid_schema("/definitions/responses_request_document",
                         bad_responses_request, "bad responses request");
  OAI_REQUIRE(openai::responses::parse_request(bad_responses_request).has_error());

  const std::string bad_responses_response =
      R"json({"object":"response","output":"oops"})json";
  require_invalid_schema("/definitions/responses_response_document",
                         bad_responses_response, "bad responses response");
  OAI_REQUIRE(
      openai::responses::parse_response(bad_responses_response).has_error());

  const std::string bad_chat_request =
      R"json({"model":"gpt-4o-mini","messages":"oops"})json";
  require_invalid_schema("/definitions/chat_request_document", bad_chat_request,
                         "bad chat request");
  OAI_REQUIRE(
      openai::chat_completions::parse_request(bad_chat_request).has_error());

  const std::string bad_completions_request =
      R"json({"model":"gpt-3.5-turbo-instruct"})json";
  require_invalid_schema("/definitions/completions_request_document",
                         bad_completions_request, "bad completions request");
  OAI_REQUIRE(
      openai::completions::parse_request(bad_completions_request).has_error());

  const std::string bad_embeddings_request =
      R"json({"model":"text-embedding-3-large","input":[1,"two"]})json";
  require_invalid_schema("/definitions/embeddings_request_document",
                         bad_embeddings_request, "bad embeddings request");
  OAI_REQUIRE(
      openai::embeddings::parse_request(bad_embeddings_request).has_error());

  const std::string bad_models_list = R"json(
{
  "object":"list",
  "data":[{"id":"gpt-4.1","object":"model","created":"bad","owned_by":"openai"}]
}
)json";
  require_invalid_schema("/definitions/models_list_response_document",
                         bad_models_list, "bad models list");
  OAI_REQUIRE(openai::models::parse_list_response(bad_models_list).has_error());

  const std::string bad_stream_event = R"json({"type":123})json";
  require_invalid_schema("/definitions/streaming_response_event_document",
                         bad_stream_event, "bad response event");
  OAI_REQUIRE(
      openai::streaming::parse_response_event_json(bad_stream_event).has_error());
}

} // namespace
