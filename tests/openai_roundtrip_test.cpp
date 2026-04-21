#include <string>
#include <variant>

#include "fixture_io.hpp"
#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("golden fixtures round-trip retained request families") {
  const auto responses_json =
      openai::test::read_fixture("responses/request-basic.json");
  auto parsed_responses = openai::responses::parse_request(responses_json);
  OAI_REQUIRE(parsed_responses.has_value());
  auto serialized_responses =
      openai::responses::serialize_request(parsed_responses.value());
  OAI_REQUIRE(serialized_responses.has_value());
  auto reparsed_responses =
      openai::responses::parse_request(serialized_responses.value());
  OAI_REQUIRE(reparsed_responses.has_value());
  OAI_REQUIRE(reparsed_responses.value().model == parsed_responses.value().model);

  const auto chat_json =
      openai::test::read_fixture("chat_completions/request-basic.json");
  auto parsed_chat = openai::chat_completions::parse_request(chat_json);
  OAI_REQUIRE(parsed_chat.has_value());
  auto serialized_chat =
      openai::chat_completions::serialize_request(parsed_chat.value());
  OAI_REQUIRE(serialized_chat.has_value());
  auto reparsed_chat =
      openai::chat_completions::parse_request(serialized_chat.value());
  OAI_REQUIRE(reparsed_chat.has_value());
  OAI_REQUIRE(reparsed_chat.value().messages.size() == parsed_chat.value().messages.size());

  const auto completions_json =
      openai::test::read_fixture("completions/request-basic.json");
  auto parsed_completions = openai::completions::parse_request(completions_json);
  OAI_REQUIRE(parsed_completions.has_value());
  auto serialized_completions =
      openai::completions::serialize_request(parsed_completions.value());
  OAI_REQUIRE(serialized_completions.has_value());
  auto reparsed_completions =
      openai::completions::parse_request(serialized_completions.value());
  OAI_REQUIRE(reparsed_completions.has_value());
  OAI_REQUIRE(reparsed_completions.value().model == parsed_completions.value().model);

  const auto embeddings_json =
      openai::test::read_fixture("embeddings/request-basic.json");
  auto parsed_embeddings = openai::embeddings::parse_request(embeddings_json);
  OAI_REQUIRE(parsed_embeddings.has_value());
  auto serialized_embeddings =
      openai::embeddings::serialize_request(parsed_embeddings.value());
  OAI_REQUIRE(serialized_embeddings.has_value());
  auto reparsed_embeddings =
      openai::embeddings::parse_request(serialized_embeddings.value());
  OAI_REQUIRE(reparsed_embeddings.has_value());
  OAI_REQUIRE(reparsed_embeddings.value().model == parsed_embeddings.value().model);
}

OAI_TEST("golden fixtures round-trip retained response families and streaming") {
  const auto responses_json =
      openai::test::read_fixture("responses/response-basic.json");
  auto parsed_responses = openai::responses::parse_response(responses_json);
  OAI_REQUIRE(parsed_responses.has_value());
  auto serialized_responses =
      openai::responses::serialize_response(parsed_responses.value());
  OAI_REQUIRE(serialized_responses.has_value());
  auto reparsed_responses =
      openai::responses::parse_response(serialized_responses.value());
  OAI_REQUIRE(reparsed_responses.has_value());
  OAI_REQUIRE(openai::responses::collect_output_text(reparsed_responses.value()) ==
              "Hello world");

  const auto chat_json =
      openai::test::read_fixture("chat_completions/response-basic.json");
  auto parsed_chat = openai::chat_completions::parse_completion(chat_json);
  OAI_REQUIRE(parsed_chat.has_value());
  auto serialized_chat =
      openai::chat_completions::serialize_completion(parsed_chat.value());
  OAI_REQUIRE(serialized_chat.has_value());
  auto reparsed_chat =
      openai::chat_completions::parse_completion(serialized_chat.value());
  OAI_REQUIRE(reparsed_chat.has_value());
  OAI_REQUIRE(reparsed_chat.value().choices.size() == 1U);

  const auto completions_json =
      openai::test::read_fixture("completions/response-basic.json");
  auto parsed_completions = openai::completions::parse_response(completions_json);
  OAI_REQUIRE(parsed_completions.has_value());
  auto serialized_completions =
      openai::completions::serialize_response(parsed_completions.value());
  OAI_REQUIRE(serialized_completions.has_value());
  auto reparsed_completions =
      openai::completions::parse_response(serialized_completions.value());
  OAI_REQUIRE(reparsed_completions.has_value());
  OAI_REQUIRE(reparsed_completions.value().choices.size() == 1U);

  const auto models_json =
      openai::test::read_fixture("models/list-basic.json");
  auto parsed_models = openai::models::parse_list_response(models_json);
  OAI_REQUIRE(parsed_models.has_value());
  auto serialized_models =
      openai::models::serialize_list_response(parsed_models.value());
  OAI_REQUIRE(serialized_models.has_value());
  auto reparsed_models =
      openai::models::parse_list_response(serialized_models.value());
  OAI_REQUIRE(reparsed_models.has_value());
  OAI_REQUIRE(reparsed_models.value().data.size() == 1U);

  const auto streaming_text =
      openai::test::read_fixture("streaming/response-created.sse");
  auto parsed_stream = openai::streaming::parse_response_event_stream(streaming_text);
  OAI_REQUIRE(parsed_stream.has_value());
  OAI_REQUIRE(parsed_stream.value().size() == 2U);

  openai::streaming::sse_event created{};
  created.event = "response.created";
  created.data = openai::streaming::serialize_response_event_json(
                     parsed_stream.value()[0])
                     .value();
  openai::streaming::sse_event done{};
  done.data = "[DONE]";
  auto serialized_stream =
      openai::streaming::serialize_sse_stream({created, done});
  OAI_REQUIRE(serialized_stream.has_value());
  auto reparsed_stream =
      openai::streaming::parse_response_event_stream(serialized_stream.value());
  OAI_REQUIRE(reparsed_stream.has_value());
  OAI_REQUIRE(reparsed_stream.value().size() == 2U);
  OAI_REQUIRE(reparsed_stream.value()[1].type == "[DONE]");
}

OAI_TEST("unknown responses item survives parse serialize parse") {
  const auto unknown_json =
      openai::test::read_fixture("responses/response-unknown-item.json");
  auto parsed = openai::responses::parse_response(unknown_json);
  OAI_REQUIRE(parsed.has_value());
  OAI_REQUIRE(parsed.value().output_items.size() == 1U);
  OAI_REQUIRE(std::holds_alternative<openai::responses::raw_json>(
      parsed.value().output_items[0]));

  auto serialized = openai::responses::serialize_response(parsed.value());
  OAI_REQUIRE(serialized.has_value());
  auto reparsed = openai::responses::parse_response(serialized.value());
  OAI_REQUIRE(reparsed.has_value());
  OAI_REQUIRE(reparsed.value().output_items.size() == 1U);
  OAI_REQUIRE(std::holds_alternative<openai::responses::raw_json>(
      reparsed.value().output_items[0]));
}

} // namespace
