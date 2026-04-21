#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "fixture_io.hpp"
#include "openai/openai.hpp"
#include "schema_validation.hpp"
#include "test_support.hpp"

namespace {

auto require_valid_schema(std::string_view schema_pointer, std::string_view json,
                          std::string_view label) -> void {
  const auto result = openai::test::validate_contract_schema(schema_pointer, json);
  OAI_REQUIRE_MSG(result.valid, std::string{label} + ": " + result.message);
}

auto require_invalid_schema(std::string_view schema_pointer, std::string_view json,
                            std::string_view label) -> void {
  const auto result = openai::test::validate_contract_schema(schema_pointer, json);
  OAI_REQUIRE_MSG(!result.valid,
                  std::string{label} + ": schema unexpectedly accepted payload");
}

OAI_TEST("expanded retained fixture corpus round-trips and satisfies contracts") {
  const auto responses_request_fixture =
      openai::test::read_fixture("responses/request-tooling.json");
  require_valid_schema("/definitions/responses_request_document",
                       responses_request_fixture, "responses/request-tooling.json");
  auto parsed_responses_request =
      openai::responses::parse_request(responses_request_fixture);
  OAI_REQUIRE(parsed_responses_request.has_value());
  OAI_REQUIRE(parsed_responses_request.value().tools.size() == 3U);
  auto serialized_responses_request =
      openai::responses::serialize_request(parsed_responses_request.value());
  OAI_REQUIRE(serialized_responses_request.has_value());
  require_valid_schema("/definitions/responses_request_document",
                       serialized_responses_request.value(),
                       "serialized responses/request-tooling.json");
  auto reparsed_responses_request =
      openai::responses::parse_request(serialized_responses_request.value());
  OAI_REQUIRE(reparsed_responses_request.has_value());
  OAI_REQUIRE(reparsed_responses_request.value().context_management.size() == 1U);

  const auto responses_response_fixture =
      openai::test::read_fixture("responses/response-tooling.json");
  require_valid_schema("/definitions/responses_response_document",
                       responses_response_fixture,
                       "responses/response-tooling.json");
  auto parsed_responses_response =
      openai::responses::parse_response(responses_response_fixture);
  OAI_REQUIRE(parsed_responses_response.has_value());
  OAI_REQUIRE(parsed_responses_response.value().output_items.size() == 4U);
  OAI_REQUIRE(openai::responses::collect_output_text(
                  parsed_responses_response.value()) ==
              "Hangzhou is 22C and sunny.");
  auto serialized_responses_response =
      openai::responses::serialize_response(parsed_responses_response.value());
  OAI_REQUIRE(serialized_responses_response.has_value());
  require_valid_schema("/definitions/responses_response_document",
                       serialized_responses_response.value(),
                       "serialized responses/response-tooling.json");
  auto reparsed_responses_response =
      openai::responses::parse_response(serialized_responses_response.value());
  OAI_REQUIRE(reparsed_responses_response.has_value());
  OAI_REQUIRE(reparsed_responses_response.value().usage_value.has_value());

  const auto chat_request_fixture =
      openai::test::read_fixture("chat_completions/request-multimodal.json");
  require_valid_schema("/definitions/chat_request_document", chat_request_fixture,
                       "chat_completions/request-multimodal.json");
  auto parsed_chat_request =
      openai::chat_completions::parse_request(chat_request_fixture);
  OAI_REQUIRE(parsed_chat_request.has_value());
  OAI_REQUIRE(parsed_chat_request.value().messages.size() == 2U);
  auto serialized_chat_request =
      openai::chat_completions::serialize_request(parsed_chat_request.value());
  OAI_REQUIRE(serialized_chat_request.has_value());
  require_valid_schema("/definitions/chat_request_document",
                       serialized_chat_request.value(),
                       "serialized chat_completions/request-multimodal.json");
  auto reparsed_chat_request =
      openai::chat_completions::parse_request(serialized_chat_request.value());
  OAI_REQUIRE(reparsed_chat_request.has_value());
  OAI_REQUIRE(reparsed_chat_request.value().stream.has_value());
  OAI_REQUIRE(reparsed_chat_request.value().stream.value());

  const auto chat_chunk_fixture =
      openai::test::read_fixture("chat_completions/chunk-tool-call.json");
  require_valid_schema("/definitions/chat_completion_chunk_document",
                       chat_chunk_fixture,
                       "chat_completions/chunk-tool-call.json");
  auto parsed_chat_chunk =
      openai::chat_completions::parse_completion_chunk(chat_chunk_fixture);
  OAI_REQUIRE(parsed_chat_chunk.has_value());
  OAI_REQUIRE(parsed_chat_chunk.value().choices.size() == 1U);
  OAI_REQUIRE(parsed_chat_chunk.value().choices[0].delta.has_value());
  OAI_REQUIRE(parsed_chat_chunk.value().choices[0].delta->tool_calls.size() == 1U);
  auto serialized_chat_chunk =
      openai::chat_completions::serialize_completion_chunk(
          parsed_chat_chunk.value());
  OAI_REQUIRE(serialized_chat_chunk.has_value());
  require_valid_schema("/definitions/chat_completion_chunk_document",
                       serialized_chat_chunk.value(),
                       "serialized chat_completions/chunk-tool-call.json");

  const auto completions_request_fixture =
      openai::test::read_fixture("completions/request-token-batch.json");
  require_valid_schema("/definitions/completions_request_document",
                       completions_request_fixture,
                       "completions/request-token-batch.json");
  auto parsed_completions_request =
      openai::completions::parse_request(completions_request_fixture);
  OAI_REQUIRE(parsed_completions_request.has_value());
  auto serialized_completions_request =
      openai::completions::serialize_request(parsed_completions_request.value());
  OAI_REQUIRE(serialized_completions_request.has_value());
  require_valid_schema("/definitions/completions_request_document",
                       serialized_completions_request.value(),
                       "serialized completions/request-token-batch.json");
  auto reparsed_completions_request =
      openai::completions::parse_request(serialized_completions_request.value());
  OAI_REQUIRE(reparsed_completions_request.has_value());
  OAI_REQUIRE(reparsed_completions_request.value().stream.has_value());

  const auto completions_chunk_fixture =
      openai::test::read_fixture("completions/chunk-basic.json");
  require_valid_schema("/definitions/completions_chunk_document",
                       completions_chunk_fixture, "completions/chunk-basic.json");
  auto parsed_completions_chunk =
      openai::completions::parse_chunk(completions_chunk_fixture);
  OAI_REQUIRE(parsed_completions_chunk.has_value());
  auto serialized_completions_chunk =
      openai::completions::serialize_chunk(parsed_completions_chunk.value());
  OAI_REQUIRE(serialized_completions_chunk.has_value());
  require_valid_schema("/definitions/completions_chunk_document",
                       serialized_completions_chunk.value(),
                       "serialized completions/chunk-basic.json");

  const auto embeddings_request_fixture =
      openai::test::read_fixture("embeddings/request-token-batch.json");
  require_valid_schema("/definitions/embeddings_request_document",
                       embeddings_request_fixture,
                       "embeddings/request-token-batch.json");
  auto parsed_embeddings_request =
      openai::embeddings::parse_request(embeddings_request_fixture);
  OAI_REQUIRE(parsed_embeddings_request.has_value());
  auto serialized_embeddings_request =
      openai::embeddings::serialize_request(parsed_embeddings_request.value());
  OAI_REQUIRE(serialized_embeddings_request.has_value());
  require_valid_schema("/definitions/embeddings_request_document",
                       serialized_embeddings_request.value(),
                       "serialized embeddings/request-token-batch.json");

  const auto embeddings_response_fixture =
      openai::test::read_fixture("embeddings/response-base64.json");
  require_valid_schema("/definitions/embeddings_response_document",
                       embeddings_response_fixture,
                       "embeddings/response-base64.json");
  auto parsed_embeddings_response =
      openai::embeddings::parse_response(embeddings_response_fixture);
  OAI_REQUIRE(parsed_embeddings_response.has_value());
  OAI_REQUIRE(parsed_embeddings_response.value().data.size() == 1U);
  OAI_REQUIRE(std::holds_alternative<std::string>(
      parsed_embeddings_response.value().data[0].values));
  auto serialized_embeddings_response =
      openai::embeddings::serialize_response(parsed_embeddings_response.value());
  OAI_REQUIRE(serialized_embeddings_response.has_value());
  require_valid_schema("/definitions/embeddings_response_document",
                       serialized_embeddings_response.value(),
                       "serialized embeddings/response-base64.json");

  const auto model_fixture = openai::test::read_fixture("models/model-basic.json");
  require_valid_schema("/definitions/model_document", model_fixture,
                       "models/model-basic.json");
  auto parsed_model = openai::models::parse_model(model_fixture);
  OAI_REQUIRE(parsed_model.has_value());
  auto serialized_model = openai::models::serialize_model(parsed_model.value());
  OAI_REQUIRE(serialized_model.has_value());
  require_valid_schema("/definitions/model_document", serialized_model.value(),
                       "serialized models/model-basic.json");

  const auto models_list_fixture =
      openai::test::read_fixture("models/list-extended.json");
  require_valid_schema("/definitions/models_list_response_document",
                       models_list_fixture, "models/list-extended.json");
  auto parsed_models_list =
      openai::models::parse_list_response(models_list_fixture);
  OAI_REQUIRE(parsed_models_list.has_value());
  OAI_REQUIRE(parsed_models_list.value().data.size() == 2U);
  auto serialized_models_list =
      openai::models::serialize_list_response(parsed_models_list.value());
  OAI_REQUIRE(serialized_models_list.has_value());
  require_valid_schema("/definitions/models_list_response_document",
                       serialized_models_list.value(),
                       "serialized models/list-extended.json");
}

OAI_TEST("rich streaming fixture round-trips through event and sse codecs") {
  const auto rich_stream = openai::test::read_fixture("streaming/response-rich.sse");
  auto parsed = openai::streaming::parse_response_event_stream(rich_stream);
  OAI_REQUIRE(parsed.has_value());
  OAI_REQUIRE(parsed.value().size() == 4U);
  OAI_REQUIRE(parsed.value()[0].response.has_value());
  OAI_REQUIRE(parsed.value()[1].item.has_value());
  OAI_REQUIRE(parsed.value()[2].delta.has_value());
  OAI_REQUIRE(parsed.value()[3].type == "[DONE]");

  std::vector<openai::streaming::sse_event> events{};
  events.reserve(parsed.value().size());
  for (const auto &event : parsed.value()) {
    if (event.type == "[DONE]") {
      openai::streaming::sse_event done{};
      done.data = "[DONE]";
      events.push_back(std::move(done));
      continue;
    }
    auto event_json = openai::streaming::serialize_response_event_json(event);
    OAI_REQUIRE(event_json.has_value());
    require_valid_schema("/definitions/streaming_response_event_document",
                         event_json.value(), "streaming/response-rich.sse event");
    openai::streaming::sse_event sse{};
    sse.event = event.type;
    sse.data = std::move(event_json.value());
    events.push_back(std::move(sse));
  }

  auto serialized = openai::streaming::serialize_sse_stream(events);
  OAI_REQUIRE(serialized.has_value());
  auto reparsed = openai::streaming::parse_response_event_stream(serialized.value());
  OAI_REQUIRE(reparsed.has_value());
  OAI_REQUIRE(reparsed.value().size() == 4U);
  OAI_REQUIRE(reparsed.value()[3].type == "[DONE]");
}

OAI_TEST("compat fixture corpus covers inferred and fallback documents") {
  const auto error_fixture = openai::test::read_fixture("compat/error-envelope.json");
  auto parsed_error = openai::compat::parse_document_json(error_fixture);
  OAI_REQUIRE(parsed_error.has_value());
  const auto *error =
      std::get_if<openai::compat::error_envelope>(&parsed_error.value());
  OAI_REQUIRE(error != nullptr);
  auto serialized_error = openai::compat::serialize_response_json(
      openai::compat::response_document{*error});
  OAI_REQUIRE(serialized_error.has_value());
  require_valid_schema("/definitions/compat_error_envelope_document",
                       serialized_error.value(),
                       "serialized compat/error-envelope.json");

  const auto list_fixture = openai::test::read_fixture("compat/list-envelope.json");
  auto parsed_list = openai::compat::parse_document_json(list_fixture);
  OAI_REQUIRE(parsed_list.has_value());
  const auto *list = std::get_if<openai::compat::list_envelope>(&parsed_list.value());
  OAI_REQUIRE(list != nullptr);
  auto serialized_list = openai::compat::serialize_response_json(
      openai::compat::response_document{*list});
  OAI_REQUIRE(serialized_list.has_value());
  require_valid_schema("/definitions/compat_list_envelope_document",
                       serialized_list.value(),
                       "serialized compat/list-envelope.json");

  const auto generic_fixture =
      openai::test::read_fixture("compat/generic-object.json");
  auto parsed_generic = openai::compat::parse_request_json(generic_fixture);
  OAI_REQUIRE(parsed_generic.has_value());
  const auto *generic =
      std::get_if<openai::compat::generic_object>(&parsed_generic.value());
  OAI_REQUIRE(generic != nullptr);
  auto serialized_generic =
      openai::compat::serialize_request_json(
          openai::compat::request_document{*generic});
  OAI_REQUIRE(serialized_generic.has_value());
  require_valid_schema("/definitions/json_object", serialized_generic.value(),
                       "serialized compat/generic-object.json");

  auto inferred_model_list = openai::compat::parse_document_json(
      openai::test::read_fixture("models/list-extended.json"));
  OAI_REQUIRE(inferred_model_list.has_value());
  OAI_REQUIRE(std::get_if<openai::models::list_response>(
                  &inferred_model_list.value()) != nullptr);

  auto inferred_embedding_list = openai::compat::parse_document_json(
      openai::test::read_fixture("embeddings/response-base64.json"));
  OAI_REQUIRE(inferred_embedding_list.has_value());
  OAI_REQUIRE(std::get_if<openai::embeddings::response>(
                  &inferred_embedding_list.value()) != nullptr);
}

OAI_TEST("invalid fixture corpus is rejected by schema and parsers") {
  const auto bad_responses_request =
      openai::test::read_fixture(
          "invalid/responses/request-stream-options-string.json");
  require_invalid_schema("/definitions/responses_request_document",
                         bad_responses_request,
                         "invalid/responses/request-stream-options-string.json");
  OAI_REQUIRE(openai::responses::parse_request(bad_responses_request).has_error());

  const auto bad_responses_response =
      openai::test::read_fixture(
          "invalid/responses/response-output-not-array.json");
  require_invalid_schema("/definitions/responses_response_document",
                         bad_responses_response,
                         "invalid/responses/response-output-not-array.json");
  OAI_REQUIRE(openai::responses::parse_response(bad_responses_response).has_error());

  const auto bad_chat_request =
      openai::test::read_fixture(
          "invalid/chat_completions/request-messages-not-array.json");
  require_invalid_schema("/definitions/chat_request_document", bad_chat_request,
                         "invalid/chat_completions/request-messages-not-array.json");
  OAI_REQUIRE(openai::chat_completions::parse_request(bad_chat_request).has_error());

  const auto bad_chat_chunk =
      openai::test::read_fixture(
          "invalid/chat_completions/chunk-choices-not-array.json");
  require_invalid_schema("/definitions/chat_completion_chunk_document",
                         bad_chat_chunk,
                         "invalid/chat_completions/chunk-choices-not-array.json");
  OAI_REQUIRE(
      openai::chat_completions::parse_completion_chunk(bad_chat_chunk).has_error());

  const auto bad_completions_request =
      openai::test::read_fixture(
          "invalid/completions/request-missing-prompt.json");
  require_invalid_schema("/definitions/completions_request_document",
                         bad_completions_request,
                         "invalid/completions/request-missing-prompt.json");
  OAI_REQUIRE(
      openai::completions::parse_request(bad_completions_request).has_error());

  const auto bad_embeddings_request =
      openai::test::read_fixture(
          "invalid/embeddings/request-heterogeneous-input.json");
  require_invalid_schema("/definitions/embeddings_request_document",
                         bad_embeddings_request,
                         "invalid/embeddings/request-heterogeneous-input.json");
  OAI_REQUIRE(openai::embeddings::parse_request(bad_embeddings_request).has_error());

  const auto bad_models_list =
      openai::test::read_fixture("invalid/models/list-created-string.json");
  require_invalid_schema("/definitions/models_list_response_document",
                         bad_models_list,
                         "invalid/models/list-created-string.json");
  OAI_REQUIRE(openai::models::parse_list_response(bad_models_list).has_error());

  const auto bad_streaming_event =
      openai::test::read_fixture("invalid/streaming/event-type-not-string.json");
  require_invalid_schema("/definitions/streaming_response_event_document",
                         bad_streaming_event,
                         "invalid/streaming/event-type-not-string.json");
  OAI_REQUIRE(
      openai::streaming::parse_response_event_json(bad_streaming_event).has_error());

  const auto bad_compat_list =
      openai::test::read_fixture("invalid/compat/list-data-not-array.json");
  require_invalid_schema("/definitions/compat_list_envelope_document",
                         bad_compat_list, "invalid/compat/list-data-not-array.json");
  OAI_REQUIRE(openai::compat::parse_document_json(bad_compat_list).has_error());
}

} // namespace
