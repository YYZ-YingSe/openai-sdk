#include <string>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("serialize response SSE event stream") {
  openai::streaming::response_event event{};
  event.type = "response.created";
  event.response_id = "resp_123";
  event.sequence_number = 1;

  openai::responses::response response{};
  response.id = "resp_123";
  response.object = "response";
  response.status = "in_progress";
  event.response = response;

  auto event_json = openai::streaming::serialize_response_event_json(event);
  OAI_REQUIRE(event_json.has_value());
  auto reparsed_event =
      openai::streaming::parse_response_event_json(event_json.value());
  OAI_REQUIRE(reparsed_event.has_value());
  OAI_REQUIRE(reparsed_event.value().type == "response.created");
  OAI_REQUIRE(reparsed_event.value().response.has_value());

  openai::streaming::sse_event sse{};
  sse.event = "response.created";
  sse.data = event_json.value();
  auto sse_text = openai::streaming::serialize_sse_event(sse);
  OAI_REQUIRE(sse_text.has_value());
  OAI_REQUIRE(sse_text.value().find("event: response.created") !=
              std::string::npos);

  auto stream = openai::streaming::serialize_sse_stream({sse});
  OAI_REQUIRE(stream.has_value());
  auto parsed_stream = openai::streaming::parse_response_event_stream(stream.value());
  OAI_REQUIRE(parsed_stream.has_value());
  OAI_REQUIRE(parsed_stream.value().size() == 1U);
  OAI_REQUIRE(parsed_stream.value()[0].type == "response.created");
}

} // namespace
