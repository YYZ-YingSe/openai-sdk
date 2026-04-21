#include <string>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("wire parser handles headers query strings multipart and binary payloads") {
  auto headers = openai::wire::parse_headers(
      "Content-Type: application/json\r\nX-Request-Id: abc-123\r\n");
  OAI_REQUIRE(headers.has_value());
  OAI_REQUIRE(headers.value().size() == 2U);
  OAI_REQUIRE(headers.value()[0].name == "Content-Type");
  OAI_REQUIRE(headers.value()[1].value == "abc-123");

  auto query = openai::wire::parse_query_string(
      "?model=gpt-4.1&user=alice%20bob&flag&space=hello+world");
  OAI_REQUIRE(query.has_value());
  OAI_REQUIRE(query.value().parameters.size() == 4U);
  OAI_REQUIRE(query.value().parameters[1].value.has_value());
  OAI_REQUIRE(query.value().parameters[1].value.value() == "alice bob");
  OAI_REQUIRE(!query.value().parameters[2].value.has_value());
  OAI_REQUIRE(query.value().parameters[3].value.value() == "hello world");

  auto disposition = openai::wire::parse_content_disposition(
      "form-data; name=\"file\"; filename=\"demo.txt\"; size=12");
  OAI_REQUIRE(disposition.has_value());
  OAI_REQUIRE(disposition.value().type == "form-data");
  OAI_REQUIRE(disposition.value().name.has_value());
  OAI_REQUIRE(disposition.value().name.value() == "file");
  OAI_REQUIRE(disposition.value().filename.has_value());
  OAI_REQUIRE(disposition.value().filename.value() == "demo.txt");
  OAI_REQUIRE(disposition.value().parameters.at("size") == "12");

  const std::string multipart_body =
      "--boundary\r\n"
      "Content-Disposition: form-data; name=\"json\"\r\n"
      "Content-Type: application/json\r\n"
      "\r\n"
      "{\"hello\":\"world\"}\r\n"
      "--boundary\r\n"
      "Content-Disposition: form-data; name=\"file\"; filename=\"demo.txt\"\r\n"
      "Content-Type: text/plain\r\n"
      "\r\n"
      "payload\r\n"
      "--boundary--\r\n";
  auto multipart =
      openai::wire::parse_multipart_body(multipart_body, "boundary");
  OAI_REQUIRE(multipart.has_value());
  OAI_REQUIRE(multipart.value().parts.size() == 2U);
  OAI_REQUIRE(multipart.value().parts[0].content_type.has_value());
  OAI_REQUIRE(multipart.value().parts[0].content_type.value() ==
              "application/json");
  OAI_REQUIRE(multipart.value().parts[0].body == "{\"hello\":\"world\"}");
  OAI_REQUIRE(multipart.value().parts[1].disposition.has_value());
  OAI_REQUIRE(multipart.value().parts[1].disposition->filename.has_value());
  OAI_REQUIRE(multipart.value().parts[1].disposition->filename.value() ==
              "demo.txt");
  OAI_REQUIRE(multipart.value().parts[1].body == "payload");

  auto binary =
      openai::wire::parse_binary_payload("abc", std::string{"text/plain"});
  OAI_REQUIRE(binary.has_value());
  OAI_REQUIRE(binary.value().content_type.has_value());
  OAI_REQUIRE(binary.value().content_type.value() == "text/plain");
  OAI_REQUIRE(binary.value().data == "abc");
}

OAI_TEST("wire parser rejects malformed inputs") {
  OAI_REQUIRE(openai::wire::parse_headers("MissingColon\r\n").has_error());
  OAI_REQUIRE(openai::wire::parse_query_string("?bad=%ZZ").has_error());
  OAI_REQUIRE(openai::wire::parse_multipart_body(
                  "--boundary\r\nContent-Disposition: form-data; name=\"x\"\r\n\r\n"
                  "payload\r\n",
                  "boundary")
                  .has_error());
  OAI_REQUIRE(openai::wire::parse_multipart_body("", "").has_error());
}

} // namespace
