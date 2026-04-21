#include <string>
#include <variant>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("compat serializes retained request and response families through one facade") {
  openai::chat_completions::request chat{};
  chat.model = "gpt-4o-mini";
  openai::chat_completions::message message{};
  message.role = "user";
  message.content = std::string{"hello"};
  chat.messages.push_back(std::move(message));

  openai::compat::request_document request_doc{chat};
  openai::compat::operation_hint hint{};
  hint.family = openai::compat::api_family::chat_completions;

  auto request_json = openai::compat::serialize_request_json(request_doc, hint);
  OAI_REQUIRE(request_json.has_value());
  OAI_REQUIRE(request_json.value().find("\"messages\"") != std::string::npos);

  openai::compat::document generic_document{request_doc};
  auto request_document_json =
      openai::compat::serialize_document_json(generic_document, hint);
  OAI_REQUIRE(request_document_json.has_value());
  OAI_REQUIRE(request_document_json.value() == request_json.value());

  auto reparsed_request =
      openai::compat::parse_request_json(request_json.value(), hint);
  OAI_REQUIRE(reparsed_request.has_value());
  OAI_REQUIRE(
      std::get_if<openai::chat_completions::request>(&reparsed_request.value()) !=
      nullptr);

  openai::models::list_response models{};
  models.object = "list";
  openai::models::model model{};
  model.id = "gpt-4.1";
  model.object = "model";
  model.created = 1744281600;
  model.owned_by = "openai";
  models.data.push_back(std::move(model));

  openai::compat::response_document response_doc{models};
  auto response_json = openai::compat::serialize_response_json(response_doc);
  OAI_REQUIRE(response_json.has_value());
  openai::compat::document response_document{response_doc};
  auto response_document_json =
      openai::compat::serialize_document_json(response_document);
  OAI_REQUIRE(response_document_json.has_value());
  OAI_REQUIRE(response_document_json.value() == response_json.value());
  auto reparsed_response =
      openai::compat::parse_document_json(response_json.value());
  OAI_REQUIRE(reparsed_response.has_value());
  OAI_REQUIRE(
      std::get_if<openai::models::list_response>(&reparsed_response.value()) !=
      nullptr);
}

OAI_TEST("compat validation forwards capability errors") {
  openai::responses::request request{};
  request.model = "demo";
  request.input = std::string{"hello"};

  openai::compat::request_document doc{request};
  openai::compat::operation_hint hint{};
  hint.family = openai::compat::api_family::responses;

  openai::capabilities::effective_profile profile{};
  profile.supported_families.clear();

  auto report = openai::compat::validate_request(doc, profile, hint);
  OAI_REQUIRE(!report.ok);
}

OAI_TEST("compat serializes generic object list and error fallbacks") {
  openai::compat::generic_object generic{};
  generic.id = "obj_1";
  generic.object = "provider.object";
  generic.type = "vendor_payload";
  generic.data = openai::core::value::object{
      {"id", openai::core::value{"obj_1"}},
      {"object", openai::core::value{"provider.object"}},
      {"type", openai::core::value{"vendor_payload"}},
      {"payload", openai::core::value::object{{"answer", openai::core::value{42}}}},
  };

  openai::compat::request_document generic_request{generic};
  auto generic_request_json =
      openai::compat::serialize_request_json(generic_request);
  OAI_REQUIRE(generic_request_json.has_value());
  auto reparsed_generic_request =
      openai::compat::parse_request_json(generic_request_json.value());
  OAI_REQUIRE(reparsed_generic_request.has_value());
  OAI_REQUIRE(
      std::get_if<openai::compat::generic_object>(&reparsed_generic_request.value()) !=
      nullptr);

  openai::compat::error_envelope error{};
  error.error.type = "invalid_request_error";
  error.error.code = "bad_request";
  error.error.message = "oops";
  openai::compat::response_document error_doc{error};
  auto error_json = openai::compat::serialize_response_json(error_doc);
  OAI_REQUIRE(error_json.has_value());
  auto reparsed_error = openai::compat::parse_document_json(error_json.value());
  OAI_REQUIRE(reparsed_error.has_value());
  OAI_REQUIRE(
      std::get_if<openai::compat::error_envelope>(&reparsed_error.value()) != nullptr);

  openai::compat::list_envelope list{};
  list.object = "list";
  list.data.push_back(openai::core::value::object{
      {"id", openai::core::value{"item_1"}},
      {"object", openai::core::value{"model"}},
  });
  openai::compat::response_document list_doc{list};
  auto list_json = openai::compat::serialize_response_json(list_doc);
  OAI_REQUIRE(list_json.has_value());
  auto reparsed_list = openai::compat::parse_document_json(list_json.value());
  OAI_REQUIRE(reparsed_list.has_value());
  OAI_REQUIRE(
      std::get_if<openai::compat::list_envelope>(&reparsed_list.value()) != nullptr);
}

} // namespace
