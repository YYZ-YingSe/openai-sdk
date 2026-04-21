#include <string>
#include <variant>
#include <vector>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("serialize embeddings request and response") {
  openai::embeddings::request request{};
  request.model = "text-embedding-3-large";
  request.input = std::vector<std::string>{"hello", "world"};
  request.dimensions = 1024;

  auto request_json = openai::embeddings::serialize_request(request);
  OAI_REQUIRE(request_json.has_value());
  auto reparsed_request =
      openai::embeddings::parse_request(request_json.value());
  OAI_REQUIRE(reparsed_request.has_value());
  OAI_REQUIRE(reparsed_request.value().model == request.model);

  openai::embeddings::response response{};
  response.object = "list";
  response.model = "text-embedding-3-large";
  response.usage_value.prompt_tokens = 3;
  response.usage_value.total_tokens = 3;
  openai::embeddings::embedding embedding{};
  embedding.object = "embedding";
  embedding.index = 0;
  embedding.values = std::vector<double>{0.1, 0.2, 0.3};
  response.data.push_back(std::move(embedding));

  auto response_json = openai::embeddings::serialize_response(response);
  OAI_REQUIRE(response_json.has_value());
  auto reparsed_response =
      openai::embeddings::parse_response(response_json.value());
  OAI_REQUIRE(reparsed_response.has_value());
  OAI_REQUIRE(reparsed_response.value().data.size() == 1U);
}

OAI_TEST("serialize models object and list response") {
  openai::models::model model{};
  model.id = "gpt-4.1";
  model.object = "model";
  model.created = 1744281600;
  model.owned_by = "openai";

  auto model_json = openai::models::serialize_model(model);
  OAI_REQUIRE(model_json.has_value());
  auto reparsed_model = openai::models::parse_model(model_json.value());
  OAI_REQUIRE(reparsed_model.has_value());
  OAI_REQUIRE(reparsed_model.value().id == model.id);

  openai::models::list_response list{};
  list.object = "list";
  list.data.push_back(model);
  auto list_json = openai::models::serialize_list_response(list);
  OAI_REQUIRE(list_json.has_value());
  auto reparsed_list = openai::models::parse_list_response(list_json.value());
  OAI_REQUIRE(reparsed_list.has_value());
  OAI_REQUIRE(reparsed_list.value().data.size() == 1U);
}

} // namespace
