#include <string>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("resolve effective capability from provider and model profiles") {
  openai::capabilities::provider_profile provider{};
  provider.supported_families = {
      openai::compat::api_family::responses,
      openai::compat::api_family::chat_completions,
  };
  provider.responses.streaming = openai::capabilities::support_level::native;

  openai::capabilities::model_profile model{};
  model.id = "gpt-4.1";
  model.structured_outputs.json_schema =
      openai::capabilities::support_level::native;

  auto effective = openai::capabilities::resolve(provider, model);
  OAI_REQUIRE(
      effective.supported_families.count(openai::compat::api_family::responses) ==
      1U);
  OAI_REQUIRE(effective.responses.streaming ==
              openai::capabilities::support_level::native);
}

OAI_TEST("validate rejects unsupported tools before serialization") {
  openai::responses::request request{};
  request.model = "demo";
  request.input = std::string{"hello"};

  openai::capabilities::effective_profile profile{};
  profile.supported_families = {openai::compat::api_family::responses};
  profile.tools.function_tools =
      openai::capabilities::support_level::unsupported;

  openai::responses::function_tool tool{};
  tool.name = "lookup_weather";
  request.tools.push_back(tool);

  auto report = openai::capabilities::validate_request(
      request, profile, openai::compat::api_family::responses);
  OAI_REQUIRE(!report.ok);
  OAI_REQUIRE(!report.issues.empty());
}

OAI_TEST("validate enforces canonical cross-field protocol rules") {
  openai::capabilities::effective_profile profile{};
  profile.supported_families = {
      openai::compat::api_family::chat_completions,
      openai::compat::api_family::completions,
      openai::compat::api_family::responses,
  };

  openai::chat_completions::request chat{};
  chat.model = "gpt-4o-mini";
  openai::chat_completions::message message{};
  message.role = "user";
  message.content = std::string{"hello"};
  chat.messages.push_back(std::move(message));
  chat.top_logprobs = 3;
  chat.logprobs = false;
  chat.stream_options_value = openai::chat_completions::stream_options{
      .include_usage = true,
  };

  auto chat_report = openai::capabilities::validate_request(
      chat, profile, openai::compat::api_family::chat_completions);
  OAI_REQUIRE(!chat_report.ok);

  openai::completions::request completions{};
  completions.model = "gpt-3.5-turbo-instruct";
  completions.prompt = std::string{"hello"};
  completions.n = 2;
  completions.best_of = 1;
  completions.stream_options_value = openai::completions::stream_options{
      .include_usage = true,
  };

  auto completions_report = openai::capabilities::validate_request(
      completions, profile, openai::compat::api_family::completions);
  OAI_REQUIRE(!completions_report.ok);

  openai::responses::request responses{};
  responses.model = "gpt-4.1";
  responses.input = std::string{"hello"};
  responses.stream_options_value = openai::responses::stream_options{
      .include_obfuscation = true,
  };
  auto responses_report = openai::capabilities::validate_request(
      responses, profile, openai::compat::api_family::responses);
  OAI_REQUIRE(!responses_report.ok);
}

OAI_TEST("validate generic passthrough depends on effective raw support") {
  openai::compat::generic_object generic{};
  generic.data = openai::core::value::object{
      {"payload", openai::core::value{"hello"}},
  };
  openai::compat::request_document doc{generic};

  openai::capabilities::effective_profile blocked{};
  blocked.raw_passthrough = openai::capabilities::support_level::unsupported;
  auto blocked_report = openai::capabilities::validate_request(doc, blocked);
  OAI_REQUIRE(!blocked_report.ok);

  openai::capabilities::effective_profile allowed{};
  allowed.raw_passthrough = openai::capabilities::support_level::passthrough;
  auto allowed_report = openai::capabilities::validate_request(doc, allowed);
  OAI_REQUIRE(allowed_report.ok);
}

} // namespace
