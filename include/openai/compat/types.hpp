#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "openai/chat_completions/types.hpp"
#include "openai/completions/types.hpp"
#include "openai/compat/catalog.hpp"
#include "openai/core/value.hpp"
#include "openai/embeddings/types.hpp"
#include "openai/models/types.hpp"
#include "openai/responses/types.hpp"
#include "openai/streaming/response_events.hpp"

namespace openai::compat {

struct operation_hint {
  api_family family{api_family::unknown};
  std::string operation{};
  bool streaming{false};
};

struct error_envelope {
  responses::response_error error{};
  std::string raw_json{};
};

struct list_envelope {
  std::optional<std::string> object{};
  std::vector<core::value> data{};
  std::optional<bool> has_more{};
  std::optional<std::string> first_id{};
  std::optional<std::string> last_id{};
  std::optional<std::string> next{};
  std::string raw_json{};
};

struct generic_object {
  std::optional<std::string> id{};
  std::optional<std::string> object{};
  std::optional<std::string> type{};
  core::value data{};
  std::string raw_json{};
};

using request_document =
    std::variant<responses::request, chat_completions::request,
                 completions::request, embeddings::request, generic_object>;

using response_document =
    std::variant<responses::response, streaming::response_event,
                 chat_completions::completion,
                 chat_completions::completion_chunk, completions::response,
                 completions::chunk, embeddings::response, models::model,
                 models::list_response, error_envelope, list_envelope,
                 generic_object>;

using document = std::variant<request_document, response_document>;

} // namespace openai::compat
