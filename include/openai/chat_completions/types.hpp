#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "openai/core/value.hpp"
#include "openai/responses/types.hpp"

namespace openai::chat_completions {

struct text_content_part {
  std::string text{};
};

struct image_content_part {
  std::string url{};
  std::optional<std::string> detail{};
};

struct audio_content_part {
  std::string data{};
  std::string format{};
};

using stop_sequence =
    std::variant<std::monostate, std::string, std::vector<std::string>>;

struct stream_options {
  std::optional<bool> include_obfuscation{};
  std::optional<bool> include_usage{};
};

using content_part =
    std::variant<text_content_part, image_content_part, audio_content_part,
                 responses::raw_json>;

struct function_call {
  std::optional<std::string> name{};
  std::optional<std::string> arguments{};
};

struct tool_call {
  std::optional<std::string> id{};
  std::optional<std::string> type{};
  std::optional<std::string> name{};
  std::optional<std::string> arguments{};
  std::string raw_json{};
};

using content_payload =
    std::variant<std::monostate, std::string, std::vector<content_part>>;

struct message {
  std::optional<std::string> role{};
  content_payload content{};
  std::optional<std::string> refusal{};
  std::optional<std::string> name{};
  std::optional<std::string> tool_call_id{};
  std::vector<tool_call> tool_calls{};
  std::optional<function_call> function_call_value{};
};

struct choice {
  std::int64_t index{};
  std::optional<message> message_value{};
  std::optional<message> delta{};
  std::optional<std::string> finish_reason{};
  std::optional<core::value> logprobs_value{};
};

struct usage {
  std::optional<std::int64_t> prompt_tokens{};
  std::optional<std::int64_t> completion_tokens{};
  std::optional<std::int64_t> total_tokens{};
};

struct request {
  std::optional<std::string> model{};
  std::vector<message> messages{};
  std::vector<responses::tool_definition> tools{};
  std::optional<responses::tool_choice> tool_choice_value{};
  std::optional<responses::response_text_config> response_format{};
  std::optional<core::value> audio{};
  std::optional<core::value> function_call{};
  std::vector<core::value> functions{};
  std::optional<double> frequency_penalty{};
  std::optional<core::value> logit_bias{};
  std::optional<bool> logprobs{};
  std::optional<std::int64_t> max_completion_tokens{};
  std::optional<std::int64_t> max_tokens{};
  std::vector<std::string> modalities{};
  std::optional<std::int64_t> n{};
  std::optional<bool> parallel_tool_calls{};
  std::optional<core::value> prediction{};
  std::optional<double> presence_penalty{};
  std::optional<std::string> prompt_cache_key{};
  std::optional<std::string> prompt_cache_retention{};
  std::optional<std::string> reasoning_effort{};
  std::optional<std::string> safety_identifier{};
  std::optional<std::int64_t> seed{};
  std::optional<std::string> service_tier{};
  stop_sequence stop{};
  std::optional<bool> store{};
  std::optional<double> temperature{};
  std::optional<std::int64_t> top_logprobs{};
  std::optional<double> top_p{};
  std::optional<bool> stream{};
  std::optional<stream_options> stream_options_value{};
  std::map<std::string, std::string> metadata{};
  std::optional<std::string> user{};
  std::optional<core::value> web_search_options{};
  std::string raw_json{};
};

struct completion {
  std::optional<std::string> id{};
  std::optional<std::string> object{};
  std::optional<std::string> model{};
  std::optional<std::uint64_t> created{};
  std::optional<std::string> service_tier{};
  std::optional<std::string> system_fingerprint{};
  std::vector<choice> choices{};
  std::optional<usage> usage_value{};
  std::string raw_json{};
};

struct completion_chunk {
  std::optional<std::string> id{};
  std::optional<std::string> object{};
  std::optional<std::string> model{};
  std::optional<std::uint64_t> created{};
  std::optional<std::string> service_tier{};
  std::optional<std::string> system_fingerprint{};
  std::vector<choice> choices{};
  std::optional<usage> usage_value{};
  std::string raw_json{};
};

} // namespace openai::chat_completions
