#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "openai/core/value.hpp"

namespace openai::completions {

using prompt_input =
    std::variant<std::monostate, std::string, std::vector<std::string>,
                 std::vector<std::int64_t>,
                 std::vector<std::vector<std::int64_t>>>;

using stop_sequence =
    std::variant<std::monostate, std::string, std::vector<std::string>>;

struct stream_options {
  std::optional<bool> include_obfuscation{};
  std::optional<bool> include_usage{};
};

struct request {
  std::string model{};
  prompt_input prompt{};
  std::optional<std::int64_t> best_of{};
  std::optional<bool> echo{};
  std::optional<double> frequency_penalty{};
  std::optional<core::value> logit_bias{};
  std::optional<std::int64_t> logprobs{};
  std::optional<std::int64_t> max_tokens{};
  std::optional<std::int64_t> n{};
  std::optional<double> presence_penalty{};
  std::optional<std::int64_t> seed{};
  stop_sequence stop{};
  std::optional<bool> stream{};
  std::optional<stream_options> stream_options_value{};
  std::optional<std::string> suffix{};
  std::optional<double> temperature{};
  std::optional<double> top_p{};
  std::optional<std::string> user{};
  std::string raw_json{};
};

struct logprobs {
  std::optional<std::vector<std::int64_t>> text_offset{};
  std::optional<std::vector<double>> token_logprobs{};
  std::optional<std::vector<std::string>> tokens{};
  std::optional<std::vector<std::map<std::string, double>>> top_logprobs{};
};

struct choice {
  std::optional<std::string> finish_reason{};
  std::int64_t index{};
  std::optional<logprobs> logprobs_value{};
  std::string text{};
};

struct completion_tokens_details {
  std::optional<std::int64_t> accepted_prediction_tokens{};
  std::optional<std::int64_t> audio_tokens{};
  std::optional<std::int64_t> reasoning_tokens{};
  std::optional<std::int64_t> rejected_prediction_tokens{};
};

struct prompt_tokens_details {
  std::optional<std::int64_t> audio_tokens{};
  std::optional<std::int64_t> cached_tokens{};
};

struct usage {
  std::int64_t completion_tokens{};
  std::int64_t prompt_tokens{};
  std::int64_t total_tokens{};
  std::optional<completion_tokens_details> completion_tokens_details_value{};
  std::optional<prompt_tokens_details> prompt_tokens_details_value{};
};

struct response {
  std::string id{};
  std::vector<choice> choices{};
  std::uint64_t created{};
  std::string model{};
  std::string object{};
  std::optional<std::string> system_fingerprint{};
  std::optional<usage> usage_value{};
  std::string raw_json{};
};

struct chunk {
  std::string id{};
  std::vector<choice> choices{};
  std::uint64_t created{};
  std::string model{};
  std::string object{};
  std::optional<std::string> system_fingerprint{};
  std::optional<usage> usage_value{};
  std::string raw_json{};
};

} // namespace openai::completions
