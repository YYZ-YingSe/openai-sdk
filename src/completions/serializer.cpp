#include "openai/completions/serializer.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "openai/core/error.hpp"

namespace openai::completions {
namespace {

using core::errc;
using core::make_error;
using core::result;

using object_t = core::value::object;
using array_t = core::value::array;

auto uint64_to_value(std::uint64_t current) -> result<core::value> {
  if (current >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return result<core::value>::failure(
        make_error(errc::not_supported, "uint64 value exceeds int64 range"));
  }
  return result<core::value>::success(
      core::value{static_cast<std::int64_t>(current)});
}

auto insert_value(object_t &object, std::string_view key, core::value value)
    -> void {
  object.emplace(std::string{key}, std::move(value));
}

auto insert_optional_string(object_t &object, std::string_view key,
                            const std::optional<std::string> &value) -> void {
  if (value.has_value()) {
    insert_value(object, key, core::value{*value});
  }
}

auto insert_optional_bool(object_t &object, std::string_view key,
                          const std::optional<bool> &value) -> void {
  if (value.has_value()) {
    insert_value(object, key, core::value{*value});
  }
}

auto insert_optional_int64(object_t &object, std::string_view key,
                           const std::optional<std::int64_t> &value) -> void {
  if (value.has_value()) {
    insert_value(object, key, core::value{*value});
  }
}

auto insert_optional_double(object_t &object, std::string_view key,
                            const std::optional<double> &value) -> void {
  if (value.has_value()) {
    insert_value(object, key, core::value{*value});
  }
}

auto serialize_prompt_input_value(const prompt_input &value)
    -> std::optional<core::value> {
  return std::visit(
      []<typename prompt_t>(const prompt_t &current) -> std::optional<core::value> {
        using current_t = std::decay_t<prompt_t>;
        if constexpr (std::is_same_v<current_t, std::monostate>) {
          return core::value{nullptr};
        } else if constexpr (std::is_same_v<current_t, std::string>) {
          return core::value{current};
        } else {
          array_t entries{};
          entries.reserve(current.size());
          for (const auto &entry : current) {
            if constexpr (std::is_same_v<current_t, std::vector<std::string>>) {
              entries.emplace_back(entry);
            } else if constexpr (std::is_same_v<current_t,
                                                std::vector<std::int64_t>>) {
              entries.emplace_back(entry);
            } else {
              array_t nested{};
              nested.reserve(entry.size());
              for (const auto token : entry) {
                nested.emplace_back(token);
              }
              entries.emplace_back(std::move(nested));
            }
          }
          return core::value{std::move(entries)};
        }
      },
      value);
}

auto serialize_stop_sequence_value(const stop_sequence &value)
    -> std::optional<core::value> {
  return std::visit(
      []<typename stop_t>(const stop_t &current) -> std::optional<core::value> {
        using current_t = std::decay_t<stop_t>;
        if constexpr (std::is_same_v<current_t, std::monostate>) {
          return std::nullopt;
        } else if constexpr (std::is_same_v<current_t, std::string>) {
          return core::value{current};
        } else {
          array_t values{};
          values.reserve(current.size());
          for (const auto &entry : current) {
            values.emplace_back(entry);
          }
          return core::value{std::move(values)};
        }
      },
      value);
}

auto serialize_logprobs_value(const logprobs &value) -> result<core::value> {
  object_t object{};
  if (value.text_offset.has_value()) {
    array_t offsets{};
    offsets.reserve(value.text_offset->size());
    for (const auto entry : *value.text_offset) {
      offsets.emplace_back(entry);
    }
    insert_value(object, "text_offset", core::value{std::move(offsets)});
  }
  if (value.token_logprobs.has_value()) {
    array_t logprobs{};
    logprobs.reserve(value.token_logprobs->size());
    for (const auto entry : *value.token_logprobs) {
      logprobs.emplace_back(entry);
    }
    insert_value(object, "token_logprobs", core::value{std::move(logprobs)});
  }
  if (value.tokens.has_value()) {
    array_t tokens{};
    tokens.reserve(value.tokens->size());
    for (const auto &entry : *value.tokens) {
      tokens.emplace_back(entry);
    }
    insert_value(object, "tokens", core::value{std::move(tokens)});
  }
  if (value.top_logprobs.has_value()) {
    array_t entries{};
    entries.reserve(value.top_logprobs->size());
    for (const auto &entry : *value.top_logprobs) {
      object_t mapped{};
      for (const auto &[token, score] : entry) {
        mapped.emplace(token, core::value{score});
      }
      entries.emplace_back(std::move(mapped));
    }
    insert_value(object, "top_logprobs", core::value{std::move(entries)});
  }
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_choice_value(const choice &value, const bool chunk)
    -> result<core::value> {
  object_t object{};
  insert_value(object, "text", core::value{value.text});
  insert_value(object, "index", core::value{value.index});
  if (value.logprobs_value.has_value()) {
    auto serialized = serialize_logprobs_value(*value.logprobs_value);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "logprobs", std::move(serialized.value()));
  }
  if (value.finish_reason.has_value()) {
    insert_value(object, "finish_reason", core::value{*value.finish_reason});
  } else if (chunk) {
    insert_value(object, "finish_reason", core::value{nullptr});
  }
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_completion_tokens_details_value(
    const completion_tokens_details &value) -> result<core::value> {
  object_t object{};
  insert_optional_int64(object, "accepted_prediction_tokens",
                        value.accepted_prediction_tokens);
  insert_optional_int64(object, "audio_tokens", value.audio_tokens);
  insert_optional_int64(object, "reasoning_tokens", value.reasoning_tokens);
  insert_optional_int64(object, "rejected_prediction_tokens",
                        value.rejected_prediction_tokens);
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_prompt_tokens_details_value(const prompt_tokens_details &value)
    -> result<core::value> {
  object_t object{};
  insert_optional_int64(object, "audio_tokens", value.audio_tokens);
  insert_optional_int64(object, "cached_tokens", value.cached_tokens);
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_usage_value(const usage &value) -> result<core::value> {
  object_t object{};
  insert_value(object, "completion_tokens", core::value{value.completion_tokens});
  insert_value(object, "prompt_tokens", core::value{value.prompt_tokens});
  insert_value(object, "total_tokens", core::value{value.total_tokens});
  if (value.completion_tokens_details_value.has_value()) {
    auto serialized = serialize_completion_tokens_details_value(
        *value.completion_tokens_details_value);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "completion_tokens_details",
                 std::move(serialized.value()));
  }
  if (value.prompt_tokens_details_value.has_value()) {
    auto serialized =
        serialize_prompt_tokens_details_value(*value.prompt_tokens_details_value);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "prompt_tokens_details", std::move(serialized.value()));
  }
  return result<core::value>::success(core::value{std::move(object)});
}

auto serialize_request_value(const request &value) -> result<core::value> {
  object_t object{};
  insert_value(object, "model", core::value{value.model});
  if (const auto prompt = serialize_prompt_input_value(value.prompt);
      prompt.has_value()) {
    insert_value(object, "prompt", *prompt);
  }
  insert_optional_int64(object, "best_of", value.best_of);
  insert_optional_bool(object, "echo", value.echo);
  insert_optional_double(object, "frequency_penalty", value.frequency_penalty);
  if (value.logit_bias.has_value()) {
    insert_value(object, "logit_bias", *value.logit_bias);
  }
  insert_optional_int64(object, "logprobs", value.logprobs);
  insert_optional_int64(object, "max_tokens", value.max_tokens);
  insert_optional_int64(object, "n", value.n);
  insert_optional_double(object, "presence_penalty", value.presence_penalty);
  insert_optional_int64(object, "seed", value.seed);
  if (const auto stop = serialize_stop_sequence_value(value.stop);
      stop.has_value()) {
    insert_value(object, "stop", *stop);
  }
  insert_optional_bool(object, "stream", value.stream);
  if (value.stream_options_value.has_value()) {
    object_t stream_options{};
    insert_optional_bool(stream_options, "include_obfuscation",
                         value.stream_options_value->include_obfuscation);
    insert_optional_bool(stream_options, "include_usage",
                         value.stream_options_value->include_usage);
    insert_value(object, "stream_options",
                 core::value{std::move(stream_options)});
  }
  insert_optional_string(object, "suffix", value.suffix);
  insert_optional_double(object, "temperature", value.temperature);
  insert_optional_double(object, "top_p", value.top_p);
  insert_optional_string(object, "user", value.user);
  return result<core::value>::success(core::value{std::move(object)});
}

template <typename output_t>
auto serialize_response_like(const output_t &value, const bool chunk)
    -> result<core::value> {
  object_t object{};
  insert_value(object, "id", core::value{value.id});
  if (!value.choices.empty()) {
    array_t choices{};
    choices.reserve(value.choices.size());
    for (const auto &entry : value.choices) {
      auto serialized = serialize_choice_value(entry, chunk);
      if (serialized.has_error()) {
        return result<core::value>::failure(serialized.error());
      }
      choices.push_back(std::move(serialized.value()));
    }
    insert_value(object, "choices", core::value{std::move(choices)});
  }
  auto created = uint64_to_value(value.created);
  if (created.has_error()) {
    return result<core::value>::failure(created.error());
  }
  insert_value(object, "created", std::move(created.value()));
  insert_value(object, "model", core::value{value.model});
  insert_value(object, "object", core::value{value.object});
  insert_optional_string(object, "system_fingerprint",
                         value.system_fingerprint);
  if (value.usage_value.has_value()) {
    auto serialized = serialize_usage_value(*value.usage_value);
    if (serialized.has_error()) {
      return result<core::value>::failure(serialized.error());
    }
    insert_value(object, "usage", std::move(serialized.value()));
  }
  return result<core::value>::success(core::value{std::move(object)});
}

} // namespace

auto serialize_request(const request &value,
                       const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_request_value(value);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_response(const response &value,
                        const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_response_like(value, false);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

auto serialize_chunk(const chunk &value, const core::serialize_options &options)
    -> core::result<std::string> {
  auto serialized = serialize_response_like(value, true);
  if (serialized.has_error()) {
    return core::result<std::string>::failure(serialized.error());
  }
  return core::serialize_json(serialized.value(), options);
}

} // namespace openai::completions
