#include "openai/completions/parser.hpp"

#include "openai/core/json.hpp"

namespace openai::completions {
namespace {

using core::errc;
using core::json_member;
using core::json_string;
using core::json_to_string;
using core::make_error;
using core::parse_json;
using core::result;

[[nodiscard]] auto required_int64_like(const core::json_value &value,
                                       std::string_view key)
    -> result<std::int64_t> {
  auto member = core::require_member(value, key);
  if (member.has_error()) {
    return result<std::int64_t>::failure(member.error());
  }
  if (!member.value()->IsInt64()) {
    return result<std::int64_t>::failure(make_error(
        errc::type_mismatch, std::string{key} + " must be an int64"));
  }
  return result<std::int64_t>::success(member.value()->GetInt64());
}

[[nodiscard]] auto parse_prompt_input_value(const core::json_value &value)
    -> result<prompt_input> {
  if (value.IsNull()) {
    return result<prompt_input>::success(prompt_input{std::monostate{}});
  }
  if (value.IsString()) {
    return result<prompt_input>::success(prompt_input{json_string(value)});
  }
  if (!value.IsArray()) {
    return result<prompt_input>::failure(make_error(
        errc::type_mismatch, "completion prompt must be null, string, or array"));
  }
  if (value.Empty()) {
    return result<prompt_input>::success(
        prompt_input{std::vector<std::string>{}});
  }
  const auto &first = value[0];
  if (first.IsString()) {
    std::vector<std::string> texts{};
    texts.reserve(value.Size());
    for (const auto &entry : value.GetArray()) {
      if (!entry.IsString()) {
        return result<prompt_input>::failure(make_error(
            errc::type_mismatch, "completion prompt string array is heterogeneous"));
      }
      texts.push_back(json_string(entry));
    }
    return result<prompt_input>::success(prompt_input{std::move(texts)});
  }
  if (first.IsInt64()) {
    std::vector<std::int64_t> tokens{};
    tokens.reserve(value.Size());
    for (const auto &entry : value.GetArray()) {
      if (!entry.IsInt64()) {
        return result<prompt_input>::failure(make_error(
            errc::type_mismatch, "completion prompt token array is heterogeneous"));
      }
      tokens.push_back(entry.GetInt64());
    }
    return result<prompt_input>::success(prompt_input{std::move(tokens)});
  }
  if (first.IsArray()) {
    std::vector<std::vector<std::int64_t>> batches{};
    batches.reserve(value.Size());
    for (const auto &entry : value.GetArray()) {
      if (!entry.IsArray()) {
        return result<prompt_input>::failure(make_error(
            errc::type_mismatch, "completion prompt nested array is heterogeneous"));
      }
      std::vector<std::int64_t> tokens{};
      tokens.reserve(entry.Size());
      for (const auto &token : entry.GetArray()) {
        if (!token.IsInt64()) {
          return result<prompt_input>::failure(make_error(
              errc::type_mismatch,
              "completion prompt token arrays must contain integers"));
        }
        tokens.push_back(token.GetInt64());
      }
      batches.push_back(std::move(tokens));
    }
    return result<prompt_input>::success(prompt_input{std::move(batches)});
  }
  return result<prompt_input>::failure(make_error(
      errc::type_mismatch, "unsupported completion prompt shape"));
}

[[nodiscard]] auto parse_stop_sequence_value(const core::json_value &value)
    -> result<stop_sequence> {
  if (value.IsNull()) {
    return result<stop_sequence>::success(stop_sequence{std::monostate{}});
  }
  if (value.IsString()) {
    return result<stop_sequence>::success(stop_sequence{json_string(value)});
  }
  if (!value.IsArray()) {
    return result<stop_sequence>::failure(make_error(
        errc::type_mismatch, "completion stop must be null, string, or array"));
  }
  std::vector<std::string> sequences{};
  sequences.reserve(value.Size());
  for (const auto &entry : value.GetArray()) {
    if (!entry.IsString()) {
      return result<stop_sequence>::failure(make_error(
          errc::type_mismatch, "completion stop array must contain strings"));
    }
    sequences.push_back(json_string(entry));
  }
  return result<stop_sequence>::success(stop_sequence{std::move(sequences)});
}

[[nodiscard]] auto parse_stream_options_value(const core::json_value &value)
    -> result<stream_options> {
  if (!value.IsObject()) {
    return result<stream_options>::failure(make_error(
        errc::type_mismatch, "completion stream_options must be an object"));
  }
  stream_options parsed{};
  if (auto field = core::optional_bool(value, "include_obfuscation");
      field.has_error()) {
    return result<stream_options>::failure(field.error());
  } else {
    parsed.include_obfuscation = field.value();
  }
  if (auto field = core::optional_bool(value, "include_usage"); field.has_error()) {
    return result<stream_options>::failure(field.error());
  } else {
    parsed.include_usage = field.value();
  }
  return result<stream_options>::success(std::move(parsed));
}

[[nodiscard]] auto parse_logprobs_value(const core::json_value &value)
    -> result<logprobs> {
  if (!value.IsObject()) {
    return result<logprobs>::failure(
        make_error(errc::type_mismatch, "completion logprobs must be an object"));
  }
  logprobs parsed{};
  const auto *text_offset = json_member(value, "text_offset");
  if (text_offset != nullptr && !text_offset->IsNull()) {
    if (!text_offset->IsArray()) {
      return result<logprobs>::failure(make_error(
          errc::type_mismatch, "text_offset must be an array"));
    }
    std::vector<std::int64_t> offsets{};
    offsets.reserve(text_offset->Size());
    for (const auto &entry : text_offset->GetArray()) {
      if (!entry.IsInt64()) {
        return result<logprobs>::failure(make_error(
            errc::type_mismatch, "text_offset entries must be integers"));
      }
      offsets.push_back(entry.GetInt64());
    }
    parsed.text_offset = std::move(offsets);
  }
  const auto *token_logprobs = json_member(value, "token_logprobs");
  if (token_logprobs != nullptr && !token_logprobs->IsNull()) {
    if (!token_logprobs->IsArray()) {
      return result<logprobs>::failure(make_error(
          errc::type_mismatch, "token_logprobs must be an array"));
    }
    std::vector<double> values{};
    values.reserve(token_logprobs->Size());
    for (const auto &entry : token_logprobs->GetArray()) {
      if (!entry.IsNumber()) {
        return result<logprobs>::failure(make_error(
            errc::type_mismatch, "token_logprobs entries must be numbers"));
      }
      values.push_back(entry.GetDouble());
    }
    parsed.token_logprobs = std::move(values);
  }
  if (auto field = core::optional_string_array(value, "tokens"); field.has_error()) {
    return result<logprobs>::failure(field.error());
  } else if (!field.value().empty()) {
    parsed.tokens = std::move(field.value());
  }
  const auto *top_logprobs = json_member(value, "top_logprobs");
  if (top_logprobs != nullptr && !top_logprobs->IsNull()) {
    if (!top_logprobs->IsArray()) {
      return result<logprobs>::failure(make_error(
          errc::type_mismatch, "top_logprobs must be an array"));
    }
    std::vector<std::map<std::string, double>> entries{};
    entries.reserve(top_logprobs->Size());
    for (const auto &entry : top_logprobs->GetArray()) {
      if (!entry.IsObject()) {
        return result<logprobs>::failure(make_error(
            errc::type_mismatch, "top_logprobs entries must be objects"));
      }
      std::map<std::string, double> values{};
      for (auto it = entry.MemberBegin(); it != entry.MemberEnd(); ++it) {
        if (!it->value.IsNumber()) {
          return result<logprobs>::failure(make_error(
              errc::type_mismatch,
              "top_logprobs object values must be numbers"));
        }
        values.emplace(json_string(it->name), it->value.GetDouble());
      }
      entries.push_back(std::move(values));
    }
    parsed.top_logprobs = std::move(entries);
  }
  return result<logprobs>::success(std::move(parsed));
}

[[nodiscard]] auto parse_choice_value(const core::json_value &value)
    -> result<choice> {
  if (!value.IsObject()) {
    return result<choice>::failure(
        make_error(errc::type_mismatch, "completion choice must be an object"));
  }
  choice parsed{};
  if (auto field = core::optional_string(value, "finish_reason"); field.has_error()) {
    return result<choice>::failure(field.error());
  } else {
    parsed.finish_reason = std::move(field.value());
  }
  if (auto field = core::optional_int64(value, "index"); field.has_error()) {
    return result<choice>::failure(field.error());
  } else {
    parsed.index = field.value().value_or(0);
  }
  const auto *logprobs_object = json_member(value, "logprobs");
  if (logprobs_object != nullptr && !logprobs_object->IsNull()) {
    auto parsed_logprobs = parse_logprobs_value(*logprobs_object);
    if (parsed_logprobs.has_error()) {
      return result<choice>::failure(parsed_logprobs.error());
    }
    parsed.logprobs_value = std::move(parsed_logprobs.value());
  }
  if (auto field = core::required_string(value, "text"); field.has_error()) {
    return result<choice>::failure(field.error());
  } else {
    parsed.text = std::move(field.value());
  }
  return result<choice>::success(std::move(parsed));
}

[[nodiscard]] auto parse_completion_tokens_details_value(
    const core::json_value &value) -> result<completion_tokens_details> {
  if (!value.IsObject()) {
    return result<completion_tokens_details>::failure(make_error(
        errc::type_mismatch,
        "completion_tokens_details must be an object"));
  }
  completion_tokens_details parsed{};
  if (auto field = core::optional_int64(value, "accepted_prediction_tokens");
      field.has_error()) {
    return result<completion_tokens_details>::failure(field.error());
  } else {
    parsed.accepted_prediction_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "audio_tokens"); field.has_error()) {
    return result<completion_tokens_details>::failure(field.error());
  } else {
    parsed.audio_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "reasoning_tokens");
      field.has_error()) {
    return result<completion_tokens_details>::failure(field.error());
  } else {
    parsed.reasoning_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "rejected_prediction_tokens");
      field.has_error()) {
    return result<completion_tokens_details>::failure(field.error());
  } else {
    parsed.rejected_prediction_tokens = field.value();
  }
  return result<completion_tokens_details>::success(std::move(parsed));
}

[[nodiscard]] auto parse_prompt_tokens_details_value(
    const core::json_value &value) -> result<prompt_tokens_details> {
  if (!value.IsObject()) {
    return result<prompt_tokens_details>::failure(make_error(
        errc::type_mismatch, "prompt_tokens_details must be an object"));
  }
  prompt_tokens_details parsed{};
  if (auto field = core::optional_int64(value, "audio_tokens"); field.has_error()) {
    return result<prompt_tokens_details>::failure(field.error());
  } else {
    parsed.audio_tokens = field.value();
  }
  if (auto field = core::optional_int64(value, "cached_tokens"); field.has_error()) {
    return result<prompt_tokens_details>::failure(field.error());
  } else {
    parsed.cached_tokens = field.value();
  }
  return result<prompt_tokens_details>::success(std::move(parsed));
}

[[nodiscard]] auto parse_usage_value(const core::json_value &value)
    -> result<usage> {
  if (!value.IsObject()) {
    return result<usage>::failure(
        make_error(errc::type_mismatch, "completion usage must be an object"));
  }
  usage parsed{};
  if (auto field = required_int64_like(value, "completion_tokens"); field.has_error()) {
    return result<usage>::failure(field.error());
  } else {
    parsed.completion_tokens = field.value();
  }
  if (auto field = required_int64_like(value, "prompt_tokens"); field.has_error()) {
    return result<usage>::failure(field.error());
  } else {
    parsed.prompt_tokens = field.value();
  }
  if (auto field = required_int64_like(value, "total_tokens"); field.has_error()) {
    return result<usage>::failure(field.error());
  } else {
    parsed.total_tokens = field.value();
  }
  const auto *completion_details = json_member(value, "completion_tokens_details");
  if (completion_details != nullptr && !completion_details->IsNull()) {
    auto parsed_details = parse_completion_tokens_details_value(*completion_details);
    if (parsed_details.has_error()) {
      return result<usage>::failure(parsed_details.error());
    }
    parsed.completion_tokens_details_value = std::move(parsed_details.value());
  }
  const auto *prompt_details = json_member(value, "prompt_tokens_details");
  if (prompt_details != nullptr && !prompt_details->IsNull()) {
    auto parsed_details = parse_prompt_tokens_details_value(*prompt_details);
    if (parsed_details.has_error()) {
      return result<usage>::failure(parsed_details.error());
    }
    parsed.prompt_tokens_details_value = std::move(parsed_details.value());
  }
  return result<usage>::success(std::move(parsed));
}

template <typename output_t>
[[nodiscard]] auto parse_response_like(const core::json_value &value)
    -> result<output_t> {
  if (!value.IsObject()) {
    return result<output_t>::failure(
        make_error(errc::type_mismatch, "completion payload must be an object"));
  }
  output_t parsed{};
  auto raw = json_to_string(value);
  if (raw.has_error()) {
    return result<output_t>::failure(raw.error());
  }
  parsed.raw_json = std::move(raw.value());
  if (auto field = core::required_string(value, "id"); field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.id = std::move(field.value());
  }
  const auto *choices = json_member(value, "choices");
  if (choices == nullptr || !choices->IsArray()) {
    return result<output_t>::failure(
        make_error(errc::type_mismatch, "completion choices must be an array"));
  }
  parsed.choices.reserve(choices->Size());
  for (const auto &entry : choices->GetArray()) {
    auto parsed_choice = parse_choice_value(entry);
    if (parsed_choice.has_error()) {
      return result<output_t>::failure(parsed_choice.error());
    }
    parsed.choices.push_back(std::move(parsed_choice.value()));
  }
  if (auto field = core::optional_uint64(value, "created"); field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.created = field.value().value_or(0);
  }
  if (auto field = core::required_string(value, "model"); field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.model = std::move(field.value());
  }
  if (auto field = core::required_string(value, "object"); field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.object = std::move(field.value());
  }
  if (auto field = core::optional_string(value, "system_fingerprint");
      field.has_error()) {
    return result<output_t>::failure(field.error());
  } else {
    parsed.system_fingerprint = std::move(field.value());
  }
  const auto *usage_object = json_member(value, "usage");
  if (usage_object != nullptr && !usage_object->IsNull()) {
    auto parsed_usage = parse_usage_value(*usage_object);
    if (parsed_usage.has_error()) {
      return result<output_t>::failure(parsed_usage.error());
    }
    parsed.usage_value = std::move(parsed_usage.value());
  }
  return result<output_t>::success(std::move(parsed));
}

template <typename output_t>
[[nodiscard]] auto parse_single_object(std::string_view json_text,
                                       auto &&parser) -> result<output_t> {
  auto document = parse_json(json_text);
  if (document.has_error()) {
    return result<output_t>::failure(document.error());
  }
  return parser(document.value());
}

} // namespace

auto parse_request(std::string_view json_text) -> core::result<request> {
  return parse_single_object<request>(json_text, [](const core::json_value &root) {
    if (!root.IsObject()) {
      return result<request>::failure(
          make_error(errc::type_mismatch, "completion request must be an object"));
    }
    request parsed{};
    auto raw = json_to_string(root);
    if (raw.has_error()) {
      return result<request>::failure(raw.error());
    }
    parsed.raw_json = std::move(raw.value());
    if (auto field = core::required_string(root, "model"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.model = std::move(field.value());
    }
    const auto *prompt = json_member(root, "prompt");
    if (prompt == nullptr) {
      return result<request>::failure(
          make_error(errc::not_found, "missing required field 'prompt'"));
    }
    auto parsed_prompt = parse_prompt_input_value(*prompt);
    if (parsed_prompt.has_error()) {
      return result<request>::failure(parsed_prompt.error());
    }
    parsed.prompt = std::move(parsed_prompt.value());
    if (auto field = core::optional_int64(root, "best_of"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.best_of = field.value();
    }
    if (auto field = core::optional_bool(root, "echo"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.echo = field.value();
    }
    if (auto field = core::optional_double(root, "frequency_penalty");
        field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.frequency_penalty = field.value();
    }
    const auto *logit_bias = json_member(root, "logit_bias");
    if (logit_bias != nullptr && !logit_bias->IsNull()) {
      auto converted = core::json_to_value(*logit_bias);
      if (converted.has_error()) {
        return result<request>::failure(converted.error());
      }
      parsed.logit_bias = std::move(converted.value());
    }
    if (auto field = core::optional_int64(root, "logprobs"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.logprobs = field.value();
    }
    if (auto field = core::optional_int64(root, "max_tokens"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.max_tokens = field.value();
    }
    if (auto field = core::optional_int64(root, "n"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.n = field.value();
    }
    if (auto field = core::optional_double(root, "presence_penalty");
        field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.presence_penalty = field.value();
    }
    if (auto field = core::optional_int64(root, "seed"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.seed = field.value();
    }
    const auto *stop = json_member(root, "stop");
    if (stop != nullptr) {
      auto parsed_stop = parse_stop_sequence_value(*stop);
      if (parsed_stop.has_error()) {
        return result<request>::failure(parsed_stop.error());
      }
      parsed.stop = std::move(parsed_stop.value());
    }
    if (auto field = core::optional_bool(root, "stream"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.stream = field.value();
    }
    const auto *stream_options_object = json_member(root, "stream_options");
    if (stream_options_object != nullptr && !stream_options_object->IsNull()) {
      auto parsed_options = parse_stream_options_value(*stream_options_object);
      if (parsed_options.has_error()) {
        return result<request>::failure(parsed_options.error());
      }
      parsed.stream_options_value = std::move(parsed_options.value());
    }
    if (auto field = core::optional_string(root, "suffix"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.suffix = std::move(field.value());
    }
    if (auto field = core::optional_double(root, "temperature"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.temperature = field.value();
    }
    if (auto field = core::optional_double(root, "top_p"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.top_p = field.value();
    }
    if (auto field = core::optional_string(root, "user"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.user = std::move(field.value());
    }
    return result<request>::success(std::move(parsed));
  });
}

auto parse_response(std::string_view json_text) -> core::result<response> {
  return parse_single_object<response>(json_text, parse_response_like<response>);
}

auto parse_chunk(std::string_view json_text) -> core::result<chunk> {
  return parse_single_object<chunk>(json_text, parse_response_like<chunk>);
}

} // namespace openai::completions
