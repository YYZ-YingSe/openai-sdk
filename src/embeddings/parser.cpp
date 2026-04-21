#include "openai/embeddings/parser.hpp"

#include "openai/core/json.hpp"

namespace openai::embeddings {
namespace {

using core::errc;
using core::json_member;
using core::json_string;
using core::json_to_string;
using core::make_error;
using core::parse_json;
using core::result;

[[nodiscard]] auto parse_input_value(const core::json_value &value)
    -> result<embedding_input> {
  if (value.IsString()) {
    return result<embedding_input>::success(embedding_input{json_string(value)});
  }
  if (!value.IsArray()) {
    return result<embedding_input>::failure(make_error(
        errc::type_mismatch, "embedding input must be a string or array"));
  }
  if (value.Empty()) {
    return result<embedding_input>::success(
        embedding_input{std::vector<std::string>{}});
  }

  const auto &first = value[0];
  if (first.IsString()) {
    std::vector<std::string> texts;
    texts.reserve(value.Size());
    for (const auto &entry : value.GetArray()) {
      if (!entry.IsString()) {
        return result<embedding_input>::failure(make_error(
            errc::type_mismatch, "embedding input string array is heterogeneous"));
      }
      texts.push_back(json_string(entry));
    }
    return result<embedding_input>::success(embedding_input{std::move(texts)});
  }

  if (first.IsInt64()) {
    std::vector<std::int64_t> tokens;
    tokens.reserve(value.Size());
    for (const auto &entry : value.GetArray()) {
      if (!entry.IsInt64()) {
        return result<embedding_input>::failure(make_error(
            errc::type_mismatch, "embedding token array is heterogeneous"));
      }
      tokens.push_back(entry.GetInt64());
    }
    return result<embedding_input>::success(embedding_input{std::move(tokens)});
  }

  if (first.IsArray()) {
    std::vector<std::vector<std::int64_t>> batches;
    batches.reserve(value.Size());
    for (const auto &entry : value.GetArray()) {
      if (!entry.IsArray()) {
        return result<embedding_input>::failure(make_error(
            errc::type_mismatch, "embedding input array-of-array is heterogeneous"));
      }
      std::vector<std::int64_t> tokens;
      tokens.reserve(entry.Size());
      for (const auto &token : entry.GetArray()) {
        if (!token.IsInt64()) {
          return result<embedding_input>::failure(make_error(
              errc::type_mismatch, "embedding token arrays must contain integers"));
        }
        tokens.push_back(token.GetInt64());
      }
      batches.push_back(std::move(tokens));
    }
    return result<embedding_input>::success(embedding_input{std::move(batches)});
  }

  return result<embedding_input>::failure(make_error(
      errc::type_mismatch, "unsupported embedding input shape"));
}

[[nodiscard]] auto parse_embedding_value(const core::json_value &value)
    -> result<embedding> {
  if (!value.IsObject()) {
    return result<embedding>::failure(
        make_error(errc::type_mismatch, "embedding must be an object"));
  }
  embedding parsed{};
  if (auto field = core::required_string(value, "object"); field.has_error()) {
    return result<embedding>::failure(field.error());
  } else {
    parsed.object = std::move(field.value());
  }
  if (auto field = core::optional_int64(value, "index"); field.has_error()) {
    return result<embedding>::failure(field.error());
  } else {
    parsed.index = field.value().value_or(0);
  }
  const auto *embedding_value = json_member(value, "embedding");
  if (embedding_value == nullptr || embedding_value->IsNull()) {
    return result<embedding>::failure(
        make_error(errc::not_found, "missing required field 'embedding'"));
  }
  if (embedding_value->IsString()) {
    parsed.values = json_string(*embedding_value);
    return result<embedding>::success(std::move(parsed));
  }
  if (!embedding_value->IsArray()) {
    return result<embedding>::failure(make_error(
        errc::type_mismatch, "embedding must be a string or array of numbers"));
  }
  std::vector<double> values;
  values.reserve(embedding_value->Size());
  for (const auto &entry : embedding_value->GetArray()) {
    if (!entry.IsNumber()) {
      return result<embedding>::failure(make_error(
          errc::type_mismatch, "embedding vector must contain numbers"));
    }
    values.push_back(entry.GetDouble());
  }
  parsed.values = std::move(values);
  return result<embedding>::success(std::move(parsed));
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
          make_error(errc::type_mismatch, "embedding request must be an object"));
    }
    request parsed{};
    auto raw = json_to_string(root);
    if (raw.has_error()) {
      return result<request>::failure(raw.error());
    }
    parsed.raw_json = std::move(raw.value());
    auto input = json_member(root, "input");
    if (input == nullptr || input->IsNull()) {
      return result<request>::failure(
          make_error(errc::not_found, "missing required field 'input'"));
    }
    auto parsed_input = parse_input_value(*input);
    if (parsed_input.has_error()) {
      return result<request>::failure(parsed_input.error());
    }
    parsed.input = std::move(parsed_input.value());
    if (auto field = core::required_string(root, "model"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.model = std::move(field.value());
    }
    if (auto field = core::optional_int64(root, "dimensions"); field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.dimensions = field.value();
    }
    if (auto field = core::optional_string(root, "encoding_format");
        field.has_error()) {
      return result<request>::failure(field.error());
    } else {
      parsed.encoding_format = std::move(field.value());
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
  return parse_single_object<response>(json_text, [](const core::json_value &root) {
    if (!root.IsObject()) {
      return result<response>::failure(
          make_error(errc::type_mismatch, "embedding response must be an object"));
    }
    response parsed{};
    auto raw = json_to_string(root);
    if (raw.has_error()) {
      return result<response>::failure(raw.error());
    }
    parsed.raw_json = std::move(raw.value());
    if (auto field = core::required_string(root, "object"); field.has_error()) {
      return result<response>::failure(field.error());
    } else {
      parsed.object = std::move(field.value());
    }
    if (auto field = core::required_string(root, "model"); field.has_error()) {
      return result<response>::failure(field.error());
    } else {
      parsed.model = std::move(field.value());
    }
    const auto *data = json_member(root, "data");
    if (data == nullptr || !data->IsArray()) {
      return result<response>::failure(
          make_error(errc::type_mismatch, "embedding response data must be an array"));
    }
    parsed.data.reserve(data->Size());
    for (const auto &entry : data->GetArray()) {
      auto parsed_embedding = parse_embedding_value(entry);
      if (parsed_embedding.has_error()) {
        return result<response>::failure(parsed_embedding.error());
      }
      parsed.data.push_back(std::move(parsed_embedding.value()));
    }
    const auto *usage = json_member(root, "usage");
    if (usage == nullptr || !usage->IsObject()) {
      return result<response>::failure(
          make_error(errc::type_mismatch, "embedding response usage must be an object"));
    }
    if (auto field = core::optional_int64(*usage, "prompt_tokens");
        field.has_error()) {
      return result<response>::failure(field.error());
    } else {
      parsed.usage_value.prompt_tokens = field.value().value_or(0);
    }
    if (auto field = core::optional_int64(*usage, "total_tokens");
        field.has_error()) {
      return result<response>::failure(field.error());
    } else {
      parsed.usage_value.total_tokens = field.value().value_or(0);
    }
    return result<response>::success(std::move(parsed));
  });
}

} // namespace openai::embeddings
