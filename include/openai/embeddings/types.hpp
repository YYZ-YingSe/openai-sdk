#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace openai::embeddings {

using embedding_input = std::variant<std::string, std::vector<std::string>,
                                     std::vector<std::int64_t>,
                                     std::vector<std::vector<std::int64_t>>>;

struct request {
  embedding_input input{};
  std::string model{};
  std::optional<std::int64_t> dimensions{};
  std::optional<std::string> encoding_format{};
  std::optional<std::string> user{};
  std::string raw_json{};
};

using embedding_vector = std::variant<std::vector<double>, std::string>;

struct embedding {
  std::string object{};
  std::int64_t index{};
  embedding_vector values{};
};

struct usage {
  std::int64_t prompt_tokens{};
  std::int64_t total_tokens{};
};

struct response {
  std::string object{};
  std::string model{};
  std::vector<embedding> data{};
  usage usage_value{};
  std::string raw_json{};
};

} // namespace openai::embeddings
