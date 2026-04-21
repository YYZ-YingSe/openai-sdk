#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace openai::models {

struct model {
  std::string id{};
  std::string object{};
  std::uint64_t created{};
  std::string owned_by{};
  std::string raw_json{};
};

struct list_response {
  std::string object{};
  std::vector<model> data{};
  std::string raw_json{};
};

} // namespace openai::models
