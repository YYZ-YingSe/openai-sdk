#pragma once

#include <string>

#include "openai/core/result.hpp"
#include "openai/core/value.hpp"

namespace openai::core {

struct serialize_options {
  bool pretty{false};
};

[[nodiscard]] auto serialize_json(const value &current,
                                  const serialize_options &options = {})
    -> result<std::string>;

} // namespace openai::core
