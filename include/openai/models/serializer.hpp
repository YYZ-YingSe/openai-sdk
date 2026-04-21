#pragma once

#include "openai/core/result.hpp"
#include "openai/core/serialization.hpp"
#include "openai/models/types.hpp"

namespace openai::models {

[[nodiscard]] auto serialize_model(const model &value,
                                   const core::serialize_options &options = {})
    -> core::result<std::string>;

[[nodiscard]] auto serialize_list_response(
    const list_response &value,
    const core::serialize_options &options = {}) -> core::result<std::string>;

} // namespace openai::models
