#pragma once

#include "openai/completions/types.hpp"
#include "openai/core/result.hpp"
#include "openai/core/serialization.hpp"

namespace openai::completions {

[[nodiscard]] auto serialize_request(const request &value,
                                     const core::serialize_options &options = {})
    -> core::result<std::string>;

[[nodiscard]] auto serialize_response(const response &value,
                                      const core::serialize_options &options = {})
    -> core::result<std::string>;

[[nodiscard]] auto serialize_chunk(const chunk &value,
                                   const core::serialize_options &options = {})
    -> core::result<std::string>;

} // namespace openai::completions
