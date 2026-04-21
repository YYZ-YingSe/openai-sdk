#pragma once

#include "openai/core/result.hpp"
#include "openai/core/serialization.hpp"
#include "openai/embeddings/types.hpp"

namespace openai::embeddings {

[[nodiscard]] auto serialize_request(const request &value,
                                     const core::serialize_options &options = {})
    -> core::result<std::string>;

[[nodiscard]] auto serialize_response(const response &value,
                                      const core::serialize_options &options = {})
    -> core::result<std::string>;

} // namespace openai::embeddings
