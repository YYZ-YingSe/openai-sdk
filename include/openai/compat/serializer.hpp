#pragma once

#include "openai/capabilities/validate.hpp"
#include "openai/compat/types.hpp"
#include "openai/core/result.hpp"
#include "openai/core/serialization.hpp"

namespace openai::compat {

[[nodiscard]] auto serialize_request_json(
    const request_document &document, const operation_hint &hint = {},
    const core::serialize_options &options = {}) -> core::result<std::string>;

[[nodiscard]] auto serialize_response_json(
    const response_document &document, const operation_hint &hint = {},
    const core::serialize_options &options = {}) -> core::result<std::string>;

[[nodiscard]] auto serialize_document_json(
    const document &document, const operation_hint &hint = {},
    const core::serialize_options &options = {}) -> core::result<std::string>;

[[nodiscard]] auto validate_request(
    const request_document &document,
    const capabilities::effective_profile &profile,
    const operation_hint &hint = {}) -> capabilities::validation_report;

} // namespace openai::compat
