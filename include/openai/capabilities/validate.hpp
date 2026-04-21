#pragma once

#include "openai/capabilities/types.hpp"
#include "openai/chat_completions/types.hpp"
#include "openai/completions/types.hpp"
#include "openai/compat/types.hpp"
#include "openai/embeddings/types.hpp"
#include "openai/responses/types.hpp"

namespace openai::capabilities {

[[nodiscard]] auto validate_request(const responses::request &request,
                                    const effective_profile &profile,
                                    compat::api_family family =
                                        compat::api_family::responses)
    -> validation_report;

[[nodiscard]] auto validate_request(
    const chat_completions::request &request, const effective_profile &profile,
    compat::api_family family = compat::api_family::chat_completions)
    -> validation_report;

[[nodiscard]] auto validate_request(
    const completions::request &request, const effective_profile &profile,
    compat::api_family family = compat::api_family::completions)
    -> validation_report;

[[nodiscard]] auto validate_request(
    const embeddings::request &request, const effective_profile &profile,
    compat::api_family family = compat::api_family::embeddings)
    -> validation_report;

[[nodiscard]] auto validate_request(
    const compat::request_document &request, const effective_profile &profile,
    const compat::operation_hint &hint = {}) -> validation_report;

} // namespace openai::capabilities
