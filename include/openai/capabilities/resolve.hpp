#pragma once

#include "openai/capabilities/types.hpp"

namespace openai::capabilities {

[[nodiscard]] auto resolve(const provider_profile &provider,
                           const model_profile &model) -> effective_profile;

} // namespace openai::capabilities
