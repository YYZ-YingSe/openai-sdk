#include "openai/capabilities/resolve.hpp"

#include <algorithm>
#include <iterator>

namespace openai::capabilities {
namespace {

auto merge_family_capabilities(const family_capabilities &lhs,
                               const family_capabilities &rhs)
    -> family_capabilities {
  return family_capabilities{
      min_support_level(lhs.request_support, rhs.request_support),
      min_support_level(lhs.response_support, rhs.response_support),
      min_support_level(lhs.streaming, rhs.streaming),
      min_support_level(lhs.tools, rhs.tools),
      min_support_level(lhs.structured_outputs, rhs.structured_outputs),
      min_support_level(lhs.parallel_tool_calls, rhs.parallel_tool_calls),
  };
}

auto merge_tool_capabilities(const tool_capabilities &left,
                             const tool_capabilities &right)
    -> tool_capabilities {
  return tool_capabilities{
      min_support_level(left.function_tools, right.function_tools),
      min_support_level(left.file_search, right.file_search),
      min_support_level(left.tool_search, right.tool_search),
      min_support_level(left.web_search, right.web_search),
      min_support_level(left.computer_use, right.computer_use),
      min_support_level(left.namespace_tools, right.namespace_tools),
      min_support_level(left.local_shell, right.local_shell),
      min_support_level(left.shell, right.shell),
      min_support_level(left.function_shell, right.function_shell),
      min_support_level(left.custom_tools, right.custom_tools),
      min_support_level(left.image_generation, right.image_generation),
      min_support_level(left.code_interpreter, right.code_interpreter),
      min_support_level(left.mcp, right.mcp),
      min_support_level(left.apply_patch, right.apply_patch),
  };
}

auto merge_modality_capabilities(const modality_capabilities &left,
                                 const modality_capabilities &right)
    -> modality_capabilities {
  return modality_capabilities{
      min_support_level(left.text_input, right.text_input),
      min_support_level(left.text_output, right.text_output),
      min_support_level(left.image_input, right.image_input),
      min_support_level(left.image_output, right.image_output),
      min_support_level(left.audio_input, right.audio_input),
      min_support_level(left.audio_output, right.audio_output),
  };
}

auto merge_structured_output_capabilities(
    const structured_output_capabilities &left,
    const structured_output_capabilities &right)
    -> structured_output_capabilities {
  return structured_output_capabilities{
      min_support_level(left.text, right.text),
      min_support_level(left.json_schema, right.json_schema),
      min_support_level(left.strict, right.strict),
  };
}

auto merge_sampling_capabilities(const sampling_capabilities &left,
                                 const sampling_capabilities &right)
    -> sampling_capabilities {
  return sampling_capabilities{
      min_support_level(left.temperature, right.temperature),
      min_support_level(left.top_p, right.top_p),
      min_support_level(left.n, right.n),
      min_support_level(left.stop, right.stop),
      min_support_level(left.logprobs, right.logprobs),
      min_support_level(left.top_logprobs, right.top_logprobs),
      min_support_level(left.seed, right.seed),
      min_support_level(left.best_of, right.best_of),
      min_support_level(left.max_tokens, right.max_tokens),
      min_support_level(left.max_completion_tokens,
                        right.max_completion_tokens),
      min_support_level(left.max_output_tokens, right.max_output_tokens),
  };
}

auto intersect_supported_families(const provider_profile &provider,
                                  const model_profile &model)
    -> std::set<compat::api_family> {
  if (provider.supported_families.empty()) {
    return model.supported_families;
  }
  if (model.supported_families.empty()) {
    return provider.supported_families;
  }
  std::set<compat::api_family> result{};
  std::set_intersection(provider.supported_families.begin(),
                        provider.supported_families.end(),
                        model.supported_families.begin(),
                        model.supported_families.end(),
                        std::inserter(result, result.end()));
  return result;
}

auto merge_limit(const std::optional<std::int64_t> &left,
                 const std::optional<std::int64_t> &right)
    -> std::optional<std::int64_t> {
  if (left.has_value() && right.has_value()) {
    return std::min(*left, *right);
  }
  if (left.has_value()) {
    return left;
  }
  return right;
}

} // namespace

auto resolve(const provider_profile &provider, const model_profile &model)
    -> effective_profile {
  effective_profile effective{};
  effective.supported_families = intersect_supported_families(provider, model);
  effective.responses =
      merge_family_capabilities(provider.responses, model.responses);
  effective.chat_completions = merge_family_capabilities(
      provider.chat_completions, model.chat_completions);
  effective.completions =
      merge_family_capabilities(provider.completions, model.completions);
  effective.embeddings =
      merge_family_capabilities(provider.embeddings, model.embeddings);
  effective.models = merge_family_capabilities(provider.models, model.models);
  effective.tools = merge_tool_capabilities(provider.tools, model.tools);
  effective.modalities =
      merge_modality_capabilities(provider.modalities, model.modalities);
  effective.structured_outputs = merge_structured_output_capabilities(
      provider.structured_outputs, model.structured_outputs);
  effective.sampling =
      merge_sampling_capabilities(provider.sampling, model.sampling);
  effective.max_tools = merge_limit(provider.max_tools, model.max_tools);
  effective.max_stop_sequences =
      merge_limit(provider.max_stop_sequences, model.max_stop_sequences);
  effective.max_n = merge_limit(provider.max_n, model.max_n);
  effective.max_top_logprobs =
      merge_limit(provider.max_top_logprobs, model.max_top_logprobs);
  effective.max_dimensions =
      merge_limit(provider.max_dimensions, model.max_dimensions);
  effective.raw_passthrough =
      min_support_level(provider.raw_passthrough, model.raw_passthrough);
  return effective;
}

} // namespace openai::capabilities
