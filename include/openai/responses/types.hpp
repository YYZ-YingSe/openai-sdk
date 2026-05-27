#pragma once

#include <any>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "openai/core/value.hpp"

namespace openai::responses {

struct raw_json {
  std::string type{};
  std::string json{};
  core::value data{};
  std::any payload{};
};

struct annotation {
  std::string type{};
  std::optional<std::string> title{};
  std::optional<std::string> url{};
  std::optional<std::string> text{};
  std::optional<std::string> file_id{};
  std::optional<std::string> filename{};
  std::optional<std::int64_t> index{};
  std::optional<std::int64_t> start_index{};
  std::optional<std::int64_t> end_index{};
  std::string raw_json{};
};

struct logprob_token {
  std::string token{};
  std::optional<double> logprob{};
  std::vector<std::int64_t> bytes{};
};

struct input_text_content {
  std::string text{};
};

struct input_image_content {
  std::optional<std::string> image_url{};
  std::optional<std::string> file_id{};
  std::optional<std::string> detail{};
};

struct input_file_content {
  std::optional<std::string> file_id{};
  std::optional<std::string> file_url{};
  std::optional<std::string> filename{};
  std::optional<std::string> file_data{};
};

struct output_text_content {
  std::string text{};
  std::vector<annotation> annotations{};
  std::vector<logprob_token> logprobs{};
};

struct refusal_content {
  std::string refusal{};
};

using message_content_part =
    std::variant<input_text_content, input_image_content, input_file_content,
                 output_text_content, refusal_content, raw_json>;

struct response_text_config {
  std::string type{"text"};
  std::optional<std::string> schema_name{};
  std::optional<std::string> schema_description{};
  std::optional<bool> strict{};
  std::optional<std::string> schema_json{};
  std::optional<std::string> verbosity{};
};

struct reasoning_config {
  std::optional<std::string> effort{};
  std::optional<std::string> generate_summary{};
  std::optional<std::string> summary{};
};

struct tool_choice {
  std::string mode{};
  std::optional<std::string> type{};
  std::optional<std::string> name{};
  std::optional<std::string> server_label{};
  std::optional<std::string> tools_json{};
  std::string raw_json{};
};

struct function_tool {
  std::string name{};
  std::optional<std::string> description{};
  std::optional<bool> strict{};
  std::optional<std::string> parameters_json{};
};

struct file_search_tool {
  std::vector<std::string> vector_store_ids{};
  std::optional<std::int64_t> max_num_results{};
  std::optional<std::string> filters_json{};
  std::optional<std::string> ranking_options_json{};
};

struct tool_search_tool {
  std::optional<std::string> description{};
  std::optional<std::string> filters_json{};
  std::optional<std::string> ranking_options_json{};
  std::optional<std::string> user_location_json{};
  std::optional<std::string> raw_config_json{};
};

struct web_search_tool {
  std::optional<std::string> search_context_size{};
  std::optional<std::string> user_location_json{};
  std::optional<std::string> filters_json{};
};

struct computer_tool {
  std::optional<std::string> environment{};
  std::optional<std::int64_t> display_width{};
  std::optional<std::int64_t> display_height{};
};

struct namespace_tool {
  std::optional<std::string> name{};
  std::optional<std::string> description{};
  std::optional<std::string> config_json{};
};

struct local_shell_tool {
  std::optional<std::string> description{};
};

struct shell_tool {
  std::optional<std::string> description{};
};

struct function_shell_tool {
  std::optional<std::string> description{};
  std::optional<std::string> parameters_json{};
  std::optional<bool> strict{};
};

struct custom_tool {
  std::string name{};
  std::optional<std::string> description{};
  std::optional<std::string> format{};
  std::optional<std::string> grammar_json{};
};

struct image_generation_tool {
  std::optional<std::string> model{};
  std::optional<std::string> background{};
  std::optional<std::string> output_format{};
  std::optional<std::string> quality{};
  std::optional<std::string> size{};
};

struct code_interpreter_tool {
  std::optional<std::string> container_json{};
};

struct mcp_tool {
  std::optional<std::string> server_label{};
  std::optional<std::string> server_url{};
  std::optional<std::string> connector_id{};
  std::vector<std::string> allowed_tools{};
  std::optional<std::string> headers_json{};
  std::optional<std::string> require_approval{};
  std::optional<std::string> authorization_json{};
};

struct apply_patch_tool {};

using tool_definition =
    std::variant<function_tool, file_search_tool, tool_search_tool,
                 web_search_tool, computer_tool, namespace_tool,
                 local_shell_tool, shell_tool, function_shell_tool,
                 custom_tool, image_generation_tool, code_interpreter_tool,
                 mcp_tool, apply_patch_tool, raw_json>;

struct message_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> phase{};
  std::string role{};
  std::vector<message_content_part> content{};
};

struct reasoning_summary_part {
  std::string text{};
};

struct reasoning_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::vector<reasoning_summary_part> summary{};
  std::optional<std::string> encrypted_content{};
};

struct compaction_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> summary_json{};
};

struct item_reference {
  std::optional<std::string> id{};
  std::optional<std::string> item_id{};
};

using tool_output_payload = std::variant<std::string, std::vector<message_content_part>>;

struct function_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::string name{};
  std::string arguments{};
};

struct function_call_output_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::string call_id{};
  std::optional<std::string> output{};
  std::optional<std::string> output_json{};
  std::optional<tool_output_payload> output_payload{};
};

struct function_shell_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::optional<std::string> action_json{};
  std::optional<std::string> result_json{};
};

struct function_shell_call_output_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::string call_id{};
  std::optional<std::string> output_json{};
};

struct file_search_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::vector<std::string> queries{};
  std::optional<std::string> results_json{};
};

struct tool_search_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::vector<std::string> queries{};
  std::optional<std::string> results_json{};
};

struct tool_search_output_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::string call_id{};
  std::optional<std::string> output_json{};
};

struct web_search_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> action_json{};
};

struct computer_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::optional<std::string> action_json{};
  std::optional<std::string> pending_safety_checks_json{};
};

struct computer_call_output_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::string call_id{};
  std::optional<std::string> output_json{};
};

struct local_shell_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::optional<std::string> action_json{};
  std::optional<std::string> output_json{};
};

struct local_shell_call_output_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::optional<std::string> output{};
  std::optional<std::string> output_json{};
};

struct shell_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::optional<std::string> action_json{};
  std::optional<std::string> output_json{};
};

struct shell_call_output_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::optional<std::string> output{};
  std::optional<std::string> output_json{};
};

struct custom_tool_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::string name{};
  std::optional<std::string> input{};
};

struct custom_tool_call_output_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::string call_id{};
  std::optional<std::string> output{};
  std::optional<std::string> output_json{};
  std::optional<tool_output_payload> output_payload{};
};

struct image_generation_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> result_json{};
};

struct code_interpreter_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::optional<std::string> code{};
  std::optional<std::string> outputs_json{};
};

struct code_interpreter_call_output_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::string call_id{};
  std::optional<std::string> output_json{};
};

struct mcp_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> name{};
  std::optional<std::string> arguments{};
  std::optional<std::string> server_label{};
  std::optional<std::string> approval_request_id{};
  std::optional<std::string> output_json{};
  std::optional<std::string> error_json{};
};

struct mcp_list_tools_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> server_label{};
  std::optional<std::string> tools_json{};
  std::optional<std::string> error{};
};

struct mcp_approval_request_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> name{};
  std::optional<std::string> arguments{};
  std::optional<std::string> server_label{};
};

struct mcp_approval_response_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> approval_request_id{};
  std::optional<bool> approve{};
  std::optional<std::string> reason{};
};

struct apply_patch_call_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::optional<std::string> call_id{};
  std::optional<std::string> input{};
  std::optional<std::string> output_json{};
};

struct apply_patch_call_output_item {
  std::optional<std::string> id{};
  std::optional<std::string> status{};
  std::string call_id{};
  std::optional<std::string> output_json{};
};

using item =
    std::variant<message_item, reasoning_item, compaction_item, item_reference,
                 function_call_item, function_call_output_item,
                 function_shell_call_item, function_shell_call_output_item,
                 file_search_call_item, tool_search_call_item,
                 tool_search_output_item, web_search_call_item,
                 computer_call_item, computer_call_output_item,
                 local_shell_call_item, local_shell_call_output_item,
                 shell_call_item, shell_call_output_item,
                 custom_tool_call_item, custom_tool_call_output_item,
                 image_generation_call_item,
                 code_interpreter_call_item,
                 code_interpreter_call_output_item, mcp_call_item,
                 mcp_list_tools_item, mcp_approval_request_item,
                 mcp_approval_response_item, apply_patch_call_item,
                 apply_patch_call_output_item, raw_json>;

using input_payload = std::variant<std::string, std::vector<item>>;
using instructions_payload = std::variant<std::string, std::vector<item>>;

struct context_management_entry {
  std::string type{};
  std::optional<std::int64_t> compact_threshold{};
};

struct stream_options {
  std::optional<bool> include_obfuscation{};
};

struct request {
  std::optional<std::string> model{};
  std::optional<input_payload> input{};
  std::optional<instructions_payload> instructions{};
  std::vector<tool_definition> tools{};
  std::optional<tool_choice> tool_choice_value{};
  std::optional<reasoning_config> reasoning{};
  std::optional<response_text_config> text{};
  std::optional<bool> background{};
  std::optional<bool> parallel_tool_calls{};
  std::optional<bool> store{};
  std::optional<bool> stream{};
  std::optional<stream_options> stream_options_value{};
  std::optional<std::string> previous_response_id{};
  std::optional<std::int64_t> max_output_tokens{};
  std::optional<std::int64_t> max_tool_calls{};
  std::optional<double> temperature{};
  std::optional<double> top_p{};
  std::optional<double> top_logprobs{};
  std::optional<std::string> truncation{};
  std::optional<std::string> prompt_cache_key{};
  std::optional<std::string> prompt_cache_retention{};
  std::optional<std::string> safety_identifier{};
  std::optional<std::string> service_tier{};
  std::vector<context_management_entry> context_management{};
  std::optional<::openai::responses::raw_json> conversation{};
  std::optional<::openai::responses::raw_json> prompt{};
  std::vector<std::string> include{};
  std::map<std::string, std::string> metadata{};
  std::optional<std::string> user{};
  std::string raw_json{};
};

struct response_error {
  std::optional<std::string> type{};
  std::optional<std::string> code{};
  std::optional<std::string> message{};
  std::optional<std::string> param{};
  std::string raw_json{};
};

struct incomplete_details {
  std::optional<std::string> reason{};
};

struct usage_details {
  std::optional<std::int64_t> cached_tokens{};
  std::optional<std::int64_t> reasoning_tokens{};
};

struct usage {
  std::optional<std::int64_t> input_tokens{};
  std::optional<std::int64_t> output_tokens{};
  std::optional<std::int64_t> total_tokens{};
  usage_details input_details{};
  usage_details output_details{};
};

struct response {
  std::optional<std::string> id{};
  std::optional<std::string> object{};
  std::optional<std::string> model{};
  std::optional<std::string> status{};
  std::optional<std::uint64_t> created_at{};
  std::optional<std::uint64_t> completed_at{};
  std::optional<instructions_payload> instructions{};
  std::vector<item> output_items{};
  std::vector<tool_definition> tools{};
  std::optional<tool_choice> tool_choice_value{};
  std::optional<response_error> error{};
  std::optional<incomplete_details> incomplete{};
  std::optional<usage> usage_value{};
  std::optional<response_text_config> text{};
  std::optional<reasoning_config> reasoning{};
  std::optional<bool> background{};
  std::optional<bool> parallel_tool_calls{};
  std::optional<bool> stream{};
  std::optional<stream_options> stream_options_value{};
  std::optional<std::string> previous_response_id{};
  std::optional<std::int64_t> max_output_tokens{};
  std::optional<std::int64_t> max_tool_calls{};
  std::optional<double> temperature{};
  std::optional<double> top_p{};
  std::optional<double> top_logprobs{};
  std::optional<std::string> truncation{};
  std::optional<std::string> prompt_cache_key{};
  std::optional<std::string> prompt_cache_retention{};
  std::optional<std::string> safety_identifier{};
  std::optional<std::string> service_tier{};
  std::optional<::openai::responses::raw_json> conversation{};
  std::optional<::openai::responses::raw_json> prompt{};
  std::map<std::string, std::string> metadata{};
  std::optional<std::string> user{};
  std::string raw_json{};
};

[[nodiscard]] inline auto item_type(const item &value) -> std::string {
  return std::visit(
      []<typename item_t>(const item_t &current) -> std::string {
        using current_t = std::decay_t<item_t>;
        if constexpr (std::is_same_v<current_t, message_item>) {
          return "message";
        } else if constexpr (std::is_same_v<current_t, reasoning_item>) {
          return "reasoning";
        } else if constexpr (std::is_same_v<current_t, compaction_item>) {
          return "compaction";
        } else if constexpr (std::is_same_v<current_t, item_reference>) {
          return "item_reference";
        } else if constexpr (std::is_same_v<current_t, function_call_item>) {
          return "function_call";
        } else if constexpr (std::is_same_v<current_t,
                                            function_call_output_item>) {
          return "function_call_output";
        } else if constexpr (std::is_same_v<current_t,
                                            function_shell_call_item>) {
          return "function_shell_call";
        } else if constexpr (std::is_same_v<
                                 current_t, function_shell_call_output_item>) {
          return "function_shell_call_output";
        } else if constexpr (std::is_same_v<current_t, file_search_call_item>) {
          return "file_search_call";
        } else if constexpr (std::is_same_v<current_t, tool_search_call_item>) {
          return "tool_search_call";
        } else if constexpr (std::is_same_v<current_t,
                                            tool_search_output_item>) {
          return "tool_search_output";
        } else if constexpr (std::is_same_v<current_t, web_search_call_item>) {
          return "web_search_call";
        } else if constexpr (std::is_same_v<current_t, computer_call_item>) {
          return "computer_call";
        } else if constexpr (std::is_same_v<current_t,
                                            computer_call_output_item>) {
          return "computer_call_output";
        } else if constexpr (std::is_same_v<current_t, local_shell_call_item>) {
          return "local_shell_call";
        } else if constexpr (std::is_same_v<current_t,
                                            local_shell_call_output_item>) {
          return "local_shell_call_output";
        } else if constexpr (std::is_same_v<current_t, shell_call_item>) {
          return "shell_call";
        } else if constexpr (std::is_same_v<current_t, shell_call_output_item>) {
          return "shell_call_output";
        } else if constexpr (std::is_same_v<current_t, custom_tool_call_item>) {
          return "custom_tool_call";
        } else if constexpr (std::is_same_v<current_t,
                                            custom_tool_call_output_item>) {
          return "custom_tool_call_output";
        } else if constexpr (std::is_same_v<current_t,
                                            image_generation_call_item>) {
          return "image_generation_call";
        } else if constexpr (std::is_same_v<current_t,
                                            code_interpreter_call_item>) {
          return "code_interpreter_call";
        } else if constexpr (std::is_same_v<
                                 current_t, code_interpreter_call_output_item>) {
          return "code_interpreter_call_output";
        } else if constexpr (std::is_same_v<current_t, mcp_call_item>) {
          return "mcp_call";
        } else if constexpr (std::is_same_v<current_t, mcp_list_tools_item>) {
          return "mcp_list_tools";
        } else if constexpr (std::is_same_v<current_t,
                                            mcp_approval_request_item>) {
          return "mcp_approval_request";
        } else if constexpr (std::is_same_v<current_t,
                                            mcp_approval_response_item>) {
          return "mcp_approval_response";
        } else if constexpr (std::is_same_v<current_t, apply_patch_call_item>) {
          return "apply_patch_call";
        } else if constexpr (std::is_same_v<
                                 current_t, apply_patch_call_output_item>) {
          return "apply_patch_call_output";
        } else {
          return current.type;
        }
      },
      value);
}

[[nodiscard]] inline auto collect_output_text(const response &value)
    -> std::string {
  std::string text;
  for (const auto &entry : value.output_items) {
    const auto *message = std::get_if<message_item>(&entry);
    if (message == nullptr) {
      continue;
    }
    for (const auto &part : message->content) {
      if (const auto *output = std::get_if<output_text_content>(&part);
          output != nullptr) {
        text += output->text;
      }
    }
  }
  return text;
}

} // namespace openai::responses
