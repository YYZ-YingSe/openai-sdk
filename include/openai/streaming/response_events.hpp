#pragma once

#include <any>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "openai/core/result.hpp"
#include "openai/core/value.hpp"
#include "openai/responses/types.hpp"
#include "openai/streaming/sse.hpp"

namespace openai::streaming {

struct response_event {
  std::string type{};
  std::optional<std::string> response_id{};
  std::optional<std::string> item_id{};
  std::optional<std::string> status{};
  std::optional<std::int64_t> output_index{};
  std::optional<std::int64_t> content_index{};
  std::optional<std::int64_t> item_index{};
  std::optional<std::int64_t> summary_index{};
  std::optional<std::int64_t> sequence_number{};
  std::optional<std::string> delta{};
  std::optional<std::string> text{};
  std::optional<std::string> arguments{};
  std::optional<std::string> code{};
  std::optional<std::string> transcript{};
  std::optional<std::string> partial_image_b64{};
  std::optional<responses::annotation> annotation{};
  std::optional<responses::item> item{};
  std::optional<responses::message_content_part> content_part{};
  std::optional<responses::response> response{};
  std::optional<responses::response_error> error{};
  core::value::object extra_fields{};
  std::any payload{};
  std::string raw_json{};
};

[[nodiscard]] auto parse_response_event_json(std::string_view json_text)
    -> core::result<response_event>;

[[nodiscard]] auto parse_response_event(const sse_event &event)
    -> core::result<response_event>;

[[nodiscard]] auto parse_response_event_stream(std::string_view stream_text)
    -> core::result<std::vector<response_event>>;

} // namespace openai::streaming
