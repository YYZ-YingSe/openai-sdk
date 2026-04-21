#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace openai::wire {

struct header {
  std::string name{};
  std::string value{};
};

struct query_parameter {
  std::string name{};
  std::optional<std::string> value{};
};

struct query_string {
  std::vector<query_parameter> parameters{};
  std::string raw_query{};
};

struct content_disposition {
  std::string type{};
  std::optional<std::string> name{};
  std::optional<std::string> filename{};
  std::map<std::string, std::string> parameters{};
};

struct multipart_part {
  std::vector<header> headers{};
  std::optional<content_disposition> disposition{};
  std::optional<std::string> content_type{};
  std::string body{};
};

struct multipart_body {
  std::string boundary{};
  std::vector<multipart_part> parts{};
  std::string raw_body{};
};

struct binary_payload {
  std::optional<std::string> content_type{};
  std::string data{};
};

} // namespace openai::wire
