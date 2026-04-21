#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/pointer.h>
#include <rapidjson/schema.h>
#include <rapidjson/stringbuffer.h>

namespace openai::test {

struct schema_validation_result {
  bool valid{false};
  std::string message{};
};

inline auto contract_schema_path() -> std::filesystem::path {
  return std::filesystem::path{__FILE__}.parent_path().parent_path() /
         "schemas" / "openai-contracts.schema.json";
}

inline auto read_text_file(const std::filesystem::path &path) -> std::string {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

inline auto validate_contract_schema(std::string_view schema_pointer,
                                     std::string_view instance_json)
    -> schema_validation_result {
  rapidjson::Document schema_root;
  const auto schema_text = read_text_file(contract_schema_path());
  schema_root.Parse(schema_text.c_str(), schema_text.size());
  if (schema_root.HasParseError()) {
    return schema_validation_result{
        false,
        std::string{"failed to parse schema: "} +
            rapidjson::GetParseError_En(schema_root.GetParseError())};
  }

  const std::string pointer_text{schema_pointer};
  rapidjson::Pointer pointer(pointer_text.c_str());
  const auto *schema_value = pointer.Get(schema_root);
  if (schema_value == nullptr) {
    return schema_validation_result{
        false, "schema pointer not found: " + pointer_text};
  }

  rapidjson::Document schema_document;
  schema_document.SetObject();
  auto &allocator = schema_document.GetAllocator();

  if (const auto definitions = schema_root.FindMember("definitions");
      definitions != schema_root.MemberEnd()) {
    rapidjson::Value copied_definitions;
    copied_definitions.CopyFrom(definitions->value, allocator);
    schema_document.AddMember("definitions", std::move(copied_definitions),
                              allocator);
  }

  rapidjson::Value all_of(rapidjson::kArrayType);
  rapidjson::Value selected_schema;
  selected_schema.CopyFrom(*schema_value, allocator);
  all_of.PushBack(std::move(selected_schema), allocator);
  schema_document.AddMember("allOf", std::move(all_of), allocator);

  rapidjson::Document instance_document;
  instance_document.Parse(instance_json.data(), instance_json.size());
  if (instance_document.HasParseError()) {
    return schema_validation_result{
        false,
        std::string{"failed to parse instance JSON: "} +
            rapidjson::GetParseError_En(instance_document.GetParseError())};
  }

  rapidjson::SchemaDocument schema(schema_document);
  rapidjson::SchemaValidator validator(schema);
  if (instance_document.Accept(validator)) {
    return schema_validation_result{true, {}};
  }

  rapidjson::StringBuffer instance_pointer;
  validator.GetInvalidDocumentPointer().StringifyUriFragment(instance_pointer);

  rapidjson::StringBuffer schema_fragment;
  validator.GetInvalidSchemaPointer().StringifyUriFragment(schema_fragment);

  std::string message = "schema validation failed";
  message += " keyword=";
  message += validator.GetInvalidSchemaKeyword();
  message += " schema=";
  message += schema_fragment.GetString();
  message += " instance=";
  message += instance_pointer.GetString();
  return schema_validation_result{false, std::move(message)};
}

} // namespace openai::test
