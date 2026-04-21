#include <any>
#include <string>
#include <variant>
#include <vector>

#include "openai/openai.hpp"
#include "test_support.hpp"

namespace {

OAI_TEST("parse responses request and response core surface") {
  const std::string request_json = R"json(
{
  "model": "gpt-4.1",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        { "type": "input_text", "text": "Write a haiku." }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "lookup_weather",
      "description": "Look up weather",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {
          "city": { "type": "string" }
        },
        "required": ["city"]
      }
    }
  ],
  "tool_choice": {
    "type": "function",
    "name": "lookup_weather"
  },
  "text": {
    "format": {
      "type": "json_schema",
      "json_schema": {
        "name": "weather_answer",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "summary": { "type": "string" }
          },
          "required": ["summary"]
        }
      }
    }
  },
  "parallel_tool_calls": true,
  "metadata": {
    "trace_id": "abc-123"
  }
}
)json";
  auto request = openai::responses::parse_request(request_json);
  OAI_REQUIRE(request.has_value());
  OAI_REQUIRE(request.value().model.has_value());
  OAI_REQUIRE(request.value().tools.size() == 1U);
  OAI_REQUIRE(request.value().tool_choice_value.has_value());
  OAI_REQUIRE(request.value().text.has_value());

  const std::string response_json = R"json(
{
  "id": "resp_123",
  "object": "response",
  "model": "gpt-4.1",
  "status": "completed",
  "created_at": 1744281600,
  "parallel_tool_calls": true,
  "output": [
    {
      "id": "msg_1",
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Hello world",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 12,
    "output_tokens": 8,
    "total_tokens": 20
  }
}
)json";
  auto response = openai::responses::parse_response(response_json);
  OAI_REQUIRE(response.has_value());
  OAI_REQUIRE(response.value().id.has_value());
  OAI_REQUIRE(response.value().output_items.size() == 1U);
  OAI_REQUIRE(openai::responses::collect_output_text(response.value()) ==
              "Hello world");
  OAI_REQUIRE(response.value().usage_value.has_value());
  OAI_REQUIRE(response.value().usage_value->total_tokens.has_value());
  OAI_REQUIRE(response.value().usage_value->total_tokens.value() == 20);
}

OAI_TEST("parse responses streaming and registry fallback") {
  const std::string stream = R"stream(event: response.created
data: {"type":"response.created","response":{"id":"resp_123","object":"response","status":"in_progress","output":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","response_id":"resp_123","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hel"}

data: [DONE]

)stream";
  auto events = openai::streaming::parse_response_event_stream(stream);
  OAI_REQUIRE(events.has_value());
  OAI_REQUIRE(events.value().size() == 3U);
  OAI_REQUIRE(events.value()[0].type == "response.created");
  OAI_REQUIRE(events.value()[1].delta.has_value());
  OAI_REQUIRE(events.value()[1].delta.value() == "Hel");
  OAI_REQUIRE(events.value()[2].type == "[DONE]");

  openai::responses::registry registry;
  registry.register_item(
      "vendor_x_call",
      [](std::string_view type, const openai::core::json_value &value,
         const openai::responses::parse_options &)
          -> openai::core::result<openai::responses::raw_json> {
        auto raw = openai::responses::make_raw_json(type, value);
        if (raw.has_error()) {
          return openai::core::result<openai::responses::raw_json>::failure(
              raw.error());
        }
        raw.value().payload = std::string{"custom-parser"};
        return raw;
      });

  openai::responses::parse_options options{};
  options.registry = &registry;
  const std::string item_json = R"json(
{
  "type": "vendor_x_call",
  "id": "vx_1",
  "payload": {
    "answer": 42,
    "ok": true
  }
}
)json";
  auto parsed_item = openai::responses::parse_item_json(item_json, options);
  OAI_REQUIRE(parsed_item.has_value());
  const auto *raw = std::get_if<openai::responses::raw_json>(&parsed_item.value());
  OAI_REQUIRE(raw != nullptr);
  OAI_REQUIRE(raw->payload.has_value());
  OAI_REQUIRE(std::any_cast<std::string>(raw->payload) == "custom-parser");
}

OAI_TEST("parse chat completions request response and chunk") {
  const std::string request_json = R"json(
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Hello" },
        {
          "type": "image_url",
          "image_url": { "url": "https://example.com/cat.png", "detail": "high" }
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "lookup_weather",
      "description": "Look up weather",
      "parameters": { "type": "object" }
    }
  ],
  "tool_choice": "auto",
  "max_completion_tokens": 64,
  "parallel_tool_calls": true,
  "stream": true,
  "stream_options": {
    "include_obfuscation": true,
    "include_usage": true
  },
  "stop": ["END"],
  "service_tier": "flex",
  "modalities": ["text"],
  "metadata": { "trace_id": "abc-123" }
}
)json";
  auto request = openai::chat_completions::parse_request(request_json);
  OAI_REQUIRE(request.has_value());
  OAI_REQUIRE(request.value().model.has_value());
  OAI_REQUIRE(request.value().messages.size() == 1U);
  OAI_REQUIRE(request.value().parallel_tool_calls.has_value());
  OAI_REQUIRE(request.value().parallel_tool_calls.value());
  OAI_REQUIRE(request.value().stream_options_value.has_value());
  OAI_REQUIRE(request.value().stream_options_value->include_usage.has_value());
  OAI_REQUIRE(request.value().tools.size() == 1U);

  const std::string completion_json = R"json(
{
  "id": "chatcmpl_123",
  "object": "chat.completion",
  "created": 1744281600,
  "model": "gpt-4o-mini",
  "service_tier": "flex",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello",
        "tool_calls": [
          {
            "id": "call_1",
            "type": "function",
            "function": {
              "name": "lookup_weather",
              "arguments": "{\"city\":\"Paris\"}"
            }
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
)json";
  auto completion = openai::chat_completions::parse_completion(completion_json);
  OAI_REQUIRE(completion.has_value());
  OAI_REQUIRE(completion.value().choices.size() == 1U);
  OAI_REQUIRE(completion.value().service_tier.has_value());
  OAI_REQUIRE(completion.value().choices[0].message_value.has_value());
  OAI_REQUIRE(completion.value().choices[0].message_value->tool_calls.size() == 1U);

  const std::string chunk_json = R"json(
{
  "id": "chatcmpl_123",
  "object": "chat.completion.chunk",
  "created": 1744281601,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",
        "content": "Hel"
      },
      "finish_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 1,
    "total_tokens": 11
  }
}
)json";
  auto chunk = openai::chat_completions::parse_completion_chunk(chunk_json);
  OAI_REQUIRE(chunk.has_value());
  OAI_REQUIRE(chunk.value().choices.size() == 1U);
  OAI_REQUIRE(chunk.value().choices[0].delta.has_value());
  OAI_REQUIRE(chunk.value().usage_value.has_value());
}

OAI_TEST("parse legacy completions request response and chunk") {
  const std::string request_json = R"json(
{
  "model": "gpt-3.5-turbo-instruct",
  "prompt": ["Hello", "World"],
  "best_of": 2,
  "echo": false,
  "max_tokens": 16,
  "n": 1,
  "stop": ["END"],
  "stream": true,
  "stream_options": {
    "include_usage": true
  },
  "temperature": 0.2,
  "top_p": 1.0,
  "user": "user_123"
}
)json";
  auto request = openai::completions::parse_request(request_json);
  OAI_REQUIRE(request.has_value());
  OAI_REQUIRE(request.value().model == "gpt-3.5-turbo-instruct");
  OAI_REQUIRE(std::holds_alternative<std::vector<std::string>>(request.value().prompt));
  OAI_REQUIRE(request.value().stream.has_value());
  OAI_REQUIRE(request.value().stream.value());
  OAI_REQUIRE(request.value().stream_options_value.has_value());

  const std::string response_json = R"json(
{
  "id": "cmpl_1",
  "object": "text_completion",
  "created": 1744281600,
  "model": "gpt-3.5-turbo-instruct",
  "choices": [
    {
      "text": "Hello",
      "index": 0,
      "finish_reason": "stop",
      "logprobs": {
        "text_offset": [0],
        "token_logprobs": [-0.1],
        "tokens": ["Hello"],
        "top_logprobs": [
          { "Hello": -0.1 }
        ]
      }
    }
  ],
  "usage": {
    "completion_tokens": 1,
    "prompt_tokens": 1,
    "total_tokens": 2,
    "completion_tokens_details": {
      "reasoning_tokens": 0
    },
    "prompt_tokens_details": {
      "cached_tokens": 0
    }
  }
}
)json";
  auto response = openai::completions::parse_response(response_json);
  OAI_REQUIRE(response.has_value());
  OAI_REQUIRE(response.value().choices.size() == 1U);
  OAI_REQUIRE(response.value().choices[0].logprobs_value.has_value());
  OAI_REQUIRE(response.value().usage_value.has_value());

  const std::string chunk_json = R"json(
{
  "id": "cmpl_1",
  "object": "text_completion",
  "created": 1744281600,
  "model": "gpt-3.5-turbo-instruct",
  "choices": [
    {
      "text": "Hel",
      "index": 0,
      "finish_reason": null
    }
  ]
}
)json";
  auto chunk = openai::completions::parse_chunk(chunk_json);
  OAI_REQUIRE(chunk.has_value());
  OAI_REQUIRE(chunk.value().choices.size() == 1U);
  OAI_REQUIRE(!chunk.value().choices[0].finish_reason.has_value());
}

OAI_TEST("parse embeddings models and wire helpers") {
  const std::string embedding_request_json = R"json(
{
  "input": ["hello", "world"],
  "model": "text-embedding-3-large",
  "dimensions": 1024,
  "encoding_format": "float"
}
)json";
  auto embedding_request = openai::embeddings::parse_request(embedding_request_json);
  OAI_REQUIRE(embedding_request.has_value());
  OAI_REQUIRE(embedding_request.value().model == "text-embedding-3-large");

  const std::string embedding_response_json = R"json(
{
  "object": "list",
  "model": "text-embedding-3-large",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, 0.3]
    }
  ],
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
)json";
  auto embedding_response = openai::embeddings::parse_response(embedding_response_json);
  OAI_REQUIRE(embedding_response.has_value());
  OAI_REQUIRE(embedding_response.value().data.size() == 1U);

  const std::string model_list_json = R"json(
{
  "object": "list",
  "data": [
    { "id": "gpt-4.1", "object": "model", "created": 1744281600, "owned_by": "openai" }
  ]
}
)json";
  auto model_list = openai::models::parse_list_response(model_list_json);
  OAI_REQUIRE(model_list.has_value());
  OAI_REQUIRE(model_list.value().data.size() == 1U);

  auto query = openai::wire::parse_query_string(
      "?model=gpt-4.1&tag=one&tag=two&empty=&flag&prompt=hello+world");
  OAI_REQUIRE(query.has_value());
  OAI_REQUIRE(query.value().parameters.size() == 6U);
  OAI_REQUIRE(query.value().parameters[1].name == "tag");
  OAI_REQUIRE(query.value().parameters[1].value.has_value());
  OAI_REQUIRE(query.value().parameters[2].value.has_value());
  OAI_REQUIRE(!query.value().parameters[4].value.has_value());

  auto headers = openai::wire::parse_headers(
      "Content-Type: application/json\r\nX-Test: 1\r\n");
  OAI_REQUIRE(headers.has_value());
  OAI_REQUIRE(headers.value().size() == 2U);

  const std::string multipart = R"multipart(--abc123
Content-Disposition: form-data; name="purpose"

assistants
--abc123
Content-Disposition: form-data; name="file"; filename="demo.txt"
Content-Type: text/plain

hello
--abc123--
)multipart";
  auto body = openai::wire::parse_multipart_body(multipart, "abc123");
  OAI_REQUIRE(body.has_value());
  OAI_REQUIRE(body.value().parts.size() == 2U);
  OAI_REQUIRE(body.value().parts[1].content_type.has_value());

  auto binary = openai::wire::parse_binary_payload("abc", "text/plain");
  OAI_REQUIRE(binary.has_value());
  OAI_REQUIRE(binary.value().content_type.has_value());
}

OAI_TEST("compat router routes retained request and response families") {
  openai::compat::operation_hint responses_request_hint{};
  responses_request_hint.family = openai::compat::api_family::responses;
  auto responses_request = openai::compat::parse_request_json(
      R"json({"model":"gpt-4.1","input":"hello"})json", responses_request_hint);
  OAI_REQUIRE(responses_request.has_value());
  OAI_REQUIRE(std::get_if<openai::responses::request>(&responses_request.value()) !=
              nullptr);

  openai::compat::operation_hint chat_request_hint{};
  chat_request_hint.family = openai::compat::api_family::chat_completions;
  auto chat_request = openai::compat::parse_request_json(
      R"json({"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]})json",
      chat_request_hint);
  OAI_REQUIRE(chat_request.has_value());
  OAI_REQUIRE(std::get_if<openai::chat_completions::request>(&chat_request.value()) !=
              nullptr);

  openai::compat::operation_hint completions_request_hint{};
  completions_request_hint.family = openai::compat::api_family::completions;
  auto completions_request = openai::compat::parse_request_json(
      R"json({"model":"gpt-3.5-turbo-instruct","prompt":"hello"})json",
      completions_request_hint);
  OAI_REQUIRE(completions_request.has_value());
  OAI_REQUIRE(std::get_if<openai::completions::request>(
                  &completions_request.value()) != nullptr);

  openai::compat::operation_hint chat_chunk_hint{};
  chat_chunk_hint.family = openai::compat::api_family::chat_completions;
  chat_chunk_hint.streaming = true;
  auto chat_chunk = openai::compat::parse_response_json(
      R"json({"id":"chatcmpl_1","object":"chat.completion.chunk","created":1744281601,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]})json",
      chat_chunk_hint);
  OAI_REQUIRE(chat_chunk.has_value());
  OAI_REQUIRE(std::get_if<openai::chat_completions::completion_chunk>(
                  &chat_chunk.value()) != nullptr);

  openai::compat::operation_hint completions_chunk_hint{};
  completions_chunk_hint.family = openai::compat::api_family::completions;
  completions_chunk_hint.streaming = true;
  auto completions_chunk = openai::compat::parse_response_json(
      R"json({"id":"cmpl_1","object":"text_completion","created":1744281600,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"Hel","index":0,"finish_reason":null}]})json",
      completions_chunk_hint);
  OAI_REQUIRE(completions_chunk.has_value());
  OAI_REQUIRE(std::get_if<openai::completions::chunk>(
                  &completions_chunk.value()) != nullptr);

  auto detected_response = openai::compat::parse_document_json(
      R"json({"id":"resp_123","object":"response","status":"completed","output":[]})json");
  OAI_REQUIRE(detected_response.has_value());
  OAI_REQUIRE(std::get_if<openai::responses::response>(&detected_response.value()) !=
              nullptr);

  auto detected_text_completion = openai::compat::parse_document_json(
      R"json({"id":"cmpl_1","object":"text_completion","created":1744281600,"model":"gpt-3.5-turbo-instruct","choices":[{"text":"Hello","index":0,"finish_reason":"stop"}]})json");
  OAI_REQUIRE(detected_text_completion.has_value());
  OAI_REQUIRE(std::get_if<openai::completions::response>(
                  &detected_text_completion.value()) != nullptr);

  auto detected_model_list = openai::compat::parse_document_json(
      R"json({"object":"list","data":[{"id":"gpt-4.1","object":"model","created":1744281600,"owned_by":"openai"}]})json");
  OAI_REQUIRE(detected_model_list.has_value());
  OAI_REQUIRE(std::get_if<openai::models::list_response>(
                  &detected_model_list.value()) != nullptr);

  auto detected_error = openai::compat::parse_document_json(
      R"json({"error":{"type":"invalid_request_error","code":"bad_request","message":"oops","param":"model"}})json");
  OAI_REQUIRE(detected_error.has_value());
  OAI_REQUIRE(std::get_if<openai::compat::error_envelope>(
                  &detected_error.value()) != nullptr);
}

} // namespace
