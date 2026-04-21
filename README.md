# OpenAI-Compatible Codec SDK

[中文说明](./README.zh-CN.md)

`openai-sdk` is a transport-free C++20 SDK for OpenAI-compatible inference protocols.

It keeps a narrow surface:

- `responses`
- `chat_completions`
- `completions`
- `embeddings`
- `models`
- `streaming`
- `wire`
- `capabilities`
- `compat`

## What It Does

- Parse and serialize retained inference family payloads
- Parse and serialize Responses streaming events and SSE frames
- Validate requests against generic provider/model capability profiles
- Route retained families through a thin `compat` facade
- Preserve fast-moving leaves through `core::value` or raw JSON where needed

## What It Does Not Do

- HTTP or WebSocket transport
- Authentication or retries
- Provider-specific networking
- Broad OpenAI platform coverage outside retained inference families
- Generic HTTP wire formatting beyond the retained parse-oriented `wire` layer

## Shape

- `openai::core`
  Shared result, error, value, and JSON serialization primitives.
- `openai::<family>`
  Typed request/response/chunk/event models with parser/serializer pairs.
- `openai::streaming`
  SSE framing and Responses event envelopes.
- `openai::wire`
  Parse-oriented helpers for query strings, headers, multipart bodies, and binary payload descriptors.
- `openai::capabilities`
  Generic provider/model/effective profiles and protocol-level validation.
- `openai::compat`
  Thin parse/serialize/validate facade across retained families and generic fallbacks.

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Primary target: `openai_compat`

## Validation

The repository now verifies the codec surface in three layers:

- parser and serializer unit tests
- golden fixture round-trip tests
- JSON Schema contract validation for retained documents under [`schemas/openai-contracts.schema.json`](./schemas/openai-contracts.schema.json)

`ctest --test-dir build --output-on-failure` runs all of them, including
contract validation for:

- `responses`
- `chat_completions`
- `completions`
- `embeddings`
- `models`
- `streaming`
- `compat` envelopes
- `wire` parser coverage

Expanded sample corpora live under [`tests/fixtures`](./tests/fixtures),
including:

- richer valid retained-family fixtures
- inferred `compat` fixtures
- malformed payload corpus under [`tests/fixtures/invalid`](./tests/fixtures/invalid)

## Example

```cpp
#include "openai/openai.hpp"

auto main() -> int {
  openai::chat_completions::request request{};
  request.model = "gpt-4o-mini";

  openai::chat_completions::message message{};
  message.role = "user";
  message.content = std::string{"Hello"};
  request.messages.push_back(std::move(message));

  openai::compat::request_document document{request};
  openai::compat::operation_hint hint{};
  hint.family = openai::compat::api_family::chat_completions;

  const auto json = openai::compat::serialize_request_json(document, hint);
  if (json.has_error()) {
    return 1;
  }

  const auto reparsed = openai::compat::parse_request_json(json.value(), hint);
  return reparsed.has_error() ? 1 : 0;
}
```
