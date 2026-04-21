# OpenAI-Compatible Codec SDK

[English README](./README.md)

`openai-sdk` 是一个面向 OpenAI-compatible inference protocol 的无网络、传输无关的 C++20 SDK。

它只保留一组很窄、很清晰的能力边界：

- `responses`
- `chat_completions`
- `completions`
- `embeddings`
- `models`
- `streaming`
- `wire`
- `capabilities`
- `compat`

## 它做什么

- 解析并序列化保留范围内的 inference family 请求与响应
- 解析并序列化 Responses 流式事件与 SSE 帧
- 基于通用 provider / model capability profile 做请求校验
- 通过 `openai::compat` 提供统一的 parse / serialize / validate 门面
- 对快速演进或未知叶子字段，通过 `core::value` 或 raw JSON 做结构化保留

## 它不做什么

- 不做 HTTP 或 WebSocket 传输
- 不做鉴权、重试、限流或 provider-specific networking
- 不做 retained inference families 之外的 OpenAI 平台资源
- 不做通用 HTTP wire serializer；`wire` 层是 parse-oriented ingress helper

## 架构

- `openai::core`
  通用错误、结果、值类型，以及确定性的 JSON 序列化原语。
- `openai::<family>`
  各 retained inference family 的 typed model 与 parser / serializer 成对实现。
- `openai::streaming`
  SSE framing 与 Responses event envelope。
- `openai::wire`
  query string、headers、multipart、binary payload descriptor 的 parse-oriented helper。
- `openai::capabilities`
  通用 provider / model / effective capability profile 与协议级 request validation。
- `openai::compat`
  针对 retained families 与 generic fallback 的薄门面层。

## 构建

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

主目标：`openai_compat`

## 验证

仓库现在通过三层方式验证 codec 能力：

- parser / serializer 单元测试
- golden fixture round-trip 回归
- 基于 [`schemas/openai-contracts.schema.json`](./schemas/openai-contracts.schema.json) 的 JSON Schema 契约校验

执行 `ctest --test-dir build --output-on-failure` 时，会一并跑完这些验证，包括：

- `responses`
- `chat_completions`
- `completions`
- `embeddings`
- `models`
- `streaming`
- `compat` envelope
- `wire` 解析覆盖

扩展样本语料位于 [`tests/fixtures`](./tests/fixtures)，其中包括：

- 更丰富的 retained family 正向样本
- `compat` 推断与 fallback 样本
- 位于 [`tests/fixtures/invalid`](./tests/fixtures/invalid) 的非法输入语料

## 示例

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

## 说明

- 这个 SDK 是 transport-free 的。
- `responses` 是当前最丰富的 family，并保留了 extension registry / strict parsing hooks。
- `capabilities` 是通用协议元数据，不是某个 consumer 的 workflow policy。
- `compat` 保持很薄，只在 retained families 与 generic envelope 之间做统一门面。
- golden fixtures 与 round-trip 测试位于 [`tests`](./tests)。
