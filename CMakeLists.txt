cmake_minimum_required(VERSION 3.25)

project(openai_compat_sdk VERSION 1.0.0 LANGUAGES CXX)

include(cmake/openai_compat_options.cmake)

add_library(openai_compat
  src/capabilities/resolve.cpp
  src/capabilities/validate.cpp
  src/core/serialization.cpp
  src/chat_completions/parser.cpp
  src/chat_completions/serializer.cpp
  src/completions/parser.cpp
  src/completions/serializer.cpp
  src/compat/parser.cpp
  src/compat/serializer.cpp
  src/embeddings/parser.cpp
  src/embeddings/serializer.cpp
  src/models/parser.cpp
  src/models/serializer.cpp
  src/responses/parser.cpp
  src/responses/serializer.cpp
  src/streaming/response_events.cpp
  src/streaming/serializer.cpp
  src/streaming/sse.cpp
  src/wire/parser.cpp)

target_compile_features(openai_compat PUBLIC cxx_std_20)
target_include_directories(openai_compat
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rapidjson/include")

if(MSVC)
  target_compile_options(openai_compat PRIVATE /W4)
else()
  target_compile_options(openai_compat PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(OPENAI_COMPAT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
