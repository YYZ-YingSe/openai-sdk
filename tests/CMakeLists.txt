add_executable(openai_compat_tests
  test_main.cpp
  openai_capabilities_test.cpp
  openai_compat_test.cpp
  openai_compat_codec_test.cpp
  openai_chat_completions_codec_test.cpp
  openai_contract_schema_test.cpp
  openai_completions_codec_test.cpp
  openai_core_serialization_test.cpp
  openai_embeddings_models_codec_test.cpp
  openai_fixture_corpus_test.cpp
  openai_responses_codec_test.cpp
  openai_roundtrip_test.cpp
  openai_streaming_codec_test.cpp
  openai_wire_parser_test.cpp)

target_compile_features(openai_compat_tests PRIVATE cxx_std_20)
target_include_directories(openai_compat_tests PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(openai_compat_tests PRIVATE openai_compat)

add_test(NAME openai_compat_tests COMMAND openai_compat_tests)
