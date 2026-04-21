#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace openai::test {

inline auto fixture_root() -> std::filesystem::path {
  return std::filesystem::path{__FILE__}.parent_path() / "fixtures";
}

inline auto read_fixture(std::string_view relative_path) -> std::string {
  const auto path = fixture_root() / std::string{relative_path};
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open fixture: " + path.string());
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

} // namespace openai::test
