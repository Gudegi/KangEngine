#include "shader_preprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace KE {
namespace Backend {
namespace {

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open shader file: " +
                                 path.string());
    }
    return std::string(std::istreambuf_iterator<char>(file), {});
}

std::string trimLeft(std::string value) {
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(), [](unsigned char c) {
                    return !std::isspace(c);
                }));
    return value;
}

bool parseIncludePath(const std::string& line, std::string& includePath) {
    const std::string trimmed = trimLeft(line);
    const std::string includeToken = "#import";
    if (trimmed.rfind(includeToken, 0) != 0)
        return false;
    if (trimmed.size() > includeToken.size() &&
        !std::isspace(static_cast<unsigned char>(trimmed[includeToken.size()])))
        return false;

    size_t quoteBegin = trimmed.find('"', includeToken.size());
    if (quoteBegin == std::string::npos)
        return false;

    size_t quoteEnd = trimmed.find('"', quoteBegin + 1);
    if (quoteEnd == std::string::npos)
        return false;

    includePath = trimmed.substr(quoteBegin + 1, quoteEnd - quoteBegin - 1);
    return !includePath.empty();
}

bool containsPath(const std::vector<std::filesystem::path>& stack,
                  const std::filesystem::path& path) {
    return std::find(stack.begin(), stack.end(), path) != stack.end();
}

std::string loadShaderSourceRecursive(const std::filesystem::path& path,
                                      std::vector<std::filesystem::path>& stack) {
    const std::filesystem::path normalized =
        std::filesystem::weakly_canonical(path);
    if (containsPath(stack, normalized)) {
        throw std::runtime_error("Recursive shader include detected: " +
                                 normalized.string());
    }

    stack.push_back(normalized);

    std::stringstream input(readTextFile(normalized));
    std::stringstream output;
    std::string line;

    while (std::getline(input, line)) {
        std::string includePath;
        if (!parseIncludePath(line, includePath)) {
            output << line << '\n';
            continue;
        }

        const std::filesystem::path resolvedInclude =
            normalized.parent_path() / includePath;
        output << loadShaderSourceRecursive(resolvedInclude, stack) << '\n';
    }

    stack.pop_back();
    return output.str();
}

} // namespace

std::string loadShaderSource(const std::string& path) {
    std::vector<std::filesystem::path> stack;
    return loadShaderSourceRecursive(path, stack);
}

} // namespace Backend
} // namespace KE
