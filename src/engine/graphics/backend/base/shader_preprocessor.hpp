#ifndef _SHADER_PREPROCESSOR_HPP_
#define _SHADER_PREPROCESSOR_HPP_

#include <string>
#include <vector>

namespace KE {
namespace Backend {

struct ShaderSourceLoadResult {
    std::string source;
    std::vector<std::string> dependencies;
};

ShaderSourceLoadResult
loadShaderSourceWithDependencies(const std::string& path);
std::string loadShaderSource(const std::string& path);

} // namespace Backend
} // namespace KE

#endif
