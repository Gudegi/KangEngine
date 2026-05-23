#ifndef _MJCF_LOADER_HPP_
#define _MJCF_LOADER_HPP_

#include "animation/character_description.hpp"
#include "asset/import_diagnostics.hpp"

#include <string>

namespace KE {
namespace Asset {

struct MJCFImportResult {
    Animation::CharacterData character;
    ImportDiagnostics diagnostics;
};

class MJCFLoader {
  public:
    static MJCFImportResult parse(const std::string& mjcfPath,
                                  float scale = 1.0f,
                                  const std::string& order = "DFS");

    // Single-pass parse: returns all data needed for visual + physics.
    static Animation::CharacterData load(const std::string& mjcfPath,
                                         float scale = 1.0f,
                                         const std::string& order = "DFS");

  private:
    MJCFLoader() = default;

    Animation::CharacterData _data;
    ImportDiagnostics _diagnostics;

    void parseIntoData(const std::string& mjcfPath, float scale,
                       const std::string& order);
};

} // namespace Asset
} // namespace KE

#endif
