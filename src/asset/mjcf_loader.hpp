#ifndef _MJCF_LOADER_HPP_
#define _MJCF_LOADER_HPP_

#include "asset/articulation_desc.hpp"
#include "asset/import_diagnostics.hpp"
#include "utils/coordinate_system.hpp"

#include <string>

namespace KE {
namespace Asset {

struct MJCFImportResult {
    ArticulationDesc articulation;
    ImportDiagnostics diagnostics;
};

class MJCFLoader {
  public:
    static MJCFImportResult
    parse(const std::string& mjcfPath, float scale = 1.0f,
          const std::string& order = "DFS",
          Utils::CoordinateSystem targetCoordinateSystem =
              Utils::CoordinateSystem::ZUpXForward);

    // Single-pass parse: returns all data needed for visual + physics.
    static ArticulationDesc
    load(const std::string& mjcfPath, float scale = 1.0f,
         const std::string& order = "DFS",
         Utils::CoordinateSystem targetCoordinateSystem =
             Utils::CoordinateSystem::ZUpXForward);

  private:
    MJCFLoader() = default;

    ArticulationDesc _data;
    ImportDiagnostics _diagnostics;

    void parseIntoData(const std::string& mjcfPath, float scale,
                       const std::string& order,
                       Utils::CoordinateSystem targetCoordinateSystem);
};

} // namespace Asset
} // namespace KE

#endif
