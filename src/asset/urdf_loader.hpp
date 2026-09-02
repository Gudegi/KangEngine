#pragma once

#include "asset/articulation_desc.hpp"
#include "asset/import_diagnostics.hpp"
#include "utils/coordinate_system.hpp"

#include <string>

namespace KE {
namespace Asset {

struct URDFImportResult {
    ArticulationDesc articulation;
    ImportDiagnostics diagnostics;
};

class URDFLoader {
  public:
    static URDFImportResult
    parse(const std::string& urdfPath, float scale = 1.0f,
          const std::string& order = "DFS",
          Utils::CoordinateSystem targetCoordinateSystem =
              Utils::CoordinateSystem::ZUpXForward);

    static ArticulationDesc
    load(const std::string& urdfPath, float scale = 1.0f,
         const std::string& order = "DFS",
         Utils::CoordinateSystem targetCoordinateSystem =
             Utils::CoordinateSystem::ZUpXForward);

  private:
    URDFLoader() = default;

    ArticulationDesc _data;
    ImportDiagnostics _diagnostics;

    void parseIntoData(const std::string& urdfPath, float scale,
                       const std::string& order,
                       Utils::CoordinateSystem targetCoordinateSystem);
};

} // namespace Asset
} // namespace KE
