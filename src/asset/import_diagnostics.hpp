#ifndef _IMPORT_DIAGNOSTICS_HPP_
#define _IMPORT_DIAGNOSTICS_HPP_

#include <string>
#include <vector>

namespace KE {

namespace Asset {
// Importers should expose a parse(...) API that returns a format-specific
// Result containing the imported data plus these diagnostics.
struct ImportDiagnostics {
    std::vector<std::string> warnings;
};
} // namespace Asset
} // namespace KE

#endif
