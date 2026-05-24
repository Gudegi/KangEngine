#ifndef _IMPORT_DIAGNOSTICS_HPP_
#define _IMPORT_DIAGNOSTICS_HPP_

#include <iostream>
#include <string>
#include <vector>

namespace KE {

namespace Asset {
// Importers should expose a parse(...) API that returns a format-specific
// Result containing the imported data plus these diagnostics.
struct ImportDiagnostics {
    std::vector<std::string> warnings;

    bool hasWarnings() const { return !warnings.empty(); }

    void printWarnings(const std::string& context,
                       std::ostream& os = std::cerr) const {
        if (warnings.empty())
            return;
        os << context << " warnings (" << warnings.size() << "):\n";
        for (const std::string& warning : warnings)
            os << "  warning: " << warning << '\n';
    }
};
} // namespace Asset
} // namespace KE

#endif
