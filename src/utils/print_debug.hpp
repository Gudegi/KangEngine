#ifndef _PRINT_DEBUG_HPP_
#define _PRINT_DEBUG_HPP_

#include <fmt/base.h>
#include <string>

namespace KE {

#ifdef KE_DEBUG
class FunctionTracer {
  private:
    std::string _functionName;

  public:
    FunctionTracer(const char* functionName) : _functionName(functionName) {
        fmt::print("[TRACE] >>> Entering: {}\n", _functionName);
    }

    ~FunctionTracer() {
        fmt::print("[TRACE] <<< Exiting:  {}\n", _functionName);
    }
};

// Helper macros for token concatenation
#define KE_CONCAT_IMPL(x, y) x##y
#define KE_CONCAT(x, y) KE_CONCAT_IMPL(x, y)

// unique variable per line
#define KE_TRACE_FUNCTION()                                                    \
    KE::FunctionTracer KE_CONCAT(__tracer__, __LINE__)(__FUNCTION__)
#define KE_TRACE(msg) fmt::print("[DEBUG] {}: {}\n", __FUNCTION__, msg)
#define KE_TRACE_VAR(var)                                                      \
    fmt::print("[DEBUG] {}: {} = {}\n", __FUNCTION__, #var, var)

#else
// Do nothing in Release build.
#define KE_TRACE_FUNCTION()
#define KE_TRACE(msg)
#define KE_TRACE_VAR(var)
#endif

} // namespace KE

#endif // _PRINT_DEBUG_HPP_
