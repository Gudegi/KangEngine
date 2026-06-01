#ifndef _PANEL_HPP_
#define _PANEL_HPP_

#include <string>
#include <utility>

namespace KE {

///
/// @brief Base class for named ImGui panels.
///
class Panel {
  private:
    std::string _name;

  public:
    explicit Panel(std::string name) : _name(std::move(name)) {}
    virtual ~Panel() {}
    const std::string& name() const { return _name; }
    virtual void buildPanel() = 0; // This function visualizes the specific
                                   // panels inheriting this class.
};

} // namespace KE

#endif
