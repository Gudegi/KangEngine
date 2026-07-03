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
    bool _open = true;

  public:
    explicit Panel(std::string name) : _name(std::move(name)) {}
    virtual ~Panel() {}
    const std::string& name() const { return _name; }
    bool isOpen() const { return _open; }
    void setOpen(bool open) { _open = open; }
    bool* openPtr() { return &_open; }
    virtual void buildPanel() = 0; // This function visualizes the specific
                                   // panels inheriting this class.
};

} // namespace KE

#endif
