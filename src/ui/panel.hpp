#ifndef _PANEL_HPP_
#define _PANEL_HPP_

namespace KE {

class Panel
{

///
/// @brief Define panel to inherit
///

private:


public:
    Panel(){}
    virtual ~Panel(){}
    virtual void buildPanel() = 0; // This function visualizes the specific panels inheriting this class.
};

} // namespace KE

#endif