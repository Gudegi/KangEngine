///
/// Author Kyungwon Kang, 2024/11
///

#ifndef _BASE_PANEL_HPP_
#define _BASE_PANEL_HPP_
#include "panel.hpp"

namespace KE {

class BasePanel: public Panel
{

/*
    Define basic panel to inherit
*/

private:


public:
    BasePanel();
    ~BasePanel();
    virtual void buildPanel();
};

} // namespace KE

#endif