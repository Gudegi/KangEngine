#ifndef _XFORM_TOKEN_HPP_
#define _XFORM_TOKEN_HPP_

#include "token.hpp"

namespace KE {
namespace Scene {

enum class XformOpType { Translate, RotateQuat, RotateXYZ, Scale, None };

struct XformTokens {
    static const Token translate;
    static const Token rotateXYZ;
    static const Token rotateQuat;
    static const Token scale;
    static const Token opOrder;
    static const std::vector<Token> defaultOpOrder;

    static XformOpType getXformOpType(const Token& token);
};

} // namespace Scene
} // namespace KE

#endif