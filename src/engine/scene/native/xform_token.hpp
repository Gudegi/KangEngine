#ifndef _XFORM_TOKEN_HPP_
#define _XFORM_TOKEN_HPP_

#include "token.hpp"
#include <vector>

namespace KE {
namespace Scene {

enum class XformOpType { Translate, RotateQuat, RotateXYZ, Scale, Matrix, None };

struct XformTokens {
    static const Token translate;
    static const Token rotateXYZ;
    static const Token rotateQuat;
    static const Token scale;
    static const Token transform;
    static const Token opOrder;
    static const std::vector<Token> defaultOpOrder;

    static XformOpType getXformOpType(const Token& token);
    static bool isXformAttribute(const Token& token);
};

} // namespace Scene
} // namespace KE

#endif
