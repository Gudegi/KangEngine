#include "xform_token.hpp"

namespace KE {
namespace Scene {

const Token XformTokens::translate("xformOp:translate");
const Token XformTokens::rotateXYZ("xformOp:rotateXYZ");
const Token XformTokens::rotateQuat("xformOp:rotateQuaternion");
const Token XformTokens::scale("xformOp:scale");
const Token XformTokens::opOrder("xformOpOrder");
const std::vector<Token> XformTokens::defaultOpOrder = {
    XformTokens::scale, XformTokens::rotateQuat, XformTokens::translate};

XformOpType XformTokens::getXformOpType(const Token& token) {
    if (token == XformTokens::translate)
        return XformOpType::Translate;
    if (token == XformTokens::rotateQuat)
        return XformOpType::RotateQuat;
    if (token == XformTokens::rotateXYZ)
        return XformOpType::RotateXYZ;
    if (token == XformTokens::scale)
        return XformOpType::Scale;
    return XformOpType::None;
}

} // namespace Scene
} // namespace KE
