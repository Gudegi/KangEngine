///
/// Author Kyungwon Kang, 2024/11
///
#ifndef _KANGENGINE_
#define _KANGENGINE_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "app/app.hpp"
#include "camera/camera.hpp"
#include "scene/native/prim.hpp"
#include "ui/base_panel.hpp"
#include "ui/panel_manager.hpp"
#include "utils/print_debug.hpp"
#include "utils/glm_utils.hpp"
#include "window/window.hpp"
#include "material/colors.hpp"

#ifndef KANGENGINE_USE_PHYSX
#include "physics/physics.hpp"
#endif

#endif