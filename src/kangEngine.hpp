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

#include "engine/core/app/app.hpp"
#include "engine/graphics/camera/camera.hpp"
#include "engine/scene/debug_draw.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/core/ui/base_panel.hpp"
#include "engine/core/ui/panel_manager.hpp"
#include "utils/asset_path.hpp"
#include "utils/print_debug.hpp"
#include "utils/glm_utils.hpp"
#include "asset/mesh_loader.hpp"
#include "engine/core/window/window.hpp"
#include "engine/graphics/material/colors.hpp"

#include "asset/mjcf_loader.hpp"
#include "bridge/physics_bridge.hpp"
#include "bridge/skeleton_bridge.hpp"
#include "bridge/skinned_character_bridge.hpp"

#ifdef KANGENGINE_USE_PHYSX
#include "physics/physics.hpp"
#include "physics/articulation.hpp"
#endif

#endif
