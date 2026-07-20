#pragma once

#include "engine/scene/scene_backend.hpp"
#include "utils/types.hpp"

namespace KE::Geometry {

// Primitive mesh construction lives outside the scene graph API. These
// functions create data only; they do not create or mutate scene Prims.
Scene::MeshData createCube(float scale = 1.0f);
Scene::MeshData createPlane(float scale, UpAxis upAxis = UpAxis::Y);
Scene::MeshData createSphere(float radius, int numLongitudes,
                             int numLatitudes);
Scene::MeshData createBox(float xScale, float yScale, float zScale);
Scene::MeshData createCylinder(float radius, float length,
                               UpAxis upAxis = UpAxis::Y, int segments = 32);
Scene::MeshData createArrow(float baseRadius, float baseHeight,
                            UpAxis upAxis = UpAxis::Y,
                            float capRadius = -1.0f,
                            float capHeight = -1.0f, int segments = 32);
Scene::MeshData createCapsule(float radius, float height,
                              UpAxis upAxis = UpAxis::Y, int segments = 32);
Scene::MeshData createCone(float radius, float height,
                           UpAxis upAxis = UpAxis::Y, int segments = 32);

} // namespace KE::Geometry
