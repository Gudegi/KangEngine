#ifndef _MJCF_LOADER_HPP_
#define _MJCF_LOADER_HPP_

#include "character_description.hpp"

#include <string>

namespace KE {
namespace Animation {

class MJCFLoader {
  public:
    // Single-pass parse: returns all data needed for visual + physics.
    static CharacterData load(const std::string& mjcfPath, float scale = 1.0f,
                              const std::string& order = "DFS");

  private:
    CharacterData _data;

    void parse(const std::string& mjcfPath, float scale,
               const std::string& order);
};

} // namespace Animation
} // namespace KE

#endif
