#pragma once
#include "engine/graphics/backend/base/profile_types.hpp"
#include <glad/glad.h>

struct GLFWwindow;
namespace KE::Backend {

// Only this backend object touches GL query names. Leases retained by recorded
// commands contain logical identities; destroying a lease never calls OpenGL.
class OpenGLTimestampPool {
  public:
    struct Lease {
        uint64_t generation = 0, serial = 0;
        size_t slot = 0;
    };
    static constexpr size_t DefaultCapacity = 256; // Begin/end query pairs.
    explicit OpenGLTimestampPool(size_t capacity = DefaultCapacity);
    ~OpenGLTimestampPool();
    static bool supported();
    std::shared_ptr<Lease> reserve(const ProfilePassOptions& options);
    void write(const std::shared_ptr<Lease>& lease, bool end);
    std::vector<ProfileQueryResult> poll(uint64_t frameIndex);
    std::vector<ProfileQueryResult> shutdown();

  private:
    struct Slot {
        bool occupied = false, beginWritten = false, endWritten = false,
             reported = false;
        uint64_t serial = 0, reservedFrame = 0, submittedFrame = 0;
        ProfileQuery query;
        std::weak_ptr<Lease> lease;
    };
    void requireContext() const;
    std::vector<GLuint> _queries;
    std::vector<Slot> _slots;
    std::thread::id _thread;
    GLFWwindow* _context = nullptr;
    uint64_t _generation = 0, _frameIndex = 0;
    GLint _counterBits = 0;
    bool _alive = true;
    size_t _pendingCount = 0;
};
} // namespace KE::Backend
