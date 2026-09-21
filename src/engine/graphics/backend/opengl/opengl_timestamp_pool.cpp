#include "opengl_timestamp_pool.hpp"
#include <GLFW/glfw3.h>
#include <atomic>
#include <limits>
#include <stdexcept>

namespace KE::Backend {
namespace {
std::atomic<uint64_t> nextGeneration{1};
}
bool OpenGLTimestampPool::supported() {
    if (!glQueryCounter || !glGetQueryObjectiv || !glGetQueryObjectui64v ||
        !glGetQueryiv)
        return false;
    GLint bits = 0;
    glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS, &bits);
    return bits > 0 && bits <= 64;
}
OpenGLTimestampPool::OpenGLTimestampPool(size_t capacity)
    : _queries(capacity * 2), _slots(capacity),
      _thread(std::this_thread::get_id()), _context(glfwGetCurrentContext()),
      _generation(nextGeneration++) {
    if (!_context || !capacity || capacity > 65536 || !supported())
        throw std::runtime_error(
            "OpenGL timestamps unavailable or invalid capacity");
    glGetQueryiv(GL_TIMESTAMP, GL_QUERY_COUNTER_BITS, &_counterBits);
    glGenQueries(static_cast<GLsizei>(_queries.size()), _queries.data());
}
OpenGLTimestampPool::~OpenGLTimestampPool() {
    if (_alive && _thread == std::this_thread::get_id() &&
        glfwGetCurrentContext() == _context)
        glDeleteQueries(static_cast<GLsizei>(_queries.size()), _queries.data());
}
void OpenGLTimestampPool::requireContext() const {
    if (!_alive || _thread != std::this_thread::get_id() ||
        glfwGetCurrentContext() != _context)
        throw std::runtime_error(
            "timestamp pool requires its owning GL context/thread");
}
std::shared_ptr<OpenGLTimestampPool::Lease>
OpenGLTimestampPool::reserve(const ProfilePassOptions& options) {
    // Recording is CPU-only, including reservation. Worker recording has no
    // frame context and deliberately produces no timing samples.
    if (!options.context || !options.context->active() || options.path.empty())
        return {};
    if (!_alive) {
        options.context->gpuSample(options.path,
                                   ProfileSampleStatus::DeviceLost);
        return {};
    }
    for (size_t i = 0; i < _slots.size(); ++i) {
        auto& slot = _slots[i];
        if (slot.occupied)
            continue;
        auto query = options.context->gpuSample(options.path,
                                                ProfileSampleStatus::Pending);
        if (!query)
            return {};
        const uint64_t serial = slot.serial + 1;
        slot = {};
        slot.serial = serial;
        slot.occupied = true;
        ++_pendingCount;
        slot.query = *query;
        slot.reservedFrame = _frameIndex;
        auto lease = std::make_shared<Lease>(Lease{_generation, serial, i});
        slot.lease = lease;
        return lease;
    }
    options.context->gpuSample(options.path,
                               ProfileSampleStatus::CapacityExceeded);
    return {};
}
void OpenGLTimestampPool::write(const std::shared_ptr<Lease>& lease, bool end) {
    if (!lease)
        return;
    requireContext();
    if (lease->generation != _generation || lease->slot >= _slots.size())
        throw std::logic_error("stale timestamp lease");
    auto& slot = _slots[lease->slot];
    // Timed-out unsubmitted commands may still be submitted for rendering.
    if (!slot.occupied || slot.serial != lease->serial || slot.reported)
        return;
    if ((!end && slot.beginWritten) ||
        (end && (!slot.beginWritten || slot.endWritten)))
        throw std::logic_error("invalid timestamp command order");
    glQueryCounter(_queries[lease->slot * 2 + (end ? 1 : 0)], GL_TIMESTAMP);
    if (end)
        slot.endWritten = true;
    else {
        slot.beginWritten = true;
        slot.submittedFrame = _frameIndex;
    }
}
std::vector<ProfileQueryResult> OpenGLTimestampPool::poll(uint64_t frameIndex) {
    _frameIndex = frameIndex;
    if (!_pendingCount)
        return {};
    requireContext();
    std::vector<ProfileQueryResult> results;
    for (size_t i = 0; i < _slots.size(); ++i) {
        auto& slot = _slots[i];
        if (!slot.occupied)
            continue;
        const bool expired = frameIndex >= slot.reservedFrame &&
                             frameIndex - slot.reservedFrame >= 240;
        const bool abandoned = slot.lease.expired() && !slot.endWritten;
        if (!slot.reported && (expired || abandoned)) {
            results.push_back(
                {slot.query, ProfileSampleStatus::Dropped, {}, frameIndex});
            slot.reported = true;
        }
        if (!slot.beginWritten) {
            if (slot.reported) {
                slot.occupied = false;
                --_pendingCount;
            }
            continue;
        }
        if (frameIndex < slot.submittedFrame ||
            frameIndex - slot.submittedFrame < 3)
            continue;
        GLint beginReady = GL_FALSE, endReady = GL_FALSE;
        glGetQueryObjectiv(_queries[2 * i], GL_QUERY_RESULT_AVAILABLE,
                           &beginReady);
        if (slot.endWritten)
            glGetQueryObjectiv(_queries[2 * i + 1], GL_QUERY_RESULT_AVAILABLE,
                               &endReady);
        if (!beginReady || (slot.endWritten && !endReady))
            continue;
        if (!slot.endWritten && !slot.reported)
            continue;
        if (!slot.reported) {
            GLuint64 begin = 0, end = 0;
            glGetQueryObjectui64v(_queries[2 * i], GL_QUERY_RESULT, &begin);
            glGetQueryObjectui64v(_queries[2 * i + 1], GL_QUERY_RESULT, &end);
            const uint64_t mask = _counterBits == 64
                                      ? std::numeric_limits<uint64_t>::max()
                                      : (uint64_t{1} << _counterBits) - 1;
            results.push_back({slot.query, ProfileSampleStatus::Ready,
                               double((end - begin) & mask) / 1000000.0,
                               frameIndex});
        }
        slot.occupied = false;
        --_pendingCount;
    }
    return results;
}
std::vector<ProfileQueryResult> OpenGLTimestampPool::shutdown() {
    std::vector<ProfileQueryResult> results;
    if (!_alive)
        return results;
    for (auto& slot : _slots) {
        if (slot.occupied && !slot.reported)
            results.push_back(
                {slot.query, ProfileSampleStatus::DeviceLost, {}, _frameIndex});
        slot.occupied = false;
    }
    if (_thread == std::this_thread::get_id() &&
        glfwGetCurrentContext() == _context)
        glDeleteQueries(static_cast<GLsizei>(_queries.size()), _queries.data());
    _alive = false;
    _pendingCount = 0;
    return results;
}
} // namespace KE::Backend
