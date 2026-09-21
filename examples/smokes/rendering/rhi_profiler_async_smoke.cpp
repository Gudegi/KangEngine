#include "engine/core/window/window.hpp"
#include "engine/graphics/backend/opengl/opengl_device.hpp"
#include "engine/graphics/backend/opengl/opengl_timestamp_pool.hpp"
#include "engine/graphics/renderer/renderer_profiler.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace KE;
using namespace KE::Backend;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
int main() {
    Window window;
    window.init(64, 64, true);
    require(window.getGlfwWindow(), "hidden GL context unavailable");
    window.setVSync(false);
    OpenGLDevice device;
    device.initialize();
    if (!device.profilerCapabilities().passTimestamps) {
        std::cout << "SKIP: OpenGL timestamp queries unavailable\n";
        return 0;
    }
    auto context = device.profileContext();
    OpenGLTimestampPool pool(2);
    context->beginFrame(1, 0, BackendType::OpenGL);
    auto first = pool.reserve({context.get(), "render/first"});
    auto abandoned = pool.reserve({context.get(), "render/abandoned"});
    auto overflow = pool.reserve({context.get(), "render/overflow"});
    require(first && abandoned && !overflow, "query capacity not enforced");
    pool.write(first, false);
    glClear(GL_COLOR_BUFFER_BIT);
    pool.write(first, true);
    const auto initial = context->endFrame();
    require(!initial->finalized && initial->samples[2].status ==
                                       ProfileSampleStatus::CapacityExceeded,
            "pending/overflow snapshot contract");
    abandoned.reset();
    const auto dropped = pool.poll(0);
    require(dropped.size() == 1 &&
                dropped[0].status == ProfileSampleStatus::Dropped,
            "abandoned reservation not reclaimed");
    require(pool.poll(1).empty() && pool.poll(2).empty(),
            "GPU result read before minimum latency");

    // Deterministic delayed driver result: no blocking read may be attempted.
    auto getAvailable = glad_glGetQueryObjectiv;
    auto getResult = glad_glGetQueryObjectui64v;
    static int blockingReads = 0;
    glad_glGetQueryObjectiv = [](GLuint, GLenum, GLint* ready) {
        *ready = GL_FALSE;
    };
    glad_glGetQueryObjectui64v = [](GLuint, GLenum, GLuint64*) {
        ++blockingReads;
    };
    const auto delayed = pool.poll(3);
    glad_glGetQueryObjectiv = getAvailable;
    glad_glGetQueryObjectui64v = getResult;
    require(delayed.empty() && blockingReads == 0,
            "unready query triggered a blocking result read");

    bool ready = false;
    uint64_t frame = 4;
    for (; frame < 200 && !ready; ++frame) {
        glfwSwapBuffers(window.getGlfwWindow());
        glfwPollEvents();
        for (const auto& result : pool.poll(frame)) {
            require(result.query.frameIndex == 0 && result.query.captureId == 1,
                    "result attributed to a different frame");
            ready = result.status == ProfileSampleStatus::Ready &&
                    result.durationMs && *result.durationMs >= 0;
        }
        if (!ready)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(ready, "GPU timestamps never became ready");
    require(initial->samples[0].status == ProfileSampleStatus::Pending,
            "old snapshot mutated");

    context->beginFrame(1, frame, BackendType::OpenGL);
    auto replacement = pool.reserve({context.get(), "render/replacement"});
    require(replacement && replacement->serial != first->serial,
            "slot not safely recycled");
    pool.write(first,
               false); // Expired slot identity cannot write into replacement.
    pool.write(replacement, false);
    pool.write(replacement, true);
    context->endFrame();
    const auto lost = pool.shutdown();
    require(lost.size() == 1 &&
                lost[0].status == ProfileSampleStatus::DeviceLost,
            "shutdown did not terminate pending query");

    // A timed-out partial submit is terminal to consumers, but its in-flight
    // query slot must remain quarantined until the driver reports completion.
    OpenGLTimestampPool stalled(1);
    context->beginFrame(2, 0, BackendType::OpenGL);
    auto partial = stalled.reserve({context.get(), "render/partial_submit"});
    stalled.write(partial, false);
    context->endFrame();
    glad_glGetQueryObjectiv = [](GLuint, GLenum, GLint* available) {
        *available = GL_FALSE;
    };
    const auto timeout = stalled.poll(240);
    context->beginFrame(2, 240, BackendType::OpenGL);
    auto unsafeReuse = stalled.reserve({context.get(), "render/unsafe_reuse"});
    context->endFrame();
    glad_glGetQueryObjectiv = getAvailable;
    require(timeout.size() == 1 &&
                timeout[0].status == ProfileSampleStatus::Dropped &&
                !unsafeReuse,
            "timed-out GPU slot was recycled before completion");
    stalled.shutdown();

    // Native bridge: off/worker paths issue no timestamp commands. An active
    // bridge closes on exception, while the aborted frame stays Invalid.
    RendererProfiler profiler;
    profiler.bind(context, &device);
    auto queryCounter = glad_glQueryCounter;
    static int timestampWrites = 0;
    glad_glQueryCounter = [](GLuint, GLenum) { ++timestampWrites; };
    auto disabledScope = device.profileExternalScope("off");
    glad_glQueryCounter = queryCounter;
    require(!disabledScope && timestampWrites == 0,
            "disabled native bridge wrote GPU timestamps");
    profiler.setEnabled(true);
    try {
        auto capturedFrame = profiler.frame(300, BackendType::OpenGL);
        std::thread worker([&] {
            require(!device.profileExternalScope("worker"),
                    "native bridge accepted worker recording");
        });
        worker.join();
        auto native = device.profileExternalScope("frame");
        require(bool(native), "native bridge unavailable");
        glClear(GL_COLOR_BUFFER_BIT);
        throw std::runtime_error("intentional aborted frame");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "intentional aborted frame",
                "unexpected native bridge failure");
    }
    const auto aborted = profiler.latestFrameProfile();
    require(aborted->finalized && aborted->samples.size() == 2 &&
                aborted->samples[1].path == "frame" &&
                aborted->samples[1].status == ProfileSampleStatus::Invalid,
            "native scope exception produced a valid/duplicate frame sample");
    profiler.setEnabled(false);
    {
        auto inactive = profiler.frame(301, BackendType::OpenGL);
        require(!device.profileExternalScope("disabled_after_capture"),
                "native scope remained enabled");
    }
    context->beginFrame(3, 302, BackendType::OpenGL);
    device.GraphicsDevice::profileExternalScope("frame");
    const auto unsupported = context->endFrame();
    require(unsupported->samples.size() == 1 &&
                unsupported->samples[0].status ==
                    ProfileSampleStatus::Unsupported,
            "unsupported native bridge lost fallback or duplicated frame");
    device.shutdown();

    // Recreate the actual GL context and reject the previous generation's
    // token.
    auto* original = glfwGetCurrentContext();
    auto* recreated =
        glfwCreateWindow(64, 64, "timestamp recreation", nullptr, nullptr);
    require(recreated, "context recreation failed");
    glfwMakeContextCurrent(recreated);
    {
        OpenGLTimestampPool fresh(2);
        bool rejected = false;
        try {
            fresh.write(replacement, false);
        } catch (const std::logic_error&) {
            rejected = true;
        }
        require(rejected, "old generation query accepted by recreated context");
    }
    glfwDestroyWindow(recreated);
    glfwMakeContextCurrent(original);
    require(glGetError() == GL_NO_ERROR, "GL error in timestamp smoke");
    std::cout << "PASS: asynchronous GPU queries, delayed availability, "
                 "exhaustion, abandonment and context recreation\n";
}
