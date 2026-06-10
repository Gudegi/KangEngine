#ifndef _SEQUENCER_WIDGET_HPP_
#define _SEQUENCER_WIDGET_HPP_

// This code is modificated from :
// https://github.com/CedricGuillemet/ImGuizmo
// v1.92.5 WIP

#include <cstddef>

struct ImDrawList;
struct ImRect;

namespace KE {
namespace UI {

enum SequencerOptions {
    SequencerEditNone = 0,
    SequencerEditStartEnd = 1 << 1,
    SequencerChangeFrame = 1 << 3,
    SequencerAdd = 1 << 4,
    SequencerDel = 1 << 5,
    SequencerCopyPaste = 1 << 6,
    SequencerEditAll = SequencerEditStartEnd | SequencerChangeFrame,
};

struct SequencerConfig {
    // Fit the full frame range into the available timeline width.
    bool fitToContent = false;
    // Refit once immediately, for example when a UI toggle is enabled.
    bool requestFitToContent = false;
    // Restore the timeline view saved before fit-to-content was enabled.
    bool requestRestoreView = false;
    // Show the bottom bar for panning and zooming the visible frame range.
    bool showScrollBar = false;
    // Smallest allowed pixels-per-frame scale.
    float minFramePixelWidth = 0.1f;
    // Largest allowed pixels-per-frame scale.
    float maxFramePixelWidth = 50.0f;
    // Smoothing factor for animated zoom changes.
    float framePixelLerp = 0.33f;
    // Width reserved for track labels on the left.
    int legendWidth = 200;
    // Optional live legend width value updated by dragging the splitter.
    float* legendWidthValue = nullptr;
    // Smallest draggable legend width.
    float minLegendWidth = 80.0f;
    // Largest draggable legend width.
    float maxLegendWidth = 420.0f;
    // Hit area around the legend/timeline splitter.
    float legendResizeHandleWidth = 10.0f;
    // Height of one track row and the frame header.
    int itemHeight = 20;
    // Height of the optional bottom panning/zoom bar.
    float scrollBarHeight = 14.0f;
    // Padding below the track content area.
    float childBottomPadding = 8.0f;
    // Minimum pixel spacing between major frame ticks.
    float minTickSpacing = 150.0f;
    // Width of the current-frame cursor line.
    float cursorWidth = 8.0f;
    // Minimum width of draggable clip handles.
    float minHandleWidth = 10.0f;
    // Minimum width of the scrollbar thumb.
    float minScrollBarWidth = 44.0f;
    // Corner radius for major timeline rectangles.
    float rectRounding = 7.0f;
};

struct SequenceInterface {
    bool focused = false;

    virtual int GetFrameMin() const = 0;
    virtual int GetFrameMax() const = 0;
    virtual int GetItemCount() const = 0;

    virtual void BeginEdit(int index) {}
    virtual void EndEdit() {}
    virtual int GetItemTypeCount() const { return 0; }
    virtual const char* GetItemTypeName(int typeIndex) const { return ""; }
    virtual const char* GetItemLabel(int index) const { return ""; }
    virtual const char* GetCollapseFmt() const {
        return "%d Frames / %d entries";
    }

    virtual void Get(int index, int** start, int** end, int* type,
                     unsigned int* color) = 0;
    virtual void Add(int type) {}
    virtual void Del(int index) {}
    virtual void Duplicate(int index) {}

    virtual void Copy() {}
    virtual void Paste() {}

    virtual size_t GetCustomHeight(int index) { return 0; }
    virtual void DoubleClick(int index) {}
    virtual void CustomDraw(int index, ImDrawList* drawList, const ImRect& rect,
                            const ImRect& legendRect,
                            const ImRect& clippingRect,
                            const ImRect& legendClippingRect) {}
    virtual void CustomDrawCompact(int index, ImDrawList* drawList,
                                   const ImRect& rect,
                                   const ImRect& clippingRect) {}

    virtual ~SequenceInterface() = default;
};

bool sequencer(SequenceInterface* sequence, int* currentFrame, bool* expanded,
               int* selectedEntry, int* firstFrame, int sequenceOptions,
               const SequencerConfig& config = {});
} // namespace UI
} // namespace KE

#endif // _SEQUENCER_WIDGET_HPP_
