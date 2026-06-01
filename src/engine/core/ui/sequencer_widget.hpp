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
    bool fitToContent = false;
    float minFramePixelWidth = 0.1f;
    float maxFramePixelWidth = 50.0f;
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
