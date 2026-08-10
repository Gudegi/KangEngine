// This code is modificated from :
// https://github.com/CedricGuillemet/ImGuizmo
// v1.92.5 WIP
#include "sequencer_widget.hpp"
#include "imgui.h"
#include "imgui_internal.h"
#include <cstdlib>
#include <string>

namespace KE::UI {
#ifndef IMGUI_DEFINE_MATH_OPERATORS
static ImVec2 operator+(const ImVec2& a, const ImVec2& b) {
    return ImVec2(a.x + b.x, a.y + b.y);
}
#endif

static std::string ellipsizeText(const char* text, float maxWidth) {
    if (!text || maxWidth <= 0.0f)
        return "";
    if (ImGui::CalcTextSize(text).x <= maxWidth)
        return text;

    constexpr const char* ellipsis = "...";
    const float ellipsisWidth = ImGui::CalcTextSize(ellipsis).x;
    if (ellipsisWidth > maxWidth)
        return "";

    const std::string source(text);
    size_t lo = 0;
    size_t hi = source.size();
    while (lo < hi) {
        const size_t mid = (lo + hi + 1) / 2;
        const std::string candidate = source.substr(0, mid) + ellipsis;
        if (ImGui::CalcTextSize(candidate.c_str()).x <= maxWidth)
            lo = mid;
        else
            hi = mid - 1;
    }
    return source.substr(0, lo) + ellipsis;
}

static bool SequencerAddDelButton(ImDrawList* draw_list, ImVec2 pos,
                                  bool add = true) {
    ImGuiIO& io = ImGui::GetIO();
    ImRect btnRect(pos, ImVec2(pos.x + 16, pos.y + 16));
    bool overBtn = btnRect.Contains(io.MousePos);
    bool containedClick = overBtn && btnRect.Contains(io.MouseClickedPos[0]);
    bool clickedBtn = containedClick && io.MouseReleased[0];
    int btnColor = overBtn ? 0xAAEAFFAA : 0x77A3B2AA;
    if (containedClick && io.MouseDownDuration[0] > 0)
        btnRect.Expand(2.0f);

    float midy = pos.y + 16 / 2 - 0.5f;
    float midx = pos.x + 16 / 2 - 0.5f;
    draw_list->AddRect(btnRect.Min, btnRect.Max, btnColor, 4);
    draw_list->AddLine(ImVec2(btnRect.Min.x + 3, midy),
                       ImVec2(btnRect.Max.x - 3, midy), btnColor, 2);
    if (add)
        draw_list->AddLine(ImVec2(midx, btnRect.Min.y + 3),
                           ImVec2(midx, btnRect.Max.y - 3), btnColor, 2);
    return clickedBtn;
}

bool sequencer(SequenceInterface* sequence, int* currentFrame, bool* expanded,
               int* selectedEntry, int* firstFrame, int sequenceOptions,
               const SequencerConfig& config) {
    bool ret = false;
    ImGuiIO& io = ImGui::GetIO();
    int cx = (int)(io.MousePos.x);
    int cy = (int)(io.MousePos.y);
    static float framePixelWidth = 10.f;
    static float framePixelWidthTarget = 10.f;
    static float framePixelWidthBeforeFit = 10.f;
    static float framePixelWidthTargetBeforeFit = 10.f;
    static int firstFrameBeforeFit = 0;
    int legendWidth = ImMax(config.legendWidth, 0);
    const float rectRounding = ImMax(config.rectRounding, 0.0f);

    static int movingEntry = -1;
    static int movingPos = -1;
    static int movingPart = -1;
    int delEntry = -1;
    int dupEntry = -1;
    const int itemHeight = ImMax(config.itemHeight, 1);

    bool popupOpened = false;
    int sequenceCount = sequence->GetItemCount();
    if (!sequenceCount)
        return false;
    ImGui::BeginGroup();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos =
        ImGui::GetCursorScreenPos(); // ImDrawList API uses screen coordinates!
    ImVec2 canvas_size =
        ImGui::GetContentRegionAvail(); // Resize canvas to what's available
    int firstFrameUsed = firstFrame ? *firstFrame : 0;
    bool legendResizeHovered = false;
    bool legendResizeHeld = false;
    if (config.legendWidthValue && canvas_size.x > 1.0f) {
        const float minLegendWidth = ImMax(config.minLegendWidth, 0.0f);
        const float maxLegendWidth =
            ImMax(minLegendWidth, config.maxLegendWidth);
        const float timelineMinWidth = ImMax(80.0f, itemHeight * 2.0f);
        const float clampedMaxLegendWidth =
            ImMin(maxLegendWidth,
                  ImMax(minLegendWidth, canvas_size.x - timelineMinWidth));
        *config.legendWidthValue = ImClamp(
            *config.legendWidthValue, minLegendWidth, clampedMaxLegendWidth);
        legendWidth = static_cast<int>(*config.legendWidthValue);

        const float handleWidth = ImMax(config.legendResizeHandleWidth, 1.0f);
        const float handleX = canvas_pos.x + *config.legendWidthValue;
        const ImRect resizeRect(
            ImVec2(handleX - handleWidth * 0.5f, canvas_pos.y),
            ImVec2(handleX + handleWidth * 0.5f, canvas_pos.y + canvas_size.y));
        const ImVec2 cursorBeforeResize = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos(resizeRect.Min);
        ImGui::InvisibleButton("legendResize", resizeRect.GetSize());
        legendResizeHovered = ImGui::IsItemHovered();
        legendResizeHeld = ImGui::IsItemActive();
        ImGui::SetCursorScreenPos(cursorBeforeResize);
        if (legendResizeHovered || legendResizeHeld)
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        if (legendResizeHeld && io.MouseDelta.x != 0.0f) {
            *config.legendWidthValue =
                ImClamp(*config.legendWidthValue + io.MouseDelta.x,
                        minLegendWidth, clampedMaxLegendWidth);
            legendWidth = static_cast<int>(*config.legendWidthValue);
        }
    }

    int controlHeight = sequenceCount * itemHeight;
    for (int i = 0; i < sequenceCount; i++)
        controlHeight += int(sequence->GetCustomHeight(i));
    int frameCount =
        ImMax(sequence->GetFrameMax() - sequence->GetFrameMin(), 1);
    if (config.requestFitToContent && firstFrame) {
        framePixelWidthBeforeFit = framePixelWidth;
        framePixelWidthTargetBeforeFit = framePixelWidthTarget;
        firstFrameBeforeFit = *firstFrame;
    }
    if (config.requestRestoreView && firstFrame) {
        framePixelWidth = framePixelWidthBeforeFit;
        framePixelWidthTarget = framePixelWidthTargetBeforeFit;
        *firstFrame = firstFrameBeforeFit;
    }
    if ((config.fitToContent || config.requestFitToContent) && firstFrame) {
        // Keep the rounded final edge away from the content clip boundary.
        const float availableWidth =
            canvas_size.x - legendWidth - rectRounding - 1.0f;
        const int inclusiveFrameCount = frameCount + 1;
        if (availableWidth > 1.0f && inclusiveFrameCount > 0) {
            framePixelWidthTarget = framePixelWidth = ImClamp(
                availableWidth / static_cast<float>(inclusiveFrameCount), 1e-4f,
                config.maxFramePixelWidth);
            *firstFrame = sequence->GetFrameMin();
        }
    }

    static bool MovingScrollBar = false;
    static bool MovingCurrentFrame = false;
    struct CustomDraw {
        int index;
        ImRect customRect;
        ImRect legendRect;
        ImRect clippingRect;
        ImRect legendClippingRect;
    };
    ImVector<CustomDraw> customDraws;
    ImVector<CustomDraw> compactCustomDraws;
    // zoom in/out
    const int visibleFrameCount =
        (int)floorf((canvas_size.x - legendWidth) / framePixelWidth);
    const float barWidthRatio =
        ImMin(visibleFrameCount / (float)frameCount, 1.f);
    const float barWidthInPixels =
        barWidthRatio * (canvas_size.x - legendWidth);

    ImRect regionRect(canvas_pos, canvas_pos + canvas_size);

    static bool panningView = false;
    static ImVec2 panningViewSource;
    static int panningViewFrame;
    if (ImGui::IsWindowFocused() && io.KeyAlt && io.MouseDown[2]) {
        if (!panningView) {
            panningViewSource = io.MousePos;
            panningView = true;
            panningViewFrame = *firstFrame;
        }
        *firstFrame =
            panningViewFrame -
            int((io.MousePos.x - panningViewSource.x) / framePixelWidth);
        *firstFrame = ImClamp(*firstFrame, sequence->GetFrameMin(),
                              sequence->GetFrameMax() - visibleFrameCount);
    }
    if (panningView && !io.MouseDown[2]) {
        panningView = false;
    }
    const float minFramePixelWidth =
        config.fitToContent ? 1e-4f : config.minFramePixelWidth;
    framePixelWidthTarget = ImClamp(framePixelWidthTarget, minFramePixelWidth,
                                    config.maxFramePixelWidth);

    framePixelWidth = ImLerp(framePixelWidth, framePixelWidthTarget,
                             ImClamp(config.framePixelLerp, 0.0f, 1.0f));

    frameCount = sequence->GetFrameMax() - sequence->GetFrameMin();
    if (visibleFrameCount >= frameCount && firstFrame)
        *firstFrame = sequence->GetFrameMin();

    // --
    if (expanded && !*expanded) {
        ImGui::InvisibleButton(
            "canvas", ImVec2(canvas_size.x - canvas_pos.x, (float)itemHeight));
        draw_list->AddRectFilled(
            canvas_pos,
            ImVec2(canvas_size.x + canvas_pos.x, canvas_pos.y + itemHeight),
            0xFF3D3837, 0);
        char tmps[512];
        ImFormatString(tmps, IM_ARRAYSIZE(tmps), sequence->GetCollapseFmt(),
                       frameCount, sequenceCount);
        draw_list->AddText(ImVec2(canvas_pos.x + 26, canvas_pos.y + 2),
                           0xFFFFFFFF, tmps);
    } else {
        // Shows the bottom timeline viewport bar used for panning and zooming.
        bool hasScrollBar = config.showScrollBar;
        /*
        int framesPixelWidth = int(frameCount * framePixelWidth);
        if ((framesPixelWidth + legendWidth) >= canvas_size.x)
        {
            hasScrollBar = true;
        }
        */
        // test scroll area
        ImVec2 headerSize(canvas_size.x, (float)itemHeight);
        ImVec2 scrollBarSize(canvas_size.x, config.scrollBarHeight);
        ImGui::InvisibleButton("topBar", headerSize);
        draw_list->AddRectFilled(canvas_pos, canvas_pos + headerSize,
                                 0xFFFF0000, 0);
        ImVec2 childFramePos = ImGui::GetCursorScreenPos();
        ImVec2 childFrameSize(
            canvas_size.x,
            ImMax(1.0f, canvas_size.y - config.childBottomPadding -
                            headerSize.y -
                            (hasScrollBar ? scrollBarSize.y : 0.0f)));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, 0);
        ImGui::BeginChild("SequencerContent", childFrameSize,
                          ImGuiChildFlags_FrameStyle);
        sequence->focused = ImGui::IsWindowFocused();
        ImGui::InvisibleButton("contentBar",
                               ImVec2(canvas_size.x, float(controlHeight)));
        const ImVec2 contentMin = ImGui::GetItemRectMin();
        const ImVec2 contentMax = ImGui::GetItemRectMax();
        const ImRect contentRect(contentMin, contentMax);
        const float contentHeight = contentMax.y - contentMin.y;

        // full background
        draw_list->AddRectFilled(canvas_pos, canvas_pos + canvas_size,
                                 0xFF242424, 0);
        if (config.legendWidthValue) {
            const unsigned int splitterColor = legendResizeHeld ? 0xFF9F8CFF
                                               : legendResizeHovered
                                                   ? 0xAA9F8CFF
                                                   : 0x55404040;
            draw_list->AddLine(
                ImVec2(canvas_pos.x + legendWidth, canvas_pos.y + itemHeight),
                ImVec2(canvas_pos.x + legendWidth,
                       canvas_pos.y + canvas_size.y),
                splitterColor, 1.0f);
        }

        // current frame top
        ImRect topRect(
            ImVec2(canvas_pos.x + legendWidth, canvas_pos.y),
            ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + itemHeight));

        if (!MovingCurrentFrame && !MovingScrollBar && movingEntry == -1 &&
            !legendResizeHovered && !legendResizeHeld &&
            sequenceOptions & SequencerChangeFrame && currentFrame &&
            *currentFrame >= 0 && topRect.Contains(io.MousePos) &&
            io.MouseDown[0]) {
            MovingCurrentFrame = true;
        }
        if (MovingCurrentFrame) {
            if (frameCount) {
                const int previousFrame = *currentFrame;
                *currentFrame =
                    (int)((io.MousePos.x - topRect.Min.x) / framePixelWidth) +
                    firstFrameUsed;
                if (*currentFrame < sequence->GetFrameMin())
                    *currentFrame = sequence->GetFrameMin();
                if (*currentFrame >= sequence->GetFrameMax())
                    *currentFrame = sequence->GetFrameMax();
                if (*currentFrame != previousFrame)
                    ret = true;
            }
            if (!io.MouseDown[0])
                MovingCurrentFrame = false;
        }

        // header
        draw_list->AddRectFilled(
            canvas_pos,
            ImVec2(canvas_size.x + canvas_pos.x, canvas_pos.y + itemHeight),
            0xFF3D3837, 0);
        if (sequenceOptions & SequencerAdd) {
            if (SequencerAddDelButton(
                    draw_list,
                    ImVec2(canvas_pos.x + legendWidth - itemHeight,
                           canvas_pos.y + 2),
                    true))
                ImGui::OpenPopup("addEntry");

            if (ImGui::BeginPopup("addEntry")) {
                for (int i = 0; i < sequence->GetItemTypeCount(); i++)
                    if (ImGui::Selectable(sequence->GetItemTypeName(i))) {
                        sequence->Add(i);
                        *selectedEntry = sequence->GetItemCount() - 1;
                    }

                ImGui::EndPopup();
                popupOpened = true;
            }
        }

        // header frame number and lines
        int modFrameCount = 10;
        int frameStep = 1;
        while ((modFrameCount * framePixelWidth) < config.minTickSpacing) {
            modFrameCount *= 2;
            frameStep *= 2;
        };
        int halfModFrameCount = modFrameCount / 2;

        auto drawLine = [&](int i, int regionHeight) {
            bool baseIndex =
                ((i % modFrameCount) == 0) ||
                (i == sequence->GetFrameMax() || i == sequence->GetFrameMin());
            bool halfIndex = (i % halfModFrameCount) == 0;
            int px = (int)canvas_pos.x + int(i * framePixelWidth) +
                     legendWidth - int(firstFrameUsed * framePixelWidth);
            int tiretStart = baseIndex ? 4 : (halfIndex ? 10 : 14);
            int tiretEnd = baseIndex ? regionHeight : itemHeight;

            if (px <= (canvas_size.x + canvas_pos.x) &&
                px >= (canvas_pos.x + legendWidth)) {
                draw_list->AddLine(
                    ImVec2((float)px, canvas_pos.y + (float)tiretStart),
                    ImVec2((float)px, canvas_pos.y + (float)tiretEnd - 1),
                    0xFF606060, 1);

                draw_list->AddLine(
                    ImVec2((float)px, canvas_pos.y + (float)itemHeight),
                    ImVec2((float)px, canvas_pos.y + (float)regionHeight - 1),
                    0x30606060, 1);
            }

            if (baseIndex && px > (canvas_pos.x + legendWidth)) {
                char tmps[512];
                ImFormatString(tmps, IM_ARRAYSIZE(tmps), "%d", i);
                const ImVec2 textSize = ImGui::CalcTextSize(tmps);
                const float textX =
                    ImClamp((float)px + 3.f, canvas_pos.x + legendWidth + 3.f,
                            canvas_pos.x + canvas_size.x - textSize.x - 3.f);
                draw_list->AddText(ImVec2(textX, canvas_pos.y), 0xFFBBBBBB,
                                   tmps);
            }
        };

        auto drawLineContent = [&](int i, int /*regionHeight*/) {
            int px = (int)canvas_pos.x + int(i * framePixelWidth) +
                     legendWidth - int(firstFrameUsed * framePixelWidth);
            int tiretStart = int(contentMin.y);
            int tiretEnd = int(contentMax.y);

            if (px <= (canvas_size.x + canvas_pos.x) &&
                px >= (canvas_pos.x + legendWidth)) {
                // draw_list->AddLine(ImVec2((float)px, canvas_pos.y +
                // (float)tiretStart), ImVec2((float)px, canvas_pos.y +
                // (float)tiretEnd - 1), 0xFF606060, 1);

                draw_list->AddLine(ImVec2(float(px), float(tiretStart)),
                                   ImVec2(float(px), float(tiretEnd)),
                                   0x30606060, 1);
            }
        };
        const int frameMin = sequence->GetFrameMin();
        const int frameMax = sequence->GetFrameMax();
        for (int i = frameMin; i <= frameMax; i += frameStep) {
            drawLine(i, itemHeight);
        }
        if ((frameMax - frameMin) % frameStep != 0)
            drawLine(frameMax, itemHeight);
        /*
                 draw_list->AddLine(canvas_pos, ImVec2(canvas_pos.x,
           canvas_pos.y + controlHeight), 0xFF000000, 1);
                 draw_list->AddLine(ImVec2(canvas_pos.x, canvas_pos.y +
           itemHeight), ImVec2(canvas_size.x, canvas_pos.y + itemHeight),
           0xFF000000, 1);
                 */
        // clip content

        draw_list->PushClipRect(childFramePos, childFramePos + childFrameSize,
                                true);

        // draw item names in the legend rect on the left
        size_t customHeight = 0;
        for (int i = 0; i < sequenceCount; i++) {
            int type;
            sequence->Get(i, NULL, NULL, &type, NULL);
            const float labelPaddingX = 3.0f;
            const float textHeight = ImGui::GetTextLineHeight();
            const float rowTop = contentMin.y + i * itemHeight + customHeight;
            const float labelRight =
                contentMin.x + legendWidth -
                ((sequenceOptions & SequencerDel) ? itemHeight * 2.0f : 0.0f) -
                labelPaddingX;
            ImVec2 tpos(contentMin.x + labelPaddingX,
                        rowTop + ImMax((itemHeight - textHeight) * 0.5f, 0.0f));
            const float labelWidth = labelRight - tpos.x;
            const std::string label =
                ellipsizeText(sequence->GetItemLabel(i), labelWidth);
            draw_list->PushClipRect(
                ImVec2(contentMin.x, rowTop),
                ImVec2(contentMin.x + legendWidth, rowTop + itemHeight), true);
            draw_list->AddText(tpos, 0xFFFFFFFF, label.c_str());
            draw_list->PopClipRect();

            if (sequenceOptions & SequencerDel) {
                if (SequencerAddDelButton(
                        draw_list,
                        ImVec2(contentMin.x + legendWidth - itemHeight + 2 - 10,
                               tpos.y + 2),
                        false))
                    delEntry = i;

                if (SequencerAddDelButton(draw_list,
                                          ImVec2(contentMin.x + legendWidth -
                                                     itemHeight - itemHeight +
                                                     2 - 10,
                                                 tpos.y + 2),
                                          true))
                    dupEntry = i;
            }
            customHeight += sequence->GetCustomHeight(i);
        }

        // slots background
        customHeight = 0;
        for (int i = 0; i < sequenceCount; i++) {
            unsigned int col = (i & 1) ? 0xFF3A3636 : 0xFF413D3D;

            size_t localCustomHeight = sequence->GetCustomHeight(i);
            ImVec2 pos =
                ImVec2(contentMin.x + legendWidth,
                       contentMin.y + itemHeight * i + 1 + customHeight);
            ImVec2 sz = ImVec2(canvas_size.x + canvas_pos.x,
                               pos.y + itemHeight - 1 + localCustomHeight);
            if (!popupOpened && cy >= pos.y &&
                cy < pos.y + (itemHeight + localCustomHeight) &&
                movingEntry == -1 && cx > contentMin.x &&
                cx < contentMin.x + canvas_size.x) {
                col += 0x80201008;
                pos.x -= legendWidth;
            }
            draw_list->AddRectFilled(pos, sz, col, 0);
            customHeight += localCustomHeight;
        }

        draw_list->PushClipRect(childFramePos + ImVec2(float(legendWidth), 0.f),
                                childFramePos + childFrameSize, true);

        // vertical frame lines in content area
        for (int i = frameMin; i <= frameMax; i += frameStep) {
            drawLineContent(i, int(contentHeight));
        }
        if ((frameMax - frameMin) % frameStep != 0)
            drawLineContent(frameMax, int(contentHeight));

        // selection
        bool selected = selectedEntry && (*selectedEntry >= 0);
        if (selected) {
            customHeight = 0;
            for (int i = 0; i < *selectedEntry; i++)
                customHeight += sequence->GetCustomHeight(i);
            draw_list->AddRectFilled(
                ImVec2(contentMin.x, contentMin.y +
                                         itemHeight * *selectedEntry +
                                         customHeight),
                ImVec2(contentMin.x + canvas_size.x,
                       contentMin.y + itemHeight * (*selectedEntry + 1) +
                           customHeight),
                0x801080FF, 1.f);
        }

        // slots
        customHeight = 0;
        for (int i = 0; i < sequenceCount; i++) {
            int *start, *end;
            unsigned int color;
            sequence->Get(i, &start, &end, NULL, &color);
            size_t localCustomHeight = sequence->GetCustomHeight(i);

            ImVec2 pos = ImVec2(
                contentMin.x + legendWidth - firstFrameUsed * framePixelWidth,
                contentMin.y + itemHeight * i + 1 + customHeight);
            ImVec2 slotP1(pos.x + *start * framePixelWidth, pos.y + 2);
            ImVec2 slotP2(pos.x + *end * framePixelWidth + framePixelWidth,
                          pos.y + itemHeight - 2);
            ImVec2 slotP3(pos.x + *end * framePixelWidth + framePixelWidth,
                          pos.y + itemHeight - 2 + localCustomHeight);
            unsigned int slotColor = color | 0xFF000000;
            unsigned int slotColorHalf = (color & 0xFFFFFF) | 0x40000000;

            if (slotP1.x <= (canvas_size.x + contentMin.x) &&
                slotP2.x >= (contentMin.x + legendWidth)) {
                draw_list->AddRectFilled(slotP1, slotP3, slotColorHalf,
                                         rectRounding);
                draw_list->AddRectFilled(slotP1, slotP2, slotColor,
                                         rectRounding);
            }
            const int itemCurrentFrame = sequence->GetItemCurrentFrame(i);
            if (itemCurrentFrame >= *start && itemCurrentFrame <= *end) {
                const float markerX =
                    pos.x + itemCurrentFrame * framePixelWidth +
                    framePixelWidth * 0.5f;
                if (markerX >= contentMin.x + legendWidth &&
                    markerX <= contentMin.x + canvas_size.x) {
                    draw_list->AddLine(
                        ImVec2(markerX, slotP1.y),
                        ImVec2(markerX, slotP2.y), 0xFFFFFFFF,
                        std::max(2.0f, config.cursorWidth * 0.35f));
                    draw_list->AddTriangleFilled(
                        ImVec2(markerX, slotP1.y),
                        ImVec2(markerX - 4.0f, slotP1.y + 5.0f),
                        ImVec2(markerX + 4.0f, slotP1.y + 5.0f),
                        0xFFFFFFFF);
                }
            }
            if (ImRect(slotP1, slotP2).Contains(io.MousePos) &&
                io.MouseDoubleClicked[0]) {
                sequence->DoubleClick(i);
            }
            // Ensure grabbable handles
            const float max_handle_width = (slotP2.x - slotP1.x) / 3.0f;
            const float min_handle_width =
                ImMin(config.minHandleWidth, max_handle_width);
            const float handle_width = ImClamp(
                framePixelWidth / 2.0f, min_handle_width, max_handle_width);
            ImRect rects[3] = {
                ImRect(slotP1, ImVec2(slotP1.x + handle_width, slotP2.y)),
                ImRect(ImVec2(slotP2.x - handle_width, slotP1.y), slotP2),
                ImRect(slotP1, slotP2)};

            const unsigned int quadColor[] = {
                0xFFFFFFFF, 0xFFFFFFFF, slotColor + (selected ? 0 : 0x202020)};
            if (movingEntry == -1 &&
                (sequenceOptions &
                 SequencerEditStartEnd)) // TODOFOCUS &&
                                         // backgroundRect.Contains(io.MousePos))
            {
                for (int j = 2; j >= 0; j--) {
                    ImRect& rc = rects[j];
                    if (!rc.Contains(io.MousePos))
                        continue;
                    draw_list->AddRectFilled(rc.Min, rc.Max, quadColor[j],
                                             rectRounding);
                }

                for (int j = 0; j < 3; j++) {
                    ImRect& rc = rects[j];
                    if (!rc.Contains(io.MousePos))
                        continue;
                    if (!ImRect(childFramePos, childFramePos + childFrameSize)
                             .Contains(io.MousePos))
                        continue;
                    if (ImGui::IsMouseClicked(0) && !MovingScrollBar &&
                        !MovingCurrentFrame) {
                        movingEntry = i;
                        movingPos = cx;
                        movingPart = j + 1;
                        sequence->BeginEdit(movingEntry);
                        break;
                    }
                }
            }

            // custom draw
            if (localCustomHeight > 0) {
                ImVec2 rp(canvas_pos.x,
                          contentMin.y + itemHeight * i + 1 + customHeight);
                ImRect customRect(
                    rp + ImVec2(legendWidth - (firstFrameUsed -
                                               sequence->GetFrameMin() - 0.5f) *
                                                  framePixelWidth,
                                float(itemHeight)),
                    rp + ImVec2(legendWidth + (sequence->GetFrameMax() -
                                               firstFrameUsed - 0.5f + 2.f) *
                                                  framePixelWidth,
                                float(localCustomHeight + itemHeight)));
                ImRect clippingRect(
                    rp + ImVec2(float(legendWidth), float(itemHeight)),
                    rp + ImVec2(canvas_size.x,
                                float(localCustomHeight + itemHeight)));

                ImRect legendRect(
                    rp + ImVec2(0.f, float(itemHeight)),
                    rp + ImVec2(float(legendWidth), float(localCustomHeight)));
                ImRect legendClippingRect(
                    canvas_pos + ImVec2(0.f, float(itemHeight)),
                    canvas_pos + ImVec2(float(legendWidth),
                                        float(localCustomHeight + itemHeight)));
                customDraws.push_back({i, customRect, legendRect, clippingRect,
                                       legendClippingRect});
            } else {
                ImVec2 rp(canvas_pos.x,
                          contentMin.y + itemHeight * i + customHeight);
                ImRect customRect(
                    rp + ImVec2(legendWidth - (firstFrameUsed -
                                               sequence->GetFrameMin() - 0.5f) *
                                                  framePixelWidth,
                                float(0.f)),
                    rp + ImVec2(legendWidth + (sequence->GetFrameMax() -
                                               firstFrameUsed - 0.5f + 2.f) *
                                                  framePixelWidth,
                                float(itemHeight)));
                ImRect clippingRect(
                    rp + ImVec2(float(legendWidth), float(0.f)),
                    rp + ImVec2(canvas_size.x, float(itemHeight)));

                compactCustomDraws.push_back(
                    {i, customRect, ImRect(), clippingRect, ImRect()});
            }
            customHeight += localCustomHeight;
        }

        // moving
        if (/*backgroundRect.Contains(io.MousePos) && */ movingEntry >= 0) {
#if IMGUI_VERSION_NUM >= 18723
            ImGui::SetNextFrameWantCaptureMouse(true);
#else
            ImGui::CaptureMouseFromApp();
#endif
            int diffFrame = int((cx - movingPos) / framePixelWidth);
            if (std::abs(diffFrame) > 0) {
                int *start, *end;
                sequence->Get(movingEntry, &start, &end, NULL, NULL);
                if (selectedEntry)
                    *selectedEntry = movingEntry;
                int& l = *start;
                int& r = *end;
                if (movingPart & 1)
                    l += diffFrame;
                if (movingPart & 2)
                    r += diffFrame;
                if (l < 0) {
                    if (movingPart & 2)
                        r -= l;
                    l = 0;
                }
                if (movingPart & 1 && l > r)
                    l = r;
                if (movingPart & 2 && r < l)
                    r = l;
                movingPos += int(diffFrame * framePixelWidth);
            }
            if (!io.MouseDown[0]) {
                // single select
                if (!diffFrame && movingPart && selectedEntry) {
                    *selectedEntry = movingEntry;
                    ret = true;
                }

                movingEntry = -1;
                sequence->EndEdit();
            }
        }

        // cursor
        if (currentFrame && firstFrame && *currentFrame >= *firstFrame &&
            *currentFrame <= sequence->GetFrameMax()) {
            const float cursorWidth = config.cursorWidth;
            float cursorOffset =
                contentMin.x + legendWidth +
                (*currentFrame - firstFrameUsed) * framePixelWidth +
                framePixelWidth / 2 - cursorWidth * 0.5f;
            draw_list->AddLine(ImVec2(cursorOffset, canvas_pos.y),
                               ImVec2(cursorOffset, contentMax.y), 0xA02A2AFF,
                               cursorWidth);
            char tmps[512];
            ImFormatString(tmps, IM_ARRAYSIZE(tmps), "%d", *currentFrame);
            const ImVec2 textSize = ImGui::CalcTextSize(tmps);
            const float textX =
                ImClamp(cursorOffset + 10, canvas_pos.x + legendWidth + 3.f,
                        canvas_pos.x + canvas_size.x - textSize.x - 3.f);
            draw_list->AddText(ImVec2(textX, canvas_pos.y + 2), 0xFF2A2AFF,
                               tmps);
        }

        draw_list->PopClipRect();
        draw_list->PopClipRect();

        for (auto& customDraw : customDraws)
            sequence->CustomDraw(customDraw.index, draw_list,
                                 customDraw.customRect, customDraw.legendRect,
                                 customDraw.clippingRect,
                                 customDraw.legendClippingRect);
        for (auto& customDraw : compactCustomDraws)
            sequence->CustomDrawCompact(customDraw.index, draw_list,
                                        customDraw.customRect,
                                        customDraw.clippingRect);

        // copy paste
        if (sequenceOptions & SequencerCopyPaste) {
            ImRect rectCopy(
                ImVec2(contentMin.x + 100, canvas_pos.y + 2),
                ImVec2(contentMin.x + 100 + 30, canvas_pos.y + itemHeight - 2));
            bool inRectCopy = rectCopy.Contains(io.MousePos);
            unsigned int copyColor = inRectCopy ? 0xFF1080FF : 0xFF000000;
            draw_list->AddText(rectCopy.Min, copyColor, "Copy");

            ImRect rectPaste(
                ImVec2(contentMin.x + 140, canvas_pos.y + 2),
                ImVec2(contentMin.x + 140 + 30, canvas_pos.y + itemHeight - 2));
            bool inRectPaste = rectPaste.Contains(io.MousePos);
            unsigned int pasteColor = inRectPaste ? 0xFF1080FF : 0xFF000000;
            draw_list->AddText(rectPaste.Min, pasteColor, "Paste");

            if (inRectCopy && io.MouseReleased[0]) {
                sequence->Copy();
            }
            if (inRectPaste && io.MouseReleased[0]) {
                sequence->Paste();
            }
        }
        //

        ImGui::EndChild();
        ImGui::PopStyleColor();
        if (hasScrollBar) {
            ImGui::InvisibleButton("scrollBar", scrollBarSize);
            ImVec2 scrollBarMin = ImGui::GetItemRectMin();
            ImVec2 scrollBarMax = ImGui::GetItemRectMax();

            // ratio = number of frames visible in control / number to total
            // frames

            float startFrameOffset =
                ((float)(firstFrameUsed - sequence->GetFrameMin()) /
                 (float)frameCount) *
                (canvas_size.x - legendWidth);
            ImVec2 scrollBarA(scrollBarMin.x + legendWidth, scrollBarMin.y - 2);
            ImVec2 scrollBarB(scrollBarMin.x + canvas_size.x,
                              scrollBarMax.y - 1);
            draw_list->AddRectFilled(scrollBarA, scrollBarB, 0xFF222222, 0);

            ImRect scrollBarRect(scrollBarA, scrollBarB);
            bool inScrollBar = scrollBarRect.Contains(io.MousePos);

            draw_list->AddRectFilled(scrollBarA, scrollBarB, 0xFF101010, 8);

            ImVec2 scrollBarC(scrollBarMin.x + legendWidth + startFrameOffset,
                              scrollBarMin.y);
            ImVec2 scrollBarD(scrollBarMin.x + legendWidth + barWidthInPixels +
                                  startFrameOffset,
                              scrollBarMax.y - 2);
            draw_list->AddRectFilled(
                scrollBarC, scrollBarD,
                (inScrollBar || MovingScrollBar) ? 0xFF606060 : 0xFF505050, 6);

            ImRect barHandleLeft(scrollBarC,
                                 ImVec2(scrollBarC.x + 14, scrollBarD.y));
            ImRect barHandleRight(ImVec2(scrollBarD.x - 14, scrollBarC.y),
                                  scrollBarD);

            bool onLeft = barHandleLeft.Contains(io.MousePos);
            bool onRight = barHandleRight.Contains(io.MousePos);

            static bool sizingRBar = false;
            static bool sizingLBar = false;

            draw_list->AddRectFilled(
                barHandleLeft.Min, barHandleLeft.Max,
                (onLeft || sizingLBar) ? 0xFFAAAAAA : 0xFF666666, 6);
            draw_list->AddRectFilled(
                barHandleRight.Min, barHandleRight.Max,
                (onRight || sizingRBar) ? 0xFFAAAAAA : 0xFF666666, 6);

            ImRect scrollBarThumb(scrollBarC, scrollBarD);
            const float minBarWidth = config.minScrollBarWidth;
            if (sizingRBar) {
                if (!io.MouseDown[0]) {
                    sizingRBar = false;
                } else {
                    float barNewWidth =
                        ImMax(barWidthInPixels + io.MouseDelta.x, minBarWidth);
                    float barRatio = barNewWidth / barWidthInPixels;
                    framePixelWidthTarget = framePixelWidth =
                        framePixelWidth / barRatio;
                    int newVisibleFrameCount = int(
                        (canvas_size.x - legendWidth) / framePixelWidthTarget);
                    int lastFrame = *firstFrame + newVisibleFrameCount;
                    if (lastFrame > sequence->GetFrameMax()) {
                        framePixelWidthTarget = framePixelWidth =
                            (canvas_size.x - legendWidth) /
                            float(sequence->GetFrameMax() - *firstFrame);
                    }
                }
            } else if (sizingLBar) {
                if (!io.MouseDown[0]) {
                    sizingLBar = false;
                } else {
                    if (fabsf(io.MouseDelta.x) > FLT_EPSILON) {
                        float barNewWidth = ImMax(
                            barWidthInPixels - io.MouseDelta.x, minBarWidth);
                        float barRatio = barNewWidth / barWidthInPixels;
                        float previousFramePixelWidthTarget =
                            framePixelWidthTarget;
                        framePixelWidthTarget = framePixelWidth =
                            framePixelWidth / barRatio;
                        int newVisibleFrameCount =
                            int(visibleFrameCount / barRatio);
                        int newFirstFrame = *firstFrame + newVisibleFrameCount -
                                            visibleFrameCount;
                        newFirstFrame = ImClamp(
                            newFirstFrame, sequence->GetFrameMin(),
                            ImMax(sequence->GetFrameMax() - visibleFrameCount,
                                  sequence->GetFrameMin()));
                        if (newFirstFrame == *firstFrame) {
                            framePixelWidth = framePixelWidthTarget =
                                previousFramePixelWidthTarget;
                        } else {
                            *firstFrame = newFirstFrame;
                        }
                    }
                }
            } else {
                if (MovingScrollBar) {
                    if (!io.MouseDown[0]) {
                        MovingScrollBar = false;
                    } else {
                        float framesPerPixelInBar =
                            barWidthInPixels / (float)visibleFrameCount;
                        *firstFrame =
                            int((io.MousePos.x - panningViewSource.x) /
                                framesPerPixelInBar) -
                            panningViewFrame;
                        *firstFrame = ImClamp(
                            *firstFrame, sequence->GetFrameMin(),
                            ImMax(sequence->GetFrameMax() - visibleFrameCount,
                                  sequence->GetFrameMin()));
                    }
                } else {
                    if (scrollBarThumb.Contains(io.MousePos) &&
                        ImGui::IsMouseClicked(0) && firstFrame &&
                        !MovingCurrentFrame && movingEntry == -1) {
                        MovingScrollBar = true;
                        panningViewSource = io.MousePos;
                        panningViewFrame = -*firstFrame;
                    }
                    if (!sizingRBar && onRight && ImGui::IsMouseClicked(0))
                        sizingRBar = true;
                    if (!sizingLBar && onLeft && ImGui::IsMouseClicked(0))
                        sizingLBar = true;
                }
            }
        }
    }

    ImGui::EndGroup();

    if (regionRect.Contains(io.MousePos)) {
        bool overCustomDraw = false;
        for (auto& custom : customDraws) {
            if (custom.customRect.Contains(io.MousePos)) {
                overCustomDraw = true;
            }
        }
        if (overCustomDraw) {
        } else {
#if 0
            frameOverCursor = *firstFrame + (int)(visibleFrameCount * ((io.MousePos.x - (float)legendWidth - canvas_pos.x) / (canvas_size.x - legendWidth)));
            //frameOverCursor = max(min(*firstFrame - visibleFrameCount / 2, frameCount - visibleFrameCount), 0);

            /**firstFrame -= frameOverCursor;
            *firstFrame *= framePixelWidthTarget / framePixelWidth;
            *firstFrame += frameOverCursor;*/
            if (io.MouseWheel < -FLT_EPSILON)
            {
               *firstFrame -= frameOverCursor;
               *firstFrame = int(*firstFrame * 1.1f);
               framePixelWidthTarget *= 0.9f;
               *firstFrame += frameOverCursor;
            }

            if (io.MouseWheel > FLT_EPSILON)
            {
               *firstFrame -= frameOverCursor;
               *firstFrame = int(*firstFrame * 0.9f);
               framePixelWidthTarget *= 1.1f;
               *firstFrame += frameOverCursor;
            }
#endif
        }
    }

    if (expanded) {
        if (SequencerAddDelButton(draw_list,
                                  ImVec2(canvas_pos.x + 2, canvas_pos.y + 2),
                                  !*expanded))
            *expanded = !*expanded;
    }

    if (delEntry != -1) {
        sequence->Del(delEntry);
        if (selectedEntry && (*selectedEntry == delEntry ||
                              *selectedEntry >= sequence->GetItemCount()))
            *selectedEntry = -1;
    }

    if (dupEntry != -1) {
        sequence->Duplicate(dupEntry);
    }
    return ret;
}
} // namespace KE::UI
