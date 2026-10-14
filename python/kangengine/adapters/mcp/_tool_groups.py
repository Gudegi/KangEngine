"""Select MCP tool exposure at server construction, without importing the SDK."""

import os


TOOL_GROUPS = {
    "scene": frozenset(
        {
            "scene_inspect",
            "scene_create_primitive",
            "scene_set_transform",
            "scene_set_material",
            "scene_set_visibility",
            "scene_delete",
            "object_inspect",
            "resource_inspect",
        }
    ),
    "camera": frozenset({"camera_get", "camera_set", "camera_focus"}),
    "capture": frozenset({"viewport_capture"}),
    "recording": frozenset({"recording_start", "recording_status", "recording_stop"}),
    "profile": frozenset(
        {
            "renderer_profile",
            "profile_start",
            "profile_status",
            "profile_result",
            "profile_cancel",
        }
    ),
    "docs": frozenset({"search_docs", "get_doc", "get_api_docs"}),
    "scenario": frozenset(
        {
            "scenario_targets",
            "scenario_start",
            "scenario_status",
            "scenario_cancel",
            "scenario_capture",
        }
    ),
}
DEFAULT_GROUPS = frozenset(TOOL_GROUPS) - {"scenario"}


def configured_groups():
    value = os.environ.get("KANGENGINE_MCP_TOOLS")
    if value is None:
        return DEFAULT_GROUPS
    groups = frozenset(part.strip() for part in value.split(",") if part.strip())
    if not groups or groups - TOOL_GROUPS.keys():
        raise ValueError(
            "KANGENGINE_MCP_TOOLS must list groups from: " + ",".join(TOOL_GROUPS)
        )
    return groups


def enabled_tools(groups):
    return frozenset({"engine_status"}).union(*(TOOL_GROUPS[group] for group in groups))
