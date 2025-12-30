from . import op, ui, tool, pg

import bpy
from bl_ui.space_toolsystem_common import activate_by_id, ToolDef

from ..utils.override_helpers import wrap_function


classes = (
    pg.BridgeToolSettings,
    op.MESH_OT_bridge_plus,
    op.MESH_OT_bridge_plus_debug,
    ui.MESH_PT_mesh_tools_bridge
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.utils.register_tool(tool.BridgePlusTool)
    wrap_function(activate_by_id, post_func=tool.on_tool_switch_post)
    
    bpy.types.Scene.bridge_tool_settings = bpy.props.PointerProperty(type=pg.BridgeToolSettings)

def unregister():
    if hasattr(bpy.types.Scene, "bridge_tool_settings"):
        del bpy.types.Scene.bridge_tool_settings

    tool.BridgePlusTool.disable(bpy.context)
    bpy.utils.unregister_tool(tool.BridgePlusTool)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
