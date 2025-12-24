from . import op, ui, tool

import bpy
from bl_ui.space_toolsystem_common import activate_by_id, ToolDef

from ..utils.override_helpers import wrap_function


def register():
    bpy.utils.register_class(op.MESH_OT_bridge_plus)
    bpy.utils.register_class(op.MESH_OT_bridge_plus_debug)
    bpy.utils.register_class(ui.MESH_PT_mesh_tools_bridge)
    bpy.utils.register_tool(tool.BridgePlusTool)
    
    wrap_function(activate_by_id, post_func=tool.on_tool_switch_post)

def unregister():
    tool.BridgePlusTool.disable(bpy.context)
    bpy.utils.unregister_tool(tool.BridgePlusTool)
    bpy.utils.unregister_class(ui.MESH_PT_mesh_tools_bridge)
    bpy.utils.unregister_class(op.MESH_OT_bridge_plus)
    bpy.utils.unregister_class(op.MESH_OT_bridge_plus_debug)
