from . import op, ui

import bpy

def register():
    bpy.utils.register_class(op.MESH_OT_bridge_plus)
    bpy.utils.register_class(op.MESH_OT_bridge_plus_debug)
    bpy.utils.register_class(ui.MESH_PT_mesh_tools_bridge)

def unregister():
    bpy.utils.unregister_class(ui.MESH_PT_mesh_tools_bridge)
    bpy.utils.unregister_class(op.MESH_OT_bridge_plus)
    bpy.utils.unregister_class(op.MESH_OT_bridge_plus_debug)
