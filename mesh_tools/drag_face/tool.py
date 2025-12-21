import bpy


class DragFaceTool(bpy.types.WorkSpaceTool):
    bl_idname = "mesh_tools.drag_face_tool"
    bl_label = "Drag Face"
    bl_description = "Drag faces with geometric resistance"
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_icon = 'ops.transform.translate'
    bl_widget = None
    bl_keymap = (
        ("mesh.drag_face_tool", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
    )

    def draw_settings(context, layout, tool):
        settings = context.scene.drag_face_settings
        layout.prop(settings, "radius")
        layout.prop(settings, "falloff_power")
