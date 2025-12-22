import bpy

class MESH_PT_mesh_tools_bridge(bpy.types.Panel):
    bl_label = "Mesh Tools"
    bl_idname = "MESH_PT_mesh_tools_bridge"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'
    bl_context = "mesh_edit"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("mesh.bridge_plus", text="Bridge Plus")
        col.operator("mesh.bridge_plus_debug", text="Start/Stop Debug")
