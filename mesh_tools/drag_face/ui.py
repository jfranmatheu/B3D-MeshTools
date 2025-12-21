import bpy


class DragFacePanel(bpy.types.Panel):
    """Panel for Drag Face Tool settings"""
    bl_label = "Drag Face Settings"
    bl_idname = "VIEW3D_PT_drag_face_settings"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    @classmethod
    def poll(cls, context):
        return (context.object is not None and 
                context.object.mode == 'EDIT')

    def draw(self, context):
        layout = self.layout
        settings = context.scene.drag_face_settings
        
        layout.prop(settings, "radius")
        layout.prop(settings, "falloff_power")
        
        layout.label(text="Instructions:")
        layout.label(text="- Select Drag Face tool in toolbar")
        layout.label(text="- Click and drag a face")
        layout.label(text="- Snap to other objects automatically")
