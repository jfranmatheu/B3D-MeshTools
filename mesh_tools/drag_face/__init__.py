from .pg import DragFaceSettings
from .op import DragFaceOperator
from .ui import DragFacePanel
from .tool import DragFaceTool

import bpy

classes = (
    DragFaceSettings,
    DragFaceOperator,
    DragFacePanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.utils.register_tool(DragFaceTool, separator=True, group=True)
    bpy.types.Scene.drag_face_settings = bpy.props.PointerProperty(type=DragFaceSettings)

def unregister():
    if hasattr(bpy.types.Scene, "drag_face_settings"):
        del bpy.types.Scene.drag_face_settings
    bpy.utils.unregister_tool(DragFaceTool)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
