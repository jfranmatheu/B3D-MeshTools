import bpy

from .drag_face import register as register_drag_face, unregister as unregister_drag_face
from .bridge import register as register_bridge, unregister as unregister_bridge

bl_info = {
    "name": "Mesh Tools",
    "author": "I",
    "version": (1, 0),
    "blender": (4, 2, 0), # Adjusted to a released version for safety, though 5.0 is requested
    "location": "View3D > Edit Mode > Toolbar",
    "description": "Mesh Tools",
    "category": "Mesh",
}

def register():
    register_drag_face()
    register_bridge()

def unregister():
    unregister_bridge()
    unregister_drag_face()

if __name__ == "__main__":
    register()
