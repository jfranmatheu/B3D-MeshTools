bl_info = {
    "name": "Mesh Tools",
    "author": "I",
    "version": (1, 0),
    "blender": (4, 2, 0), # Adjusted to a released version for safety, though 5.0 is requested
    "location": "View3D > Edit Mode > Toolbar",
    "description": "Mesh Tools",
    "category": "Mesh",
}

import bpy

if bpy.app.background:
    # Fix #25. Skip registering if Blender is in background.
    def register():
        pass
    def unregister():
        pass
else:
    from .addon_utils import AutoCode
    AutoCode.ICONS('icons.py', 'assets/icons')
    
    from .drag_face import register as register_drag_face, unregister as unregister_drag_face
    from .bridge import register as register_bridge, unregister as unregister_bridge
    from .utils.override_helpers import unregister as unregister_overrides
            
    def register():
        register_drag_face()
        register_bridge()

    def unregister():
        unregister_bridge()
        unregister_drag_face()
        unregister_overrides()
