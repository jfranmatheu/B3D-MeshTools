import bpy
import bmesh
import mathutils
from bpy_extras import view3d_utils
import gpu
from gpu_extras.batch import batch_for_shader
import math

class DragFaceOperator(bpy.types.Operator):
    """Drag Face with Geometric Resistance"""
    bl_idname = "mesh.drag_face_tool"
    bl_label = "Drag Face"
    bl_options = {'REGISTER', 'UNDO', 'GRAB_CURSOR', 'BLOCKING'}

    # Properties for modal operation
    first_mouse_x: bpy.props.IntProperty()
    first_mouse_y: bpy.props.IntProperty()
    
    # Internal state
    bm = None
    bvh = None
    selected_face_index = -1
    initial_verts_co = {} # {vert_index: Vector}
    drag_start_co = None # Vector (world space)
    initial_face_center = None # Vector (local space)
    initial_face_center_world = None # Vector (world space)
    _handle = None

    def invoke(self, context, event):
        if context.object is None or context.object.mode != 'EDIT':
            self.report({'WARNING'}, "Active object must be in Edit Mode")
            return {'CANCELLED'}

        self.first_mouse_x = event.mouse_x
        self.first_mouse_y = event.mouse_y
        
        # Initialize raycast to find the face
        if not self.raycast_selection(context, event):
            # If no face hit, cancel.
            return {'CANCELLED'} 
            
        # Initialize dragging state
        self.init_drag(context, event)
        
        # Add draw handler
        args = (self, context)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_callback_3d, args, 'WINDOW', 'POST_VIEW')
        
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            self.update_drag(context, event)
            return {'RUNNING_MODAL'}
        
        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            self.finish_drag(context)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel_drag(context)
            return {'CANCELLED'}
            
        return {'RUNNING_MODAL'}

    def raycast_selection(self, context, event):
        obj = context.object
        self.bm = bmesh.from_edit_mesh(obj.data)
        
        # Build BVH for efficient raycasting
        self.bm.faces.ensure_lookup_table()
        self.bvh = mathutils.bvhtree.BVHTree.FromBMesh(self.bm)
        
        # Get ray from viewport
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
        
        # Unproject to get ray origin and direction in world space
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        
        # Transform to local space
        matrix_inv = obj.matrix_world.inverted()
        ray_origin_local = matrix_inv @ ray_origin
        view_vector_local = matrix_inv.to_3x3() @ view_vector
        
        # Raycast
        location, normal, index, dist = self.bvh.ray_cast(ray_origin_local, view_vector_local)
        
        if location:
            self.selected_face_index = index
            # Select the face in bmesh
            bpy.ops.mesh.select_all(action='DESELECT')
            self.bm.faces.ensure_lookup_table()
            face = self.bm.faces[index]
            face.select = True
            
            # Store initial face center for drag calculation
            self.initial_face_center = face.calc_center_median()
            self.initial_face_center_world = obj.matrix_world @ self.initial_face_center
            
            # Force update to show selection
            bmesh.update_edit_mesh(obj.data)
            return True
            
        return False

    def init_drag(self, context, event):
        # Store initial positions of ALL vertices
        self.initial_verts_co = {v.index: v.co.copy() for v in self.bm.verts}

    def update_drag(self, context, event):
        obj = context.object
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
        
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        
        target_world_pos = None
        
        # Try snapping
        has_hit, hit_loc, hit_normal, hit_index, hit_obj, matrix = context.scene.ray_cast(
            context.view_layer.depsgraph,
            ray_origin,
            view_vector
        )
        
        if has_hit and hit_obj != obj:
            # Snap to this location
            target_world_pos = hit_loc
        else:
            # Fallback: Move on a plane parallel to the view
            C = self.initial_face_center_world
            O = ray_origin
            D = view_vector
            N = view_vector # Plane normal
            
            denom = D.dot(N)
            if abs(denom) > 1e-6:
                t = (C - O).dot(N) / denom
                target_world_pos = O + t * D
            else:
                target_world_pos = C

        # Calculate displacement vector
        mat_inv = obj.matrix_world.inverted()
        displacement_local = (mat_inv @ target_world_pos) - self.initial_face_center
        
        # Apply deformation
        self.apply_deformation(context, displacement_local)
        
        # Update mesh
        bmesh.update_edit_mesh(obj.data)

    def apply_deformation(self, context, displacement_local):
        if self.selected_face_index == -1:
            return

        settings = context.scene.drag_face_settings
        radius = settings.radius
        falloff_power = settings.falloff_power

        face = self.bm.faces[self.selected_face_index]
        face_center = self.initial_face_center
        
        radius_sq = radius * radius
        
        for v_index, original_co in self.initial_verts_co.items():
            vert = self.bm.verts[v_index]
            
            dist_sq = (original_co - face_center).length_squared
            
            if dist_sq > radius_sq:
                vert.co = original_co.copy()
                continue
                
            dist = dist_sq ** 0.5
            
            normalized_dist = dist / radius
            factor = (1.0 - normalized_dist)
            if factor < 0: factor = 0
            
            factor = factor ** falloff_power
            
            vert.co = original_co + displacement_local * factor

    def finish_drag(self, context):
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None
            
        if self.bm:
            bmesh.update_edit_mesh(context.object.data)
        self.bm = None
        self.bvh = None
        self.initial_verts_co = {}

    def cancel_drag(self, context):
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None

        if self.bm:
            for v_index, original_co in self.initial_verts_co.items():
                if v_index < len(self.bm.verts):
                    self.bm.verts[v_index].co = original_co
            bmesh.update_edit_mesh(context.object.data)
        
        self.bm = None
        self.bvh = None
        self.initial_verts_co = {}

    @staticmethod
    def draw_callback_3d(self, context):
        settings = context.scene.drag_face_settings
        radius = settings.radius
        center = self.initial_face_center_world
        
        if center is None: return
        
        # Draw circle facing the view
        if not context.space_data or not isinstance(context.space_data, bpy.types.SpaceView3D):
            return
            
        rv3d = context.space_data.region_3d
        if not rv3d: return
        
        view_mat = rv3d.view_matrix
        view_inv = view_mat.inverted()
        cam_right = view_inv.col[0].xyz.normalized()
        cam_up = view_inv.col[1].xyz.normalized()
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)
        shader.bind()
        shader.uniform_float("color", (1.0, 0.8, 0.0, 0.5))
        
        points = []
        segments = 64
        for i in range(segments + 1):
             angle = 2 * math.pi * i / segments
             offset = (cam_right * math.cos(angle) + cam_up * math.sin(angle)) * radius
             points.append(center + offset)
             
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": points})
        batch.draw(shader)
        
        gpu.state.blend_set('NONE')
