from bmesh.types import BMEdge, BMFace, BMVert
import bpy
import bmesh
from mathutils import Vector

import gpu
from gpu_extras.batch import batch_for_shader

from .config import TRANSITION_PATTERNS
from .tool import BridgePlusTool, LINE_SHADER

from typing import List, Set, Tuple
from collections import defaultdict
from itertools import chain
from math import degrees, sin


debug_main_lines = []
debug_crossed_lines = []



class MESH_OT_bridge_plus_debug(bpy.types.Operator):
    """Bridge two edge selections with support for unequal counts and projection"""
    bl_idname = "mesh.bridge_plus_debug"
    bl_label = "Bridge Plus Debug"

    _debug_instance = None

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        if self.__class__._debug_instance:
            self.stop_debug(context)
            self.__class__._debug_instance = None
        else:
            self.__class__._debug_instance = self.start_debug(context)
        return {'FINISHED'}

    def start_debug(self, context: bpy.types.Context):
        context.area.tag_redraw()
        return context.space_data.draw_handler_add(self.__class__.draw_debug, (), 'WINDOW', 'POST_VIEW')

    def stop_debug(self, context: bpy.types.Context):
        context.space_data.draw_handler_remove(self.__class__._debug_instance, 'WINDOW')
        context.area.tag_redraw()

    @staticmethod
    def draw_debug():
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        global debug_main_lines, debug_crossed_lines

        def _draw_line(coords: Tuple[Vector, Vector], color: Tuple[float, float, float, float]):
            batch = batch_for_shader(shader, 'LINES', {"pos": coords})
            shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
            shader.uniform_float("lineWidth", 4.5)
            shader.uniform_float("color", color)
            batch.draw(shader)

        if debug_main_lines:
            for line in debug_main_lines:
                _draw_line(line, color=(1, 1, 0, 1))
        if debug_crossed_lines:
            for line in debug_crossed_lines:
                _draw_line(line, color=(1, 0, 0, 1))

    


class MESH_OT_bridge_plus(bpy.types.Operator):
    """Bridge two edge selections with support for unequal counts and projection"""
    bl_idname = "mesh.bridge_plus"
    bl_label = "Bridge Plus"
    bl_options = {'REGISTER', 'UNDO'}

    use_auto_cuts: bpy.props.BoolProperty(
        name="Auto Cuts",
        description="Automatically determine the number of cuts based on the edge counts",
        default=True
    )

    cuts: bpy.props.IntProperty(
        name="Cuts",
        description="Number of subdivisions",
        default=5,
        min=0
    )

    smoothness: bpy.props.FloatProperty(
        name="Smoothness",
        description="Smoothness factor for the bridge",
        default=1.0,
        min=0.0,
        max=100.0
    )

    use_projection: bpy.props.BoolProperty(
        name="Project Over Meshes",
        description="Project the bridge geometry onto other visible meshes",
        default=True
    )

    projection_mode: bpy.props.EnumProperty(
        name="Projection Mode",
        description="Mode of projection to apply",
        items=[
            ('FACE_PROJECT', "Face Project", "Project onto faces from view"),
            ('CLOSEST_POINT', "Closest Point", "Project onto closest points in mesh"),
        ],
        default='CLOSEST_POINT'
    )

    offset: bpy.props.FloatProperty(
        name="Projection Offset",
        description="Offset from the surface (only for Closest-Point projection)",
        default=0.01
    )

    only_quads: bpy.props.BoolProperty(
        name="Only Quads",
        description="Use only quads for bridging (uses poles for transitions between different edge counts)",
        default=True
    )

    use_tool: bpy.props.BoolProperty(
        name="Use Tool",
        description="Internal property used when we use this operator from a tool",
        default=False,
        options={'SKIP_SAVE', 'SKIP_PRESET', 'HIDDEN'}
    )

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        print(event.type, event.value)
        # Check if the tool is active (keymap properties don't get passed to operators)
        # Import here to avoid circular imports
        is_tool_active = BridgePlusTool.is_active

        # Use tool mode if use_tool is explicitly set OR if the tool is active
        if self.use_tool or is_tool_active:
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.mesh.select_mode(type='EDGE')
            if not BridgePlusTool.select_hovered_edge_loops(context):
                self.report({'ERROR'}, "No edge loops hovered")
                return {'CANCELLED'}
            if context.window_manager.modal_handler_add(self):
                self.start_modal(context, event)
                return {'RUNNING_MODAL'}
            return {'CANCELLED'}
        print("Not using tool")
        return self.execute(context)

    def start_modal(self, context: bpy.types.Context, event: bpy.types.Event):
        self.mouse_start = Vector((event.mouse_region_x, event.mouse_region_y))
        self.mouse_current = self.mouse_start.copy()
        self.draw_path = []  # 3D world-space points
        self.draw_path_2d = []  # 2D pixel-space points
        self._context = context  # Store context for draw handler
        self._draw_handler_2d = context.space_data.draw_handler_add(self.modal_draw_post_pixel, (), 'WINDOW', 'POST_PIXEL')

    def stop_modal(self, context: bpy.types.Context, cancel: bool = False):
        context.space_data.draw_handler_remove(self._draw_handler_2d, 'WINDOW')
        del self._draw_handler_2d

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.stop_modal(context, cancel=True)
            return {'CANCELLED'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                
                return {'RUNNING_MODAL'}
            elif event.value == 'RELEASE':
                if BridgePlusTool.select_hovered_edge_loops(context):
                    self.commit(context)
                self.stop_modal(context, cancel=False)
                return {'FINISHED'}

        elif event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            mouse_current = Vector((event.mouse_region_x, event.mouse_region_y))
            # Delta mouse for a movement threshold of 6px.
            delta_mouse = mouse_current - self.mouse_current
            if delta_mouse.length > 6:
                self.mouse_current = mouse_current
                # Raycast for a new point.
                hit, position, normal, index, object, matrix = BridgePlusTool._scene_raycast(context, self.mouse_current)
                if hit:
                    self.draw_path.append(position)  # Store 3D point
                    self.draw_path_2d.append(mouse_current.copy())  # Store 2D pixel point
                context.region.tag_redraw()

        return {'PASS_THROUGH'}

    def modal_draw_post_pixel(self):
        if not self.draw_path_2d or len(self.draw_path_2d) < 2:
            return
        
        # Calculate cumulative distances along the 2D path
        cumulative_distances = [0.0]
        total_length = 0.0
        for i in range(1, len(self.draw_path_2d)):
            segment_length = (self.draw_path_2d[i] - self.draw_path_2d[i-1]).length
            total_length += segment_length
            cumulative_distances.append(total_length)
        
        if total_length == 0:
            return
        
        # Resample the path with evenly spaced points (one per 10 pixels)
        target_spacing = 10.0  # pixels
        resampled_2d = []
        
        # Always start with the first point
        resampled_2d.append(self.draw_path_2d[0])
        
        # Sample points at regular intervals
        current_distance = target_spacing
        while current_distance < total_length:
            # Find which segment contains this distance
            segment_idx = 0
            for i in range(1, len(cumulative_distances)):
                if cumulative_distances[i] >= current_distance:
                    segment_idx = i - 1
                    break
            
            # Calculate interpolation parameter within the segment
            seg_start_dist = cumulative_distances[segment_idx]
            seg_end_dist = cumulative_distances[segment_idx + 1]
            seg_length = seg_end_dist - seg_start_dist
            
            if seg_length > 0:
                t = (current_distance - seg_start_dist) / seg_length
                t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
                
                # Interpolate in 2D pixel space
                sample_2d = self.draw_path_2d[segment_idx].lerp(self.draw_path_2d[segment_idx + 1], t)
                resampled_2d.append(sample_2d)
            
            current_distance += target_spacing
        
        # Always include the last point
        if len(resampled_2d) == 0 or (resampled_2d[-1] - self.draw_path_2d[-1]).length > 0.001:
            resampled_2d.append(self.draw_path_2d[-1])
        
        if len(resampled_2d) < 2:
            return
        
        # Draw the resampled path using LINE_SHADER
        batch = batch_for_shader(LINE_SHADER, 'LINES', {'pos': resampled_2d})
        LINE_SHADER.bind()
        LINE_SHADER.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
        LINE_SHADER.uniform_float("lineWidth", 6)
        LINE_SHADER.uniform_float('color', (.1, 1, .6, .8))
        batch.draw(LINE_SHADER)

    def commit(self, context: bpy.types.Context):
        self.execute(context)

    def execute(self, context):
        obj = context.edit_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a mesh in Edit Mode")
            return {'CANCELLED'}
            
        me = obj.data
        bm = bmesh.from_edit_mesh(me)

        # 1. Identify components
        selected_edges = [e for e in bm.edges if e.select]
        if not selected_edges:
            self.report({'ERROR'}, "No edges selected")
            return {'CANCELLED'}

        islands = self.get_edge_islands(bm, selected_edges)
        
        if len(islands) != 2:
            self.report({'ERROR'}, f"Need exactly 2 disconnected edge groups. Found {len(islands)}")
            return {'CANCELLED'}

        # 2. Bridge
        # Determine if we use standard bridge or custom interpolation
        # If counts are equal, standard bridge is fine.
        # If counts are unequal, we use our custom interpolation logic if possible.
        # However, bridge_loops is very robust.
        # To get "good flow" for unequal counts, we might want to manually build it.
        
        try:
            # Sort islands by edge count
            if len(islands[0]) > len(islands[1]):
                l1_edges, l2_edges = islands[1], islands[0]
            else:
                l1_edges, l2_edges = islands[0], islands[1]

            # Determine which bridge method to use
            count_small = len(l1_edges)
            count_large = len(l2_edges)
            use_quads_only = self.only_quads and (count_small != count_large)

            if use_quads_only:
                ret = self.custom_bridge_quads_e5_e3(bm, l1_edges, l2_edges, self.cuts, self.smoothness)
            else:
                ret = self.custom_bridge(bm, l1_edges, l2_edges, self.cuts, self.smoothness)

        except Exception as e:
            self.report({'ERROR'}, f"Bridge failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        
        new_faces = ret['faces']

        # 3. Project
        if self.use_projection and self.projection_mode == 'CLOSEST_POINT':
            # Identify new vertices (inner vertices of the bridge)
            # We want to project the vertices that were created by the bridge, 
            # especially the intermediate ones (from cuts).
            # The original boundary vertices should probably stay put? 
            # Usually yes, unless we want to move the whole strip. 
            # Let's assume we only move the NEW vertices.
            
            original_verts: Set[BMVert] = set()
            for e in selected_edges:
                original_verts.update(e.verts)

            new_verts: Set[BMVert] = set()
            for f in new_faces:
                for v in f.verts:
                    if v not in original_verts:
                        new_verts.add(v)

            if new_verts:
                self.project_verts(context, list(new_verts), obj)

        # Recalculate normals to ensure consistency
        bmesh.ops.recalc_face_normals(bm, faces=new_faces)

        # Ensure new faces are selected
        for e in selected_edges:
            e.select = False
        for f in bm.faces:
            f.select = False
        for f in new_faces:
            f.select = True

        bmesh.update_edit_mesh(me)

        # Switch to Face Mode and perform Translate with Face Project Snap
        bpy.ops.mesh.select_mode(type='FACE')
        
        if self.projection_mode == 'FACE_PROJECT':
            bpy.ops.transform.translate(
                value=(0, 0, 0), 
                orient_type='GLOBAL', 
                orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                orient_matrix_type='GLOBAL', 
                mirror=True, 
                snap=True, 
                snap_elements={'FACE_PROJECT'}, 
                use_snap_project=True, 
                snap_target='CLOSEST', 
                use_snap_self=True, 
                use_snap_edit=True, 
                use_snap_nonedit=True, 
                use_snap_selectable=False
            )

        return {'FINISHED'}

    def custom_bridge(self, bm, edges_small, edges_large, cuts, smoothness, path_start_t=0.0):
        """
        Custom bridging for unequal edge counts.
        Creates a transition where the smaller loop connects to a grid 
        that matches the larger loop's count, ensuring mostly quads.
        
        Args:
            path_start_t: Starting parameter (0-1) for path guide when called recursively.
                         0.0 means start from beginning, >0 means we're continuing from a previous bridge.
        """
        # 1. Order vertices for both loops
        verts_small, closed_small = self.get_ordered_verts(edges_small)
        verts_large, closed_large = self.get_ordered_verts(edges_large)
        
        if not verts_small or not verts_large:
             raise Exception("Could not order vertices")
        
        # Assume both have same closed state for now
        is_closed = closed_small and closed_large
        
        # Check if we have a draw_path to use as a guide curve
        use_path_guide = hasattr(self, 'draw_path') and self.draw_path and len(self.draw_path) >= 2
        
        # Prepare path guide if available
        path_sample_func = None
        if use_path_guide:
            path_points = self.draw_path
            path_cumulative_lengths = [0.0]
            total_path_length = 0.0
            for i in range(1, len(path_points)):
                segment_length = (path_points[i] - path_points[i-1]).length
                total_path_length += segment_length
                path_cumulative_lengths.append(total_path_length)
            
            if total_path_length > 0:
                # Remap path_start_t to account for recursive calls
                # path_start_t indicates where we are in the original path (0-1)
                # We need to remap the local t parameter to the global path parameter
                path_t_range = 1.0 - path_start_t  # Remaining portion of path to use
                
                # Helper function to sample a point along the path at parameter t (0 to 1, local to this bridge)
                def sample_path_point(t_param_local):
                    # Remap local t (0-1 for this bridge) to global t (0-1 for entire path)
                    t_param_global = path_start_t + (t_param_local * path_t_range)
                    t_param_global = max(0.0, min(1.0, t_param_global))  # Clamp to [0, 1]
                    
                    target_distance = t_param_global * total_path_length
                    # Find which segment contains this distance
                    segment_idx = 0
                    for j in range(1, len(path_cumulative_lengths)):
                        if path_cumulative_lengths[j] >= target_distance:
                            segment_idx = j - 1
                            break
                    # Interpolate within the segment
                    seg_start_dist = path_cumulative_lengths[segment_idx]
                    seg_end_dist = path_cumulative_lengths[segment_idx + 1]
                    seg_length = seg_end_dist - seg_start_dist
                    if seg_length > 0:
                        seg_t = (target_distance - seg_start_dist) / seg_length
                        seg_t = max(0.0, min(1.0, seg_t))
                        return path_points[segment_idx].lerp(path_points[segment_idx + 1], seg_t)
                    else:
                        return path_points[segment_idx]
                
                path_sample_func = sample_path_point
                
                # Calculate the center points of the start and end edge loops
                center_start = sum((v.co for v in verts_small), Vector((0, 0, 0))) / len(verts_small)
                center_end = sum((v.co for v in verts_large), Vector((0, 0, 0))) / len(verts_large)
                
                # Calculate the path's start and end points for this bridge segment
                # Remap the straight line endpoints based on path_start_t
                path_start_point = sample_path_point(0.0) if path_start_t > 0 else path_points[0]
                path_end_point = sample_path_point(1.0) if path_start_t < 1.0 else path_points[-1]
                
                # Calculate the translation from straight line to path
                # For recursive calls, we need to adjust the straight line reference
                straight_start = center_start
                straight_end = center_end
            else:
                use_path_guide = False
             
        # 2. Align loops (find best start index for large loop)
        verts_large = self.align_loops(verts_small, verts_large, is_closed)
        
        # 3. Create intermediate geometry
        # We want (cuts) intermediate loops.
        # Strategy: The intermediate loops will all have len(verts_large) vertices.
        # This keeps the grid clean. The reduction happens at the connection to verts_small.
        
        new_faces = []
        new_verts = []
        
        # We treat the bridge as a loft from t=0 (small) to t=1 (large)
        # We generate `cuts` intermediate rows.
        # Each intermediate row has len(verts_large) vertices to match the larger side.
        
        rows = []
        rows.append(verts_small) # Row 0
        
        # Create intermediate rows
        count_large = len(verts_large)
        
        for i in range(cuts):
            t = (i + 1) / (cuts + 1)
            row_verts = []
            for j in range(count_large):
                # Map j (large index) to small domain
                # We need to interpolate position
                
                # Position on Large Loop
                v_large = verts_large[j]
                
                # Position on Small Loop
                # Map j to fractional index on small loop
                idx_small_float = j * len(verts_small) / count_large
                idx_small_base = int(idx_small_float)
                factor = idx_small_float - idx_small_base
                
                v_small_a = verts_small[idx_small_base % len(verts_small)]
                if is_closed:
                    v_small_b = verts_small[(idx_small_base + 1) % len(verts_small)]
                else:
                    # Clamp for open
                    idx_next = min(idx_small_base + 1, len(verts_small) - 1)
                    v_small_b = verts_small[idx_next]
                
                co_small = v_small_a.co.lerp(v_small_b.co, factor)
                co_large = v_large.co
                
                # Interpolate between small loop and large loop
                final_co = co_small.lerp(co_large, t) # Linear for now, can use smoothness
                
                # Apply path guide if available
                if use_path_guide and path_sample_func:
                    # Sample point along the path
                    path_center = path_sample_func(t)
                    
                    # Calculate where we would be on the straight line
                    straight_center = straight_start.lerp(straight_end, t)
                    
                    # Calculate the offset from straight line to path
                    path_offset = path_center - straight_center
                    
                    # Apply the path offset to follow the curve
                    final_co = final_co + path_offset
                
                # Create vertex
                new_v = bm.verts.new(final_co)
                row_verts.append(new_v)
                new_verts.append(new_v)
            rows.append(row_verts)
            
        rows.append(verts_large) # Row Last
        
        # 4. Connect Rows
        # Row 0 (Small) to Row 1 (Large count) -> Fan/Reduction
        # Row 1..N (Large count) -> Grid of Quads
        
        # Connect Row 0 to Row 1
        # Row 0 has N verts, Row 1 has M verts. (N < M)
        # We group vertices of Row 1 to connect to edges of Row 0.
        
        # We iterate over Row 1 (the denser one)
        # For each segment in Row 1 (j to j+1), we connect to the nearest segment in Row 0?
        # Better: We iterate over Row 1 indices j.
        # We find corresponding index in Row 0: k = floor(j * N / M)
        
        r0 = rows[0]
        r1 = rows[1]
        len_0 = len(r0)
        len_1 = len(r1)
        
        # Loop limit depends on closed/open
        loop_limit = len_1 if is_closed else len_1 - 1
        
        for j in range(loop_limit):
            j_next = (j + 1) % len_1 if is_closed else j + 1
            
            # Map to Row 0 indices
            # For open loops, we must map strictly range to range
            if is_closed:
                k_curr = int(j * len_0 / len_1) % len_0
                k_next = int((j + 1) * len_0 / len_1) % len_0
            else:
                k_curr = min(int(j * len_0 / len_1), len_0 - 1)
                k_next = min(int((j + 1) * len_0 / len_1), len_0 - 1)
            
            v1 = r1[j]
            v2 = r1[j_next]
            u1 = r0[k_curr]
            
            if k_curr == k_next:
                # Both match to same u1.
                # Triangle: u1 - v1 - v2
                try:
                    f = bm.faces.new((u1, v1, v2))
                    new_faces.append(f)
                except ValueError: pass 
            else:
                # Step occurred. u1 changes to u2.
                # Quad (u1, v1, v2, u2)
                u2 = r0[k_next]
                try:
                    f = bm.faces.new((u1, v1, v2, u2))
                    new_faces.append(f)
                except ValueError: pass
        
        # Connect remaining rows (1 to End)
        # All have same count -> Quads
        for i in range(1, len(rows) - 1):
            curr_row = rows[i]
            next_row = rows[i+1]
            count = len(curr_row)
            limit = count if is_closed else count - 1
            
            for j in range(limit):
                j_next = (j + 1) % count if is_closed else j + 1
                
                v1 = curr_row[j]
                v2 = curr_row[j_next]
                v3 = next_row[j_next]
                v4 = next_row[j]
                
                try:
                    f = bm.faces.new((v1, v2, v3, v4))
                    new_faces.append(f)
                except ValueError: pass

        return {'faces': new_faces}

    def custom_bridge_quads_e5_e3(self, bm: bmesh.types.BMesh, edges_small: List[BMEdge], edges_large: List[BMEdge], cuts: int, smoothness: float):
        """
        Bridge with only quads using poles (E5/E3) for smooth transitions.
        E5 poles (valence 5) increase edge count by +1
        E3 poles (valence 3) decrease edge count by -1
        """
        # Find the ideal the pattern of the edge count for both ends.
        small_count = len(edges_small)
        large_count = len(edges_large)

        transition_pattern = TRANSITION_PATTERNS.get(f'{small_count}_to_{large_count}')
        if not transition_pattern:
            raise Exception(f"No transition pattern found for {small_count} to {large_count}")

        # Create the geometry.
        new_faces: List[BMFace] = []
        vertex_grid: List[List[int | None]] = transition_pattern['vertex_grid']
        face_indices: List[List[int]] = transition_pattern['face_indices']

        # Get vertices for first and last layers, in order.
        first_layer_vertices = self.walk_vertices_along_edges(edges_small)
        last_layer_vertices = self.walk_vertices_along_edges(edges_large)

        # Get the ends of both edge loops.
        v0_small = first_layer_vertices[0]
        v0_large = last_layer_vertices[0]
        v1_small = first_layer_vertices[-1]
        v1_large = last_layer_vertices[-1]

        # See if v0_small is better aligned with v0_large or v1_large.
        # Closer to the same end (by index) or the other one?
        # If to the opposite, then we need to reverse one of the lists.
        d0 = (v0_small.co - v0_large.co).length
        d1 = (v1_small.co - v0_large.co).length
        if d0 > d1:
            v0_large, v1_large = v1_large, v0_large
            last_layer_vertices = last_layer_vertices[::-1]

        # Virtual lines between v0_small and v0_large, and v0_small and v1_large.
        line0 = (v0_small.co, v0_large.co)
        line1 = (v1_small.co, v1_large.co)
        
        # Check if we have a draw_path to use as a guide curve
        use_path_guide = hasattr(self, 'draw_path') and self.draw_path and len(self.draw_path) >= 2
        
        # Calculate the number of cuts needed.
        first_layer = vertex_grid[0]
        last_layer = vertex_grid[-1]
        tot_verts = max(last_layer) + 1
        last_layer_index = len(vertex_grid) - 1

        if self.use_auto_cuts:
            """
            Calculate the number of cuts needed to bridge the first and last layer based on quad size of both ends.
            """
            end_faces: Set[BMFace] = {f for e in chain(edges_small, edges_large) for f in e.link_faces}
            median_quad_area: float = sum([f.calc_area() for f in end_faces]) / len(end_faces)
            target_quad_side: float = median_quad_area ** 0.5
            # Line from the middle of first layer to the middle of last_layer.
            mid_first = v0_small.co.lerp(v1_small.co, 0.5)
            mid_last = v0_large.co.lerp(v1_large.co, 0.5)

            # Straight-line distance between the midpoints of the two edge loops
            mid_line_length = (mid_last - mid_first).length

            # Direction vectors of the two edge loops (using face normals as reference)
            median_f_normal_small: Vector = sum([f.normal for e in edges_small for f in e.link_faces], Vector((0.0, 0.0, 0.0))) / len(edges_small)
            median_f_normal_large: Vector = sum([f.normal for e in edges_large for f in e.link_faces], Vector((0.0, 0.0, 0.0))) / len(edges_large)

            dir_small = median_f_normal_small.normalized()
            dir_large = median_f_normal_large.normalized()

            # Angle between the two loop directions (always 0 to 180 degrees)
            angle_radians = dir_small.angle(dir_large)
            angle_degrees = degrees(angle_radians)

            # Base number of cuts assuming straight bridge (flat)
            flat_segments = mid_line_length // target_quad_side
            flat_cuts = int(flat_segments) # + 1  # +1 to include the ending layer

            # Check if all end_faces are coplanar (axis-aligned)
            end_verts_coords: List[Vector] = [v.co for f in end_faces for v in f.verts]
            # Check for constant X, Y, or Z
            eps = 0.0001
            one_plane: bool = any([
                (max(co.x for co in end_verts_coords) - min(co.x for co in end_verts_coords)) < eps,
                (max(co.y for co in end_verts_coords) - min(co.y for co in end_verts_coords)) < eps,
                (max(co.z for co in end_verts_coords) - min(co.z for co in end_verts_coords)) < eps,
            ])

            if one_plane or angle_degrees < 1.0:  # Treat nearly parallel as flat
                cuts = flat_cuts
            else:
                # Adjust for curvature/bend using chord length approximation
                # This models the bridge as two straight halves meeting at the bend angle
                # Effective path length ≈ 2 * distance * sin(θ/2)
                # This gives more segments when bent (good for sharp angles), fewer when almost flat
                chord_length = 2.0 * mid_line_length * sin(angle_radians / 2.0)
                curved_segments = chord_length // target_quad_side
                cuts = int(curved_segments)  # max(flat_cuts, int(curved_segments) + 1)  # At least the flat amount

                # Optional: add a small safety buffer for very sharp bends
                # if angle_degrees > 90:
                #     cuts += 1

            # Final cuts value (number of intermediate layers + 1 for the end)
            # cuts = cuts + 1

        # Clamp the number of cuts to the number of vertices in the last layer.
        cuts = max(last_layer_index, cuts) # len(vertex_grid)
        extra_cuts = cuts - last_layer_index + 1
        
        # Calculate path_start_t for recursive custom_bridge call
        # We've created last_layer_index layers, so we're at t = last_layer_index / (cuts + 1) in the path
        path_start_t_for_extra = last_layer_index / (cuts + 1) if cuts > 0 else 0.0

        # Create virtual lines along both lines, this lines should cut the line0 and line1 at the same t value.
        # If we have a path guide, use it to curve the bridge instead of straight lines.
        slice_lines = []
        if use_path_guide:
            # Parameterize the path by arc length
            path_points = self.draw_path
            path_cumulative_lengths = [0.0]
            total_path_length = 0.0
            for i in range(1, len(path_points)):
                segment_length = (path_points[i] - path_points[i-1]).length
                total_path_length += segment_length
                path_cumulative_lengths.append(total_path_length)
            
            if total_path_length > 0:
                # Helper function to sample a point along the path at parameter t (0 to 1)
                def sample_path_point(t_param):
                    target_distance = t_param * total_path_length
                    # Find which segment contains this distance
                    segment_idx = 0
                    for j in range(1, len(path_cumulative_lengths)):
                        if path_cumulative_lengths[j] >= target_distance:
                            segment_idx = j - 1
                            break
                    # Interpolate within the segment
                    seg_start_dist = path_cumulative_lengths[segment_idx]
                    seg_end_dist = path_cumulative_lengths[segment_idx + 1]
                    seg_length = seg_end_dist - seg_start_dist
                    if seg_length > 0:
                        seg_t = (target_distance - seg_start_dist) / seg_length
                        seg_t = max(0.0, min(1.0, seg_t))
                        return path_points[segment_idx].lerp(path_points[segment_idx + 1], seg_t)
                    else:
                        return path_points[segment_idx]
                
                # Calculate the center points of the start and end edge loops
                center_start = v0_small.co.lerp(v1_small.co, 0.5)
                center_end = v0_large.co.lerp(v1_large.co, 0.5)
                
                # Calculate the path's start and end points
                path_start = path_points[0]
                path_end = path_points[-1]
                
                # Calculate the translation from straight line to path
                straight_start = center_start
                straight_end = center_end
                
                for i in range(cuts):
                    t = (i + 1) / (cuts + 1)
                    
                    # Sample point along the path (this is the center of the bridge at this slice)
                    path_center = sample_path_point(t)
                    
                    # Calculate where we would be on the straight line
                    straight_center = straight_start.lerp(straight_end, t)
                    
                    # Calculate the offset from straight line to path
                    path_offset = path_center - straight_center
                    
                    # Calculate the straight line slice endpoints
                    straight_slice_start = self.sample_point_in_line(*line0, t)
                    straight_slice_end = self.sample_point_in_line(*line1, t)
                    
                    # Apply the path offset to follow the curve
                    slice_start = straight_slice_start + path_offset
                    slice_end = straight_slice_end + path_offset
                    
                    slice_lines.append((slice_start, slice_end))
            else:
                # Path has zero length, fall back to straight lines
                use_path_guide = False
        
        if not use_path_guide:
            # No path guide or path processing failed, use straight lines
            for i in range(cuts):
                t = (i + 1) / (cuts + 1)
                slice_lines.append((
                    self.sample_point_in_line(*line0, t),
                    self.sample_point_in_line(*line1, t)
                ))

        global debug_main_lines, debug_crossed_lines
        debug_main_lines = (line0, line1)
        debug_crossed_lines = slice_lines

        ## print("last_layer_index", last_layer_index)
        ## print("first_layer", first_layer)
        ## print("last_layer", last_layer)
        ## print("line0", line0)
        ## print("line1", line1)
        ## print("slice_lines", slice_lines)

        # Add verts.
        new_verts = first_layer_vertices
        for layer_index, vert_indices in enumerate(vertex_grid):
            if (layer_index == last_layer_index and extra_cuts == 0) or layer_index == 0:
                continue
            layer_slice_count = len(vert_indices) - 1
            slice_line = slice_lines[layer_index-1]

            ## print("layer_index", layer_index-1)
            ## print("\t- vert_indices", vert_indices)
            ## print("\t- layer_slice_count", layer_slice_count)
            ## print("\t- slice_line", slice_line)
            
            for slice_position, v in enumerate(vert_indices):
                if v < 0: continue
                slice_t = slice_position / layer_slice_count
                point = self.sample_point_in_line(*slice_line, slice_t)
                ## print("\t- slice_position", slice_position)
                ## print("\t- slice_t", slice_t)
                ## print("\t- point", point)
                new_verts.append(bm.verts.new(point))

        if extra_cuts > 0:
            pass
        else:
            new_verts.extend(last_layer_vertices)

        # Create faces.
        for face_index in face_indices:
            new_faces.append(bm.faces.new([
                new_verts[i] for i in face_index
            ]))

        if extra_cuts > 0:
            last_cut_verts = new_verts[-len(last_layer_vertices):]
            last_cut_edges = []
            for v in last_cut_verts:
                for e in v.link_edges:
                    if e not in last_cut_edges and len(e.link_faces) == 1 and e.verts[0] in last_cut_verts and e.verts[1] in last_cut_verts:
                        last_cut_edges.append(e)

            ret = self.custom_bridge(bm, edges_small=last_cut_edges, edges_large=edges_large, cuts=extra_cuts, smoothness=smoothness, path_start_t=path_start_t_for_extra)
            if ret and 'faces' in ret:
                new_faces.extend(ret['faces'])
            else:
                print("Error bridging extra cuts")

        return {'faces': new_faces}

    def walk_vertices_along_edges(self, edges):
        vertices = defaultdict(list)
        for e in edges:
            v1, v2 = e.verts
            vertices[v1].append(e)
            vertices[v2].append(e)

        # Find first vert to walk.
        first_vert = None
        for v, edges in vertices.items():
            if len(edges) == 1:
                first_vert = v
                break

        def _walk_vert_over_edges(v: BMVert, _edges: Set[BMEdge], visited: Set[BMVert], vertices_walk: List[BMVert]):
            if v in visited: return vertices_walk
            visited.add(v)
            vertices_walk.append(v)
            for e in v.link_edges:
                if e not in _edges: continue
                v1, v2 = e.verts
                if v1 != v:
                    _walk_vert_over_edges(v1, vertices[v1], visited, vertices_walk)
                if v2 != v:
                    _walk_vert_over_edges(v2, vertices[v2], visited, vertices_walk)
            return vertices_walk

        return _walk_vert_over_edges(first_vert, set(edges), set(), [])

    def sample_n_points_along_line(self, a, b, count):
        # Sample 'count' 3d points along the line_a and line_b.
        points = []
        for i in range(count):
            t = (i + 1) / count
            points.append(a.lerp(b, t))
        return points
    
    def sample_point_in_line(self, a, b, t):
        # Sample a point in the line from a to b at t.
        return a.lerp(b, t)

    def get_ordered_verts(self, edges):
        # Convert edge soup to ordered verts
        # Simple DFS/Walk
        if not edges: return [], False
        
        # Build adjacency
        adj = {}
        for e in edges:
            v1, v2 = e.verts
            if v1 not in adj: adj[v1] = []
            if v2 not in adj: adj[v2] = []
            adj[v1].append(v2)
            adj[v2].append(v1)
            
        # Find start (valence 1 if open)
        start_node = edges[0].verts[0]
        is_closed = True
        
        # Check for endpoints (valence 1)
        leaves = [v for v, n in adj.items() if len(n) == 1]
        if leaves:
            is_closed = False
            start_node = leaves[0]
            
        # Walk
        path = [start_node]
        visited = {start_node}
        curr = start_node
        
        while True:
            neighbors = adj.get(curr, [])
            next_node = None
            for n in neighbors:
                if n not in visited:
                    next_node = n
                    break
            
            if next_node:
                visited.add(next_node)
                path.append(next_node)
                curr = next_node
            else:
                break
                
        return path, is_closed

    def align_loops(self, loop_a, loop_b, is_closed):
        # Rotate loop_b to minimize distance to loop_a
        
        if is_closed:
            # Heuristic: minimize dist(a[0], b[k])
            best_shift = 0
            min_dist = float('inf')
            v0 = loop_a[0].co
            
            for i, v in enumerate(loop_b):
                d = (v.co - v0).length
                if d < min_dist:
                    min_dist = d
                    best_shift = i
                    
            new_b = loop_b[best_shift:] + loop_b[:best_shift]
            
            # Check direction
            if len(loop_a) > 1:
                v_a = loop_a[1].co - loop_a[0].co
                v_b = new_b[1].co - new_b[0].co
                if v_a.dot(v_b) < 0:
                    # Reverse b, keeping pivot at 0
                    # pivot is new_b[0]
                    pivot = new_b[0]
                    rest = new_b[1:]
                    rest.reverse()
                    new_b = [pivot] + rest
        else:
            # Open strips: Check if we need to reverse B
            # Compare dist(A_start, B_start) vs dist(A_start, B_end)
            d_normal = (loop_a[0].co - loop_b[0].co).length
            d_reversed = (loop_a[0].co - loop_b[-1].co).length
            
            if d_reversed < d_normal:
                new_b = list(reversed(loop_b))
            else:
                new_b = loop_b
                
        return new_b

    def get_edge_islands(self, bm, edges):
        """
        Identify edge islands by checking vertex connectivity.
        Optimized to use a vert -> group index mapping.
        """
        if not edges:
            return []
            
        # 1. Map each edge to its vertices for quick access (already available via edge.verts)
        # 2. Build a mapping of vert_index -> group_id
        vert_to_group = {}
        groups = [] # List of lists of edges
        
        # We iterate through edges and assign them to groups based on vertex connectivity
        for e in edges:
            v1_idx = e.verts[0].index
            v2_idx = e.verts[1].index
            
            g1 = vert_to_group.get(v1_idx)
            g2 = vert_to_group.get(v2_idx)
            
            if g1 is None and g2 is None:
                # New group
                new_group_id = len(groups)
                groups.append([e])
                vert_to_group[v1_idx] = new_group_id
                vert_to_group[v2_idx] = new_group_id
                
            elif g1 is not None and g2 is None:
                # Add to g1
                groups[g1].append(e)
                vert_to_group[v2_idx] = g1
                
            elif g1 is None and g2 is not None:
                # Add to g2
                groups[g2].append(e)
                vert_to_group[v1_idx] = g2
                
            elif g1 is not None and g2 is not None:
                if g1 == g2:
                    # Already same group, just add edge
                    groups[g1].append(e)
                else:
                    # Merge groups!
                    # We merge g2 into g1
                    target_group_id = g1
                    source_group_id = g2
                    
                    # Move all edges from source to target
                    groups[target_group_id].extend(groups[source_group_id])
                    groups[source_group_id] = [] # Clear source, we'll filter empty later
                    
                    # Update all verts pointing to source to point to target
                    # This is the expensive part of this simple union-find approach
                    # But for selection sizes it's negligible
                    for v_idx, g_id in vert_to_group.items():
                        if g_id == source_group_id:
                            vert_to_group[v_idx] = target_group_id
                            
                    # Add current edge
                    groups[target_group_id].append(e)
        
        # Filter out empty groups (from merges)
        final_islands = [g for g in groups if g]
        return final_islands

    def project_verts(self, context, verts, obj):
        # Gather collision objects
        # We use depsgraph to get evaluated objects (proper world space positions)
        depsgraph = context.evaluated_depsgraph_get()
        
        targets = []
        for o in context.visible_objects:
            if o.type == 'MESH' and o != obj:
                # Get evaluated object
                eval_obj = o.evaluated_get(depsgraph)
                targets.append(eval_obj)
                
        if not targets: return
        
        mw = obj.matrix_world
        mwi = mw.inverted()
        
        for v in verts:
            world_co = mw @ v.co
            best_loc = None
            min_dist = float('inf')
            best_normal = None
            
            for target in targets:
                # closest_point_on_mesh works in object local space of the target
                # so we need to transform point to target local
                target_mw = target.matrix_world
                target_mwi = target_mw.inverted()
                
                success, pt_local, normal_local, _ = target.closest_point_on_mesh(target_mwi @ world_co)
                
                if success:
                    pt_world = target_mw @ pt_local
                    
                    dist = (pt_world - world_co).length
                    if dist < min_dist:
                        min_dist = dist
                        best_loc = pt_world
                        
                        # Transform normal to world
                        # Normal transform: (inverse transpose of upper 3x3)
                        # Or simpler: rotation part of matrix
                        best_normal = target_mw.to_3x3() @ normal_local
                        best_normal.normalize()
            
            if best_loc:
                # Apply offset
                final_pos = best_loc + (best_normal * self.offset)
                v.co = mwi @ final_pos
