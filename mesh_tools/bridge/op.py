from bmesh.types import BMEdge, BMFace, BMVert
import bpy
import bmesh
from mathutils import Vector

import gpu
from gpu_extras.batch import batch_for_shader

from .config import TRANSITION_PATTERNS

from typing import List, Set, Tuple
from collections import defaultdict


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

    def custom_bridge(self, bm, edges_small, edges_large, cuts, smoothness):
        """
        Custom bridging for unequal edge counts.
        Creates a transition where the smaller loop connects to a grid 
        that matches the larger loop's count, ensuring mostly quads.
        """
        # 1. Order vertices for both loops
        verts_small, closed_small = self.get_ordered_verts(edges_small)
        verts_large, closed_large = self.get_ordered_verts(edges_large)
        
        if not verts_small or not verts_large:
             raise Exception("Could not order vertices")
        
        # Assume both have same closed state for now
        is_closed = closed_small and closed_large
             
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

    def custom_bridge_quads_e5_e3(self, bm, edges_small, edges_large, cuts, smoothness):
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

        first_layer = vertex_grid[0]
        last_layer = vertex_grid[-1]
        tot_verts = max(last_layer) + 1
        last_layer_index = len(vertex_grid) - 1
        cuts = max(last_layer_index, cuts) # len(vertex_grid)
        extra_cuts = cuts - last_layer_index + 1

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

        # Create virtual lines along both lines, this lines should cut the line0 and line1 at the same t value.
        slice_lines = []
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

            ret = self.custom_bridge(bm, edges_small=last_cut_edges, edges_large=edges_large, cuts=extra_cuts, smoothness=smoothness)
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
