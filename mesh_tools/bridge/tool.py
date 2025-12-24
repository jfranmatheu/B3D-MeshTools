from dataclasses import dataclass
from math import degrees, dist
from bmesh.types import BMEdge, BMFace, BMVert, BMesh
import bpy
import bmesh
from bpy.types import WorkSpace, WorkSpaceTool
from gpu.types import GPUBatch
from mathutils import Color, Matrix, Vector

import gpu
from gpu_extras.batch import batch_for_shader
from gpu_extras.presets import draw_circle_2d
from bpy_extras.view3d_utils import location_3d_to_region_2d, region_2d_to_vector_3d, region_2d_to_origin_3d

from colorsys import rgb_to_hsv, hsv_to_rgb
from typing import List, Optional, Set, Tuple, Callable, Dict
from dataclasses import field, dataclass
from time import time

from mathutils.bvhtree import BVHTree

from ..utils.geometry import is_point_inside_circle, line_segment_inside_or_intersecting_circle, dist_to_segment


if not bpy.app.background:
    LINE_SHADER = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    LINE_WIDTH = 6
    ACTIVE_LINE_WIDTH = 8
    EDGE_CO_DRAW_OFFSET = 0.01
    MIN_MOUSE_MOVE_DISTANCE = 6  # pixels
    CURSOR_RADIUS = 36  # pixels
    FACE_NORMAL_ANGLE_THRESHOLD = 1  # angle threshold in degrees
    PICK_METHOD = 'SCENE_RAYCAST' # 'BVH_TREE_RAYCAST', 'SCENE_RAYCAST'
    USE_RANDOM_COLORS = False


@dataclass
class EdgeCandidate:
    index: int
    verts_indices: Tuple[int, int]
    verts_co: Tuple[Vector, Vector]
    face_index: int
    face_normal: Vector
    verts_co_2d: Tuple[Vector, Vector] = field(default_factory=lambda: (Vector((-100000, -100000)), Vector((-100000, -100000))))
    
    def __hash__(self) -> int:
        return hash((self.index, self.face_index))

    @property
    def verts_co_draw(self):
        return (self.verts_co[0] + self.face_normal * EDGE_CO_DRAW_OFFSET, self.verts_co[1] + self.face_normal * EDGE_CO_DRAW_OFFSET)

    def pick_test(self, mouse_pos: Vector, hit_position: Vector, face_normal: Vector, face_index: int, distance: float) -> bool:
        # Check if the edge belongs to the same face as the hovered face
        if self.face_index != face_index:
            return False

        # Check if the mouse position is within the edge segment
        if not line_segment_inside_or_intersecting_circle(*self.verts_co_2d, mouse_pos, CURSOR_RADIUS):
            return False

        return True


@dataclass
class EdgeLoopCandidate:
    edges: List[EdgeCandidate] = field(default_factory=list)
    draw_batch: GPUBatch = None
    hovered: bool = False
    color: Color = Color((1, 1, 0))

    def __hash__(self) -> int:
        return hash(tuple(self.edges))

    def add_edge(self, edge_cand: BMEdge, mw: Matrix):
        coords = (mw @ edge_cand.verts[0].co, mw @ edge_cand.verts[1].co)
        self.edges.append(EdgeCandidate(
            index=edge_cand.index,
            verts_indices=(edge_cand.verts[0].index, edge_cand.verts[1].index),
            verts_co=coords,
            face_index=edge_cand.link_faces[0].index,
            face_normal=edge_cand.link_faces[0].normal.normalized()
        ))

    def setup_draw_batch(self):
        # Flatten the vertex coordinates: each edge has (Vector, Vector), we need a flat list
        # For LINES mode, each pair of consecutive vectors represents a line segment
        positions = [v for edge in self.edges for v in edge.verts_co_draw]
        self.draw_batch = batch_for_shader(
            LINE_SHADER,
            'LINES',
            {"pos": positions}
        )

    def pick_test(self, mouse_pos: Vector, position: Vector, normal: Vector, index: int, distance: float) -> bool:
        for edge in self.edges:
            if edge.pick_test(mouse_pos, position, normal, index, distance):
                print(f"Edge {edge.index} picked")
                return True
        return False

    def draw(self):
        if self.draw_batch:
            LINE_SHADER.bind()
            LINE_SHADER.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
            LINE_SHADER.uniform_float("lineWidth", ACTIVE_LINE_WIDTH if self.hovered else LINE_WIDTH)
            LINE_SHADER.uniform_float("color", (*self.color, 1 if self.hovered else 0.5) if USE_RANDOM_COLORS else (.1, 1, .6, .92) if self.hovered else (1, 1, 1, .5))
            self.draw_batch.draw(LINE_SHADER)


class BridgePlusTool(bpy.types.WorkSpaceTool):
    bl_idname = "mesh_tools.bridge_plus"
    bl_label = "Bridge Tool"
    bl_description = "Bridge faces with geometric resistance"
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'EDIT_MESH'
    bl_icon = 'ops.transform.translate'
    bl_widget = None
    bl_keymap = (
        ("mesh.bridge_plus", {"type": 'LEFTMOUSE', "value": 'PRESS'}, {"use_tool": True}),
    )

    is_active = False
    potential_edge_loops: List[EdgeLoopCandidate] = []
    edge_to_edge_loop: Dict[EdgeCandidate, EdgeLoopCandidate] = None
    _draw_post_view_handler = None
    _cursor_handler = None
    _hovered_edge_loop: EdgeLoopCandidate = None
    last_mouse_pos: Vector = Vector((0, 0))
    current_mouse_pos: Vector = Vector((0, 0))
    bm: bmesh.types.BMesh = None
    bvh_tree: BVHTree = None

    last_view_location: tuple = (0, 0, 0)
    last_view_rotation: tuple = (0, 0, 0, 0)
    last_view_distance: float
    last_view_change_time: float = 0.0

    @classmethod
    def enable(cls, context: bpy.types.Context):
        print("Enable BridgePlusTool")
        if cls.is_active:
            return
        cls.is_active = True
        cls._hovered_edge_loop = None
        cls.update_edge_loops_candidates(context)
        cls._draw_post_view_handler = context.space_data.draw_handler_add(cls.modal_draw_post_view, (context,), 'WINDOW', 'POST_VIEW')
        # cls._cursor_handler = context.window_manager.draw_cursor_add(cls.draw_cursor, (context, ), 'VIEW_3D', 'WINDOW')  # no needed when the tool is active, Blender does register it for us!
        context.window.cursor_set('CROSSHAIR')
        context.region.tag_redraw()
        cls.ensure_bmesh(context)

    @classmethod
    def disable(cls, context: bpy.types.Context):
        print("Disable BridgePlusTool")
        if not cls.is_active:
            return
        cls.is_active = False
        cls.potential_edge_loops = []
        cls.edge_to_edge_loop = None
        context.space_data.draw_handler_remove(cls._draw_post_view_handler, 'WINDOW')
        cls._draw_post_view_handler = None
        # context.window_manager.draw_cursor_remove(cls._cursor_handler)
        # cls._cursor_handler = None
        context.window.cursor_set('DEFAULT')
        context.region.tag_redraw()

    @classmethod
    def ensure_bmesh(cls, context: bpy.types.Context) -> bmesh.types.BMesh:
        obj = context.edit_object
        if not obj or obj.type != 'MESH':
            print("Active object must be a mesh in Edit Mode")
            return None
        if cls.bm is not None and cls.bm.is_valid:
            return cls.bm
        me = obj.data
        bm: bmesh.types.BMesh = bmesh.from_edit_mesh(me)
        if bm is None or not bm.is_valid:
            print("BMesh is not valid")
            return None
        bm.verts.index_update()
        bm.edges.index_update()
        bm.faces.index_update()
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        cls.bm = bm
        return cls.bm

    @classmethod
    def ensure_bvh_tree(cls, context: bpy.types.Context) -> BVHTree:
        bm = cls.ensure_bmesh(context)
        if bm is None or not bm.is_valid:
            return None
        if cls.bvh_tree is not None:
            return cls.bvh_tree
        bvh_tree: BVHTree = BVHTree.FromBMesh(bm)
        cls.bvh_tree = bvh_tree
        return bvh_tree

    @classmethod
    def select_hovered_edge_loops(cls, context: bpy.types.Context) -> bool:
        if cls._hovered_edge_loop is None:
            return False
        bm = cls.ensure_bmesh(context)
        if bm is None or not bm.is_valid:
            return False
        try:
            for edge in cls._hovered_edge_loop.edges:
                bm.edges[edge.index].select_set(True)
        except Exception as e:
            print(e)
            return False
        return True

    @classmethod
    def draw_cursor(cls, context: bpy.types.Context, tool: WorkSpaceTool, xy: Optional[Tuple[int, int]] = None):
        if not cls.is_active:
            return

        gpu.state.line_width_set(2.0)
        draw_circle_2d(xy, (1, 1, 0, 0.5), CURSOR_RADIUS, segments=32)
        gpu.state.line_width_set(1.0)

        if xy is None:
            return

        # Calculate the delta mouse position
        mouse_pos = Vector(xy) - Vector((context.area.x, context.area.y))
        delta_mouse_pos = mouse_pos - cls.last_mouse_pos
        if delta_mouse_pos.length_squared < MIN_MOUSE_MOVE_DISTANCE:
            return

        # NAVIGATION HACK.
        reg3d = context.region_data
        if cls.last_view_location != reg3d.view_location.to_tuple() or\
           cls.last_view_rotation != tuple(reg3d.view_rotation) or\
           cls.last_view_distance != reg3d.view_distance:
            cls.last_view_location = reg3d.view_location.to_tuple()
            cls.last_view_rotation = tuple(reg3d.view_rotation)
            cls.last_view_distance = reg3d.view_distance
            cls.last_view_change_time = time()
            cls.update_edge_loops_2d_coords(context) # if view changes, we need to update 2D projections!
            return
        elif time() - cls.last_view_change_time < 0.15:
            return

        # Update mouse position.
        cls.last_mouse_pos = cls.current_mouse_pos
        cls.current_mouse_pos = mouse_pos

        # Pick the edge loop under the mouse cursor, if any.
        cls.pick_edge_loop(context, mouse_pos)

    @classmethod
    def _get_raycast_origin_and_direction(cls, context: bpy.types.Context, mouse_pos: Vector) -> Tuple[Vector, Vector]:
        mw = context.object.matrix_world
        loc, rot, sca = mw.decompose()
        mat_origin = Matrix.LocRotScale(Vector((0, 0, 0)), rot, sca)
        mwi = mat_origin.inverted()
        view_vector = mwi @ region_2d_to_vector_3d(context.region, context.region_data, mouse_pos).normalized()
        ray_origin = mwi @ (region_2d_to_origin_3d(context.region, context.region_data, mouse_pos) - loc)
        return ray_origin, view_vector
    
    
    @classmethod
    def pick_edge_loop(cls, context: bpy.types.Context, mouse_pos: Vector):
        hovered_edge_loop = None
        if PICK_METHOD == 'BVH_TREE_RAYCAST':
            hovered_edge_loop = cls.pick_edge_loop_with_bvh_tree(context, mouse_pos)
        elif PICK_METHOD == 'SCENE_RAYCAST':
            hovered_edge_loop = cls.pick_edge_loop_with_scene_raycast(context, mouse_pos)

        if hovered_edge_loop == cls._hovered_edge_loop:
            # Same result...
            return

        if cls._hovered_edge_loop:
            cls._hovered_edge_loop.hovered = False
        cls._hovered_edge_loop = hovered_edge_loop
        if hovered_edge_loop:
            hovered_edge_loop.hovered = True
        context.region.tag_redraw()

    @classmethod
    def pick_edge_loop_with_bvh_tree(cls, context: bpy.types.Context, mouse_pos: Vector) -> Optional[EdgeLoopCandidate]:
        bvh_tree = cls.ensure_bvh_tree(context)
        if bvh_tree is None:
            return None

        ray_origin, view_vector = cls._get_raycast_origin_and_direction(context, mouse_pos)
        position, normal, index, distance = bvh_tree.ray_cast(ray_origin, view_vector)
        if position is None:
            # print("No hit")
            return None
        if normal.dot(view_vector) > 0:
            # print("Hit face is not facing the camera")
            return None
        # print(f"Hit position: {position}, normal: {normal}, face index: {index}, distance: {distance}")
        hovered_edge_loop = None
        for edge_loop_cand in cls.potential_edge_loops:
            if edge_loop_cand.pick_test(mouse_pos, position, normal, index, distance):
                hovered_edge_loop = edge_loop_cand
                break
        return hovered_edge_loop
    
    @staticmethod
    def _scene_raycast(context: bpy.types.Context, mouse_pos: Vector):
        # scene.ray_cast expects world-space coordinates, not object-space
        scene = context.scene
        depsgraph = context.evaluated_depsgraph_get()
        # Get world-space ray origin and direction
        ray_origin_world = region_2d_to_origin_3d(context.region, context.region_data, mouse_pos)
        view_vector_world = region_2d_to_vector_3d(context.region, context.region_data, mouse_pos).normalized()

        # Returns (hit, position, normal, index, object, matrix)
        return scene.ray_cast(depsgraph, ray_origin_world, view_vector_world)

    @classmethod
    def pick_edge_loop_with_scene_raycast(cls, context: bpy.types.Context, mouse_pos: Vector) -> Optional[EdgeLoopCandidate]:
        # Returns (hit, position, normal, index, object, matrix)
        hit, position, normal, index, object, matrix = cls._scene_raycast(context, mouse_pos)

        if not hit:
            return None
        
        view_vector_world = region_2d_to_vector_3d(context.region, context.region_data, mouse_pos).normalized()

        # Check if face is facing the camera (normal should point opposite to view direction)
        if normal.dot(view_vector_world) > 0:
            return None
        
        hit_edit_ob = object == context.edit_object
        mw = context.edit_object.matrix_world
        
        closest_edge_loop = None
        closest_distance = float('inf')
        
        for edge, edge_loop in cls.edge_to_edge_loop.items():
            # If we hit the edit object, only consider edges from the hit face
            if hit_edit_ob and edge.face_index != index:
                continue
            
            # Transform edge face normal to world space for comparison
            edge_face_normal_world = (mw.to_3x3() @ edge.face_normal).normalized()
            
            # Check if edge face normal is facing away from the view
            if edge_face_normal_world.dot(view_vector_world) > 0:
                continue
            
            # Check if line segment is inside or intersecting the cursor circle (2D check)
            if not line_segment_inside_or_intersecting_circle(*edge.verts_co_2d, mouse_pos, CURSOR_RADIUS):
                continue
            
            # Check distance from hit position to edge segment (in world space)
            distance = dist_to_segment(position, *edge.verts_co)
            if distance < closest_distance:
                closest_edge_loop = edge_loop
                closest_distance = distance

        return closest_edge_loop

    @classmethod
    def update_edge_loops_candidates(cls, context: bpy.types.Context):
        mw = context.edit_object.matrix_world
        bm = bmesh.from_edit_mesh(context.edit_object.data)
        # Get all boundary edge loops (edges with only 1 face linked)
        # Edges must be connected, adjacent edges must have different faces, and each edge must be a boundary
        def _validate_edge(_e: BMEdge) -> bool:
            return _e.is_valid and _e.is_boundary
        
        cls.potential_edge_loops: List[EdgeLoopCandidate] = []
        walked_edges: Set[int] = set()
        
        # Iterate through all edges to find boundary edge loops
        for edge in bm.edges:
            # Skip if already walked or not a valid boundary edge
            if edge.index in walked_edges or not _validate_edge(edge):
                continue
                
            edge_loop_cand = EdgeLoopCandidate()
            edge_loop_cand.add_edge(edge, mw)
            walked_edges.add(edge.index)
            
            # Walk along the boundary edge loop by following vertices
            # Start from one vertex of the edge and follow the boundary
            current_edge = edge
            prev_edge = None  # Track previous edge for determining which vertex to follow
            
            # Walk in one direction along the boundary
            while True:
                # Get the vertex at the "end" of the current edge (not shared with previous edge)
                # For the first iteration, pick one vertex to start from
                if prev_edge is None:
                    current_vertex = edge.verts[1]  # Start from one vertex
                else:
                    # Find the vertex that's not shared with the previous edge
                    prev_edge_verts = set(prev_edge.verts)
                    current_vertex = [v for v in current_edge.verts if v not in prev_edge_verts][0]
                
                # Find the next boundary edge connected to current_vertex
                next_edge = None
                for linked_edge in current_vertex.link_edges:
                    if (linked_edge != current_edge and _validate_edge(linked_edge)):
                        # Check if this is the starting edge (closed loop)
                        if linked_edge == edge:
                            # We've completed a full loop
                            break
                        
                        # Check if already walked (skip if so)
                        if linked_edge.index in walked_edges:
                            continue
                        
                        # Validate: next edge's face should be different from current edge's face
                        # AND current edge's face should share an edge with next edge's face
                        if len(current_edge.link_faces) > 0 and len(linked_edge.link_faces) > 0:
                            current_face = current_edge.link_faces[0]
                            next_face = linked_edge.link_faces[0]
                            
                            # Faces must be different
                            if current_face == next_face:
                                continue
                            
                            # Current face and next face must share at least one edge
                            current_face_edges = set(current_face.edges)
                            next_face_edges = set(next_face.edges)
                            if not (current_face_edges & next_face_edges):  # No shared edges
                                continue
                            
                            next_edge = linked_edge
                            break
                
                # Check if we've looped back to the start (closed loop)
                if next_edge is None:
                    # Check if we can complete the loop by connecting back to start
                    for linked_edge in current_vertex.link_edges:
                        if linked_edge == edge and linked_edge != current_edge:
                            # We've completed a closed loop
                            break
                    # Otherwise, we've reached the end of an open boundary loop
                    break
                    
                edge_loop_cand.add_edge(next_edge, mw)
                walked_edges.add(next_edge.index)
                prev_edge = current_edge
                current_edge = next_edge
            
            # Only add if we found at least one edge (which we did, since we started with one)
            if len(edge_loop_cand.edges) > 0:
                cls.potential_edge_loops.append(edge_loop_cand)

        if PICK_METHOD == 'SCENE_RAYCAST':
            edge_to_edge_loop = {}
            for edge_loop_cand in cls.potential_edge_loops:
                for edge in edge_loop_cand.edges:
                    edge_to_edge_loop[edge] = edge_loop_cand
            cls.edge_to_edge_loop = edge_to_edge_loop

        # Prepare and cache draw batches for the different edge loops!
        cls.update_edge_loops_2d_coords(context)
        tot_edge_loop_candidates = len(cls.potential_edge_loops)
        if tot_edge_loop_candidates > 0:
            hue_step = 1.0 / tot_edge_loop_candidates
            for i, edge_loop_cand in enumerate(cls.potential_edge_loops):
                hue = i * hue_step
                r, g, b = hsv_to_rgb(hue, 0.8, 0.8)
                edge_loop_cand.color = Color((r, g, b))
                edge_loop_cand.setup_draw_batch()

        bm.free()
        del bm
        
    @classmethod
    def update_edge_loops_2d_coords(cls, context: bpy.types.Context):
        def _project_to_2d(co: Vector) -> Vector:
            return location_3d_to_region_2d(context.region, context.space_data.region_3d, co, default=Vector((-10000, -10000)))

        for edge_loop_cand in cls.potential_edge_loops:
            for edge in edge_loop_cand.edges:
                edge.verts_co_2d = (
                    _project_to_2d(edge.verts_co[0]),
                    _project_to_2d(edge.verts_co[1])
                )

    @classmethod
    def modal_draw_post_view(cls, context: bpy.types.Context):
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)
        for edge_loop_cand in cls.potential_edge_loops:
            edge_loop_cand.draw()
        gpu.state.blend_set('NONE')
        gpu.state.depth_mask_set(False)

    # def draw_settings(context, layout, tool):
    #     settings = context.scene.bridge_plus_settings


def on_tool_switch_post(context: bpy.types.Context, space_type: str, item_idname: str | None, *, as_fallback=False):
    if item_idname != BridgePlusTool.bl_idname:
        BridgePlusTool.disable(context)
        return

    BridgePlusTool.enable(context)
