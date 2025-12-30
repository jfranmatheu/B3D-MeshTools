import bpy


class BridgeToolSettings(bpy.types.PropertyGroup):
    mode: bpy.props.EnumProperty(
        name="Mode",
        description="Mode of the bridge plus",
        items=[
            ("DRAW_PATH", "Draw Path", "Click&Drag from an edge loop to draw a bridge guide to another edge loop"),
            ("CLICK_CLICK", "Click Click", "Click an edge loop, then click another edge loop to bridge"),
            ("PREVIZ", "Preview", "Click an edge loop, then hover over another to preview the bridge"),
        ],
        default="CLICK_CLICK",
    )
