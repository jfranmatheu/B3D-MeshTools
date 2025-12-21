import bpy


class DragFaceSettings(bpy.types.PropertyGroup):
    radius: bpy.props.FloatProperty(
        name="Influence Radius",
        description="Radius of influence for the drag effect",
        default=5.0,
        min=0.1,
        max=100.0
    )
    falloff_power: bpy.props.FloatProperty(
        name="Falloff Power",
        description="Power of the falloff curve (1=Linear, 2=Quadratic)",
        default=2.0,
        min=0.1,
        max=10.0
    )
