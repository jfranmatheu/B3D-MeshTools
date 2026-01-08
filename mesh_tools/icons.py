# /icons.py
# Code automatically generated!
from pathlib import Path
from .addon_utils.auto_code.icons import IconsEnum, IconsViewer

icons_dirpath = Path("assets/icons")


class Icons(IconsViewer):

	class MAIN(IconsEnum):
		BRIDGE_TOOL = icons_dirpath / "bridge_tool.png"
		BRIDGE_TOOL_DRAW = icons_dirpath / "bridge_tool_draw.png"

