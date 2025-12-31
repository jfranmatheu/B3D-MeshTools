from typing import Optional

from .icons import generate_icons_py


__all__ = ['AutoCode']


class AutoCode:
    """ Auto-code utilities for the addon. """
    @staticmethod
    def ICONS(filename: str = 'icons', icons_path: Optional[str] = None):
        """ Generate a {filename}.py file with an Icon class to get icons to draw in Blender interface or custom interfaces.

        Args:
            filename (str, optional): The name of the icons.py file. Defaults to 'icons' (please, exclude the .py extension).
            icons_path (Optional[str], optional): The path to the icons folder. Defaults to None (which fallback to 'assets/icons').
        """
        generate_icons_py(filename, icons_path)
