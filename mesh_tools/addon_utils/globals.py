import bpy
import sys
from pathlib import Path
import ctypes
import platform

from .. import __package__ as __main_package__#, bl_info



def is_junction(path: Path) -> bool:
    # Check if the path exists
    if not path.exists():
        return False

    if platform.system() != 'Windows':
        return False

    # Use GetFileAttributes to check if it's a reparse point and a directory
    file_attributes = ctypes.windll.kernel32.GetFileAttributesW(str(path))

    if file_attributes == -1:
        return False

    is_reparse_point = (file_attributes & 0x400) != 0
    is_directory = (file_attributes & 0x10) != 0

    return is_reparse_point and is_directory



class GLOBALS:
    """ Addon globals. """
    PYTHON_PATH = sys.executable

    BLENDER_VERSION = bpy.app.version
    IN_BACKGROUND = bpy.app.background
    USE_DEBUG_FLAG = bpy.app.debug_value == 1

    ADDON_SOURCE_PATH = Path(__file__).parent.parent
    ADDON_MODULE = __main_package__
    ADDON_MODULE_LOWER = ADDON_MODULE.split('.')[-1].replace('_', '').lower()
    ADDON_MODULE_UPPER = ADDON_MODULE_LOWER.upper()

    ADDON_MODULE_NAME = ADDON_MODULE_LOWER.replace('_', ' ').title().replace(' ', '')
    #ADDON_NAME = bl_info['name']
    #ADDON_VERSION = bl_info['version']
    #SUPPORTED_BLENDER_VERSION = bl_info['blender']
    ICONS_PATH = ADDON_SOURCE_PATH / 'assets' / 'icons'

    IS_JUNCTION = is_junction(ADDON_SOURCE_PATH)  # windows only

    IN_DEVELOPMENT = USE_DEBUG_FLAG or IS_JUNCTION or 'vscode_development' in __main_package__ # (hasattr(sys, 'gettrace') and sys.gettrace() is not None)

    @staticmethod
    def get_addon_global_value(key: str, default_value = None):
        """ Get a custom value from the addon globals.
        
        Args:
            key (str): The key of the value to get.
            default_value (any): The default value to return if the key is not found.
        """
        return getattr(bpy, GLOBALS.ADDON_MODULE).get(key, default_value)

    @staticmethod
    def set_addon_global_value(key: str, value) -> None:
        """ Set a custom value in the addon globals.
        
        Args:
            key (str): The key of the value to set.
            value (any): The value to set.
        """
        getattr(bpy, GLOBALS.ADDON_MODULE)[key] = value


setattr(bpy, GLOBALS.ADDON_MODULE, dict())
