from functools import wraps
from typing import Callable, List
import sys


_restore_wrap_override_func: List[Callable] = []


def wrap_function(original_func: Callable, pre_func: Callable | None = None, post_func: Callable | None = None) -> Callable:
    """ Wrap a function with pre and post functions. """
    module_original = sys.modules.get(original_func.__module__, None)
    assert module_original, f"Module {original_func.__module__} not found for function {original_func.__name__}"

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        if pre_func:
            pre_func(*args, **kwargs)
        res = original_func(*args, **kwargs)
        if res == False:
            return False
        if post_func:
            post_func(*args, **kwargs)
        return res

    def restore():
        # Restore original function.
        setattr(module_original, original_func.__name__, original_func)

    # Backup the original function.
    _restore_wrap_override_func.append(restore)
    
    # Override original function.
    setattr(module_original, original_func.__name__, wrapper)

    return restore


def unregister():
    """ Unregister the wrapped functions. """
    # If we are overriding any Blender internal function
    # we should always restore them to their original state.
    for restore_func in _restore_wrap_override_func:
        restore_func()
