import os, re, time
import json
import asyncio
import nest_asyncio

from gai.lib import constants

def get_rc():
    """
    eg. ~/.gairc
    """
    if (not os.path.exists(os.path.expanduser(constants.GAIRC))):
        raise Exception(f"Config file {constants.GAIRC} not found. Please run 'gai init' to initialize the configuration.")
    with open(os.path.expanduser(constants.GAIRC), 'r') as f:
        return json.load(f)


def get_app_path():
    """
    eg. "app_dir" from ~/.gairc
    """
    rc = get_rc()
    app_dir=os.path.abspath(os.path.expanduser(rc["app_dir"]))
    return app_dir

def get_here() -> str:
    """
    Returns the absolute path of the caller script's directory.
    If running in a notebook, returns the current working directory.
    """
    import inspect
    try:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        if module is None or not hasattr(module, "__file__"):
            raise RuntimeError

        caller_file = module.__file__
        caller_dir = os.path.dirname(os.path.abspath(caller_file))
        return caller_dir

    except RuntimeError:
        # ⚠️ Probably running in notebook or interactive shell
        return os.getcwd()
    
def run_async_function(coro_func, *args, **kwargs):
    """
    Runs an async function either directly or via executor if the event loop is already running.
    
    Parameters:
        coro_func (coroutine function): The async function to run.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the coroutine.
    """
    loop = None
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            if nest_asyncio and not globals().get("_nest_asyncio_applied", False):
                nest_asyncio.apply(loop)
                globals()["_nest_asyncio_applied"] = True
            return loop.run_until_complete(coro_func(*args, **kwargs))
        else:
            return loop.run_until_complete(coro_func(*args, **kwargs))
    except RuntimeError:
        # In case there's no running loop (common in some environments)
        return asyncio.run(coro_func(*args, **kwargs))