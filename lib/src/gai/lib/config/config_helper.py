import os
import copy
import yaml
from typing import Literal, Optional, Union, overload
from ..utils import get_app_path
from .gai_config import GaiConfig
from .gai_generator_config import GaiGeneratorConfig, MissingGeneratorConfigError
from .gai_client_config import GaiClientConfig
from .gai_tool_config import GaiToolConfig, MissingToolConfigError
from .download_config import DownloadConfig, HuggingfaceDownloadConfig, CivitaiDownloadConfig


# ──────────────── REBUILD MODEL ────────────────
# resolve forward refs & discriminators:
HuggingfaceDownloadConfig.model_rebuild()
CivitaiDownloadConfig.model_rebuild()
GaiGeneratorConfig.model_rebuild()
GaiConfig.model_rebuild()

## GaiConfig Helper Functions ---------------------------------------------------------------------------------------------

def _resolve_references(raw_config: dict) -> dict:

    """
    This function performs post-processing on the raw config after loading it from file.
    For example:
    - it resolves alias references "ref" to the actual config.
    - It also renames the "class_" field to "class" in the module config.
    """
    
    if not isinstance(raw_config,dict):
        raise ValueError("config_helper: raw_config must be a dictionary")
    
    resolved_config = raw_config.copy()

    for config_type in ["clients", "generators"]:
        if config_type not in raw_config:
            continue

        if raw_config.get(config_type,None):
            for k, v in raw_config[config_type].items():
                config = copy.deepcopy(v)
                if v.get("ref"):
                    
                    # Resolve alias references
                    
                    ref = v["ref"]
                    
                    # Make a copy of the referenced config to avoid mutating the original
                    
                    config = copy.deepcopy(raw_config[config_type][ref])

                # By now, config is either a copy of the referenced config or copy of the referencing config

                if config.get("module") and config["module"].get("class_"):
                    config["module"]["class"] = config["module"].pop("class_")

                # Save to assign the config to the resolved_config

                resolved_config[config_type][k] = config

    return resolved_config    
    
@overload
def get_gai_config(config: dict) -> GaiConfig:
    ...

@overload
def get_gai_config(file_path: str) -> GaiConfig:
    ...

def get_gai_config(config_or_path: Optional[Union[dict,str]] = None) -> GaiConfig:
    """
    Load a GaiConfig object from either a dictionary or a file path.
    
    Args:
        config_or_path: Either a dictionary containing config values or a path to a YAML file.
                        If None, loads from the default config file path.
    
    Returns:
        GaiConfig: A configuration object.
    """

    # If it's a dictionary, use it directly
    if isinstance(config_or_path, dict):
        config = _resolve_references(config_or_path)
        return GaiConfig(**config_or_path)    
    
    # If it's not a dictionary, it must be a file path
    file_path = config_or_path if isinstance(config_or_path, str) else None
    
    # If file_path is None, use the default gai config path
    if not file_path:
        app_dir = get_app_path()
        file_path = os.path.join(app_dir, 'gai.yml')
    
    try:
        # Load config from file_path
        
        with open(file_path, 'r') as f:
            raw_config = yaml.load(f, Loader=yaml.FullLoader)

        # raw_config is a config that can contain references to other config in the gai config.
        # resolved_config resolves the references to the actual config and replaces them in the config.
        
        config = _resolve_references(raw_config)
        
        if not config.get("generators",None):
            config["generators"] = {}
            
        # Convert class_ to class before converting to GaiConfig
        
        for k,v in config["generators"].items():
            if v.get("module",None):
                if v["module"].get("class_",None):
                    v["module"]["class"] = v["module"].pop("class_")
            
        return GaiConfig(**config)
    
    except Exception as e:
        raise ValueError(f"config_helper: Error loading config from file: {e}")

## GaiClientConfig Helper Functions ---------------------------------------------------------------------------------------------

@overload
def get_client_config(client_config: dict) -> GaiClientConfig:
    ...
    
@overload
def get_client_config(name: str, file_path: Optional[str] = None) -> GaiClientConfig:
    ...
@overload
def get_client_config(config: GaiClientConfig) -> GaiClientConfig:
    ...
    
def get_client_config(
    config: Union[GaiClientConfig, dict, str, None] = None,
    file_path: Optional[str] = None    
    ) -> GaiClientConfig:
    
    if isinstance(config, GaiClientConfig):
        return config

    if isinstance(config, dict):
        return GaiClientConfig(**config)
    
    if isinstance(config, str):
        name = config
        try:
            gai_config = get_gai_config(file_path)
        except Exception as e:
            raise ValueError(f"config_helper: Error loading client config from file: {e}")
        
        client_config = gai_config.clients.get(name)
        if not client_config:
            raise ValueError(f"config_helper: Client config not found. name={name}")
        
        return client_config

    raise ValueError("config_helper: Invalid arguments. Provide a GaiClientConfig, a dict, or a name (str).")

## GaiGeneratorConfig Helper Functions ---------------------------------------------------------------------------------------------

@overload
def get_generator_config(name_or_config: Union[str,dict]) -> Optional[GaiGeneratorConfig]:
    ...

@overload
def get_generator_config(generator_config: dict) -> Optional[GaiGeneratorConfig]:
    ...

@overload
def get_generator_config(name: str, file_path: Optional[str] = None) -> Optional[GaiGeneratorConfig]:
    ...

def get_generator_config(
        name_or_config: Optional[Union[str,dict]] = None,
        name: Optional[str] = None,
        generator_config: Optional[dict] = None,
        file_path: Optional[str] = None    
    ) -> Optional[GaiGeneratorConfig]:
    """
    Returns a GaiGeneratorConfig (or subclass) by either:
      - Passing a raw dict (generator_config) to parse_obj
      - Specifying a name and optional file_path to load + parse from disk
    If a name is provided and the config is not found, it will return None.
    This is to allow the global config to be updated with new the generator in subsequent calls.
    """

    # — dispatch `name_or_config` into either name or generator_config —
    if name_or_config is not None:
        if name is not None:
            raise ValueError("config_helper: name_or_config and name cannot be used together")
        if generator_config is not None:
            raise ValueError("config_helper: name_or_config and generator_config cannot be used together")
        
        if isinstance(name_or_config, dict):
            generator_config = name_or_config
        elif isinstance(name_or_config, str):
            name = name_or_config
        else:
            raise ValueError(
                "config_helper: name_or_config must be either a dict or a str"
            )

    # 1) If caller passed a dict, ignore name/file_path entirely
    if generator_config is not None:
        if name is not None or file_path is not None:
            raise ValueError("config_helper: When providing generator_config dict, do not also pass name or file_path")
    
    # 2) Otherwise, caller must supply a name
    elif name is None:
        raise ValueError("config_helper: Invalid arguments. Either 'name' or 'config' must be provided.")

    if name:

        # If name is provided, load the tool config from gai.yml
        # If not, return None to update the global config
        
        try:
            gai_config = get_gai_config(file_path)
            generator_config = gai_config.generators.get(name, None)
            if not generator_config:
                return None
        except Exception as e:
            raise ValueError(f"config_helper: Error loading generator config from file: {e}")
    else:
        generator_config = GaiGeneratorConfig(**generator_config) 

    # Final Processing
    
            
    if generator_config.module:
        
        # Sometimes "class" maybe stored as class_ after exporting because class is a reserved word in python
        # So we need to convert class_ to class before converting to GaiGeneratorConfig
        
        if not generator_config.module.class_:
            raise ValueError(f"config_helper: module.class_ is required for generator config {name}")
        
    return generator_config

def list_generator_configs(file_path: Optional[str]=None) -> dict[str, GaiGeneratorConfig]:
    """
    List all generator configs in either the local or global gai.yml file.
    """
    try:
        gai_config = get_gai_config(file_path)
    except Exception as e:
        raise ValueError(f"config_helper: Error loading generator config from file: {e}")

    if not gai_config.generators:
        return {}
    
    return copy.deepcopy(gai_config.generators)

def get_download_config(name_or_config: Union[str,dict],file_path:Optional[str]=None) -> DownloadConfig:

    generator_config = get_generator_config(name_or_config=name_or_config, file_path=file_path)
    if not generator_config.source:
        raise Exception(f"config_helper: Generator '{name_or_config}' does not have a source defined. Make sure you are using a 'gai' generator.")
    return generator_config.source

## GaiToolConfig Helper Functions ---------------------------------------------------------------------------------------------

@overload
def get_tool_config(tool_config: dict) -> GaiToolConfig:
    ...
    
@overload
def get_tool_config(name: str, file_path: Optional[str] = None) -> GaiToolConfig:
    ...
    
def get_tool_config(
    name: Optional[str] = None,
    tool_config: Optional[dict] = None,
    file_path: Optional[str] = None
    ) -> Optional[GaiToolConfig]:
    """
    This method is used to load a single tool entry from gai.yml or from a config dictionary.
    """

    # 1) If caller passed a dict, ignore name/file_path entirely
    if tool_config is not None:
        if name is not None or file_path is not None:
            raise ValueError("config_helper: When providing tool_config dict, do not also pass name or file_path")

    # 2) Otherwise, caller must supply a name
    elif name is None:
        raise ValueError("config_helper: Invalid arguments. Either 'name' or 'config' must be provided.")


    if name:
        
        # If name is provided, load the tool config from gai.yml
        # If not, return None to update the global config
        
        try:
            gai_config = get_gai_config(file_path)
            tool_config = gai_config.tools.get(name, None)
            if not tool_config:
                return None
        except Exception as e:
            raise ValueError(f"config_helper: Error loading generator config from file: {e}")
        
    return tool_config

def list_tool_configs(file_path: Optional[str]=None) -> dict[str, GaiToolConfig]:
    """
    List all tool configs in either the local or global gai.yml file.
    """
    try:
        gai_config = get_gai_config(file_path)
    except Exception as e:
        raise ValueError(f"config_helper: Error loading tool config from file: {e}")

    if not gai_config.tools:
        return {}
    
    return copy.deepcopy(gai_config.tools)

## General Helper Functions ---------------------------------------------------------------------------------------------

def update_gai_config(
    updateable_config_type:Literal["generators","tools"],
    builtin_config_path:str, 
    global_config_path:Optional[str]=None
    ) -> GaiConfig:
    """
    This method is used to graft the current generator config into gai config["generators"]
    builtin-config refers to the config that is shipped with the function-level library and obtained via "get_builtin_config_path()"
    """
    import copy
    if not global_config_path:
        app_path = get_app_path()
        global_config_path = os.path.join(app_path, 'gai.yml')        
    
    global_gai_config = get_gai_config(global_config_path)

    local_gai_config = get_gai_config(builtin_config_path)
    
    # Copy the builtin config to the global gai config if it doesn't exist
    
    if updateable_config_type == "tools":
        for k,v in local_gai_config.tools.items():
            v=copy.deepcopy(v)
            if not global_gai_config.tools.get(k,None):
                global_gai_config.tools[k] = v
    elif updateable_config_type == "generators":
        for k,v in local_gai_config.generators.items():
            v=copy.deepcopy(v)
            if not global_gai_config.generators.get(k,None):
                global_gai_config.generators[k] = v
    else:
        raise ValueError(f"config_helper: Unknown updateable_config_type {updateable_config_type}. Must be either 'tools' or 'generators'")
            
    # Save the global gai config to the file
    with open(global_config_path, "w") as f:
        y = global_gai_config.to_yaml()
        f.write(y)
    return global_gai_config

def get_and_update_tool_config(tool_name,builtin_config_path) -> GaiToolConfig:

    """
    Get the tool config from the global gai.yml file and if not found, update the global config with the local config.
    """
    
    tool_config = None
    try:
        tool_config = get_tool_config(tool_name)
    except MissingToolConfigError:
        pass
    except Exception as e:
        print(f"Failed to load tool config: {str(e)}")
        raise e
    
    if not tool_config:
        
        # This means "tools" section is not present in global gai.yml
        # Update global config with local config
        
        update_gai_config(updateable_config_type="tools",builtin_config_path=builtin_config_path)
        
        try:
            # Try to load the tool config again after updating global config
            tool_config = get_tool_config(tool_name)
        except MissingToolConfigError:
            print("Failed to load tool config after updating global config.")
            raise MissingToolConfigError("Failed to load tool config after updating global config.")
        except Exception as e:
            print(f"Failed to load tool config after updating global config: {str(e)}")
            raise e
        
    return tool_config

def get_and_update_generator_config(generator_name,builtin_config_path) -> GaiGeneratorConfig:

    """
    Get the generator config from the global gai.yml file and if not found, update the global config with the local config.
    """
    
    generator_config = None
    try:
        generator_config = get_generator_config(generator_name)
    except MissingToolConfigError:
        pass
    except Exception as e:
        print(f"Failed to load generator config: {str(e)}")
        raise e
    
    if not generator_config:
        
        # This means "tools" section is not present in global gai.yml
        # Update global config with local config
        
        update_gai_config(updateable_config_type="generators",builtin_config_path=builtin_config_path)
        
        try:
            # Try to load the generator config again after updating global config
            generator_config = get_generator_config(generator_name)            
        except MissingGeneratorConfigError:
            print("Failed to load generator config after updating global config.")
            raise MissingGeneratorConfigError("Failed to load generator config after updating global config.")
        except Exception as e:
            print(f"Failed to load generator config after updating global config: {str(e)}")
            raise e
        
    return generator_config