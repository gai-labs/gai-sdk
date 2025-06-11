import os
import shutil
def init():
    from gai.lib.tests import make_local_tmp
    
    # Get source files
    
    here = os.getcwd()
    orig_dir = os.path.abspath(os.path.join(here,"..","data","templates"))
    
    # Create local temp dir
    
    app_dir = make_local_tmp()
    dest_dir = os.path.abspath(os.path.join(app_dir,"data","templates"))
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy source to local temp dir
    shutil.copy(orig_dir + "/component_templates.original.json",dest_dir + "/component_templates.json")
    shutil.copy(orig_dir + "/agent_class_templates.original.json",dest_dir + "/agent_class_templates.json")

    return app_dir