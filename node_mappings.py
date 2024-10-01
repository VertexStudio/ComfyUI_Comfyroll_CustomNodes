try:
    from .nodes.nodes_core import *
    from .nodes.nodes_aspect_ratio import *
    from .nodes.nodes_list import *
except ImportError:
    print("\033[34mComfyroll Studio: \033[92mFailed to load Essential nodes\033[0m")

NODE_CLASS_MAPPINGS = {  
    ### List IO
    "CR Load Image List Plus - Modified": CR_LoadImageListPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    ### List IO
    "CR Load Image List Plus - Modified": "⌨️ CR Load Image List Plus - Modified", 
    
}
