class InitializeState:
    
    # Main purpose of this state is to initialize the global state bag with external input data.
    # output_data is not required for this state.
    
    """
    "INIT": {
        "module_path": "gai.agents.async_states.InitializeState",
        "class_name": "InitializeState",
        "title": "INIT",
        "input_data": {
            "name": "Sara",
            "user_question": get_user_question,
            "dialogue_messages": []
        },
    }    
    """
    
    def __init__(self,machine):
        self.machine=machine
    
    async def run_async(self):

        if not hasattr(self.machine, "state_bag"):
            raise ValueError("state_bag not found in machine")        
        

        
