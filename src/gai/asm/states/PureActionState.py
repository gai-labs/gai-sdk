from ..base import StateBase
from gai.lib.logging import getLogger
logger = getLogger(__name__)

"""
PredicateState

The output of this state is a boolean result: "true" or "false"

"""
class PureActionState(StateBase):
    """
    state schema:
    {
        "SOME_STATE": {
            "module_path": "gai.agents.async_states.PureActionState",
            "class_name": "PureActionState",
            "input_data": {
                "title": "SOME_STATE"
            },
            "action": {
                "type": "action",
                "action": "callable_name"
            },
            "output_data": ["action_result"]
        },
    }    
    """    

    def __init__(self,machine):
        super().__init__(machine)
        
        # Define output data
        self.action_result=None

    async def run_async(self):
        
        if hasattr(self,"action") and self.action:
            self.action_result = await self.action(self)
            
        
        

