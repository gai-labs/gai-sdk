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

    async def run_async(self):

        # Define output data. Only the state bag is carried into the state output,
        # state attributes are discarded when the state is done.
        self.machine.state_bag["action_result"]=None

        if hasattr(self,"action") and self.action:
            self.machine.state_bag["action_result"] = await self.action(self)
            
        
        

