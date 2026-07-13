from gai.lib.logging import getLogger
logger = getLogger(__name__)

from ..base import StateBase, call_handler

"""
PredicateState

The output of this state is a boolean result: "true" or "false"

"""
class PurePredicateState(StateBase):
    """
    state schema:
    {
        "PREDICATE": {
            "module_path": "gai.agents.async_states.PurePredicateState",
            "class_name": "PurePredicateState",
            "predicate": "sky_is_blue",  # This is the predicate to evaluate
            "output_data": ["predicate_result"]        
        }
    }    
    """    

    ## STATE-LEVEL CONDITIONS ###
    @classmethod
    def condition_true(cls,state_model):
        """Condition function for transitioning to condition_true"""
        condition = (state_model.state_history[-1]["output"]["predicate_result"] == True)
        if condition:
            logger.info(f"State={state_model.state} - Condition=True")
        return condition

    @classmethod
    def condition_false(cls,state_model):
        """Condition function for transitioning to condition_false"""
        condition = (state_model.state_history[-1]["output"]["predicate_result"] == False)
        if condition:
            logger.info(f"State={state_model.state} - Condition=False")
        return condition

    async def run_async(self):

        if not self.predicate:
            raise ValueError("Predicate function not defined.")

        self.machine.state_bag["predicate_result"] = await call_handler(self.predicate, self)
