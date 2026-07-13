import inspect

async def call_handler(handler, state):
    """
    Invoke a state handler (an action or a predicate) that may be either sync or async.

    Note that `callable()` is also True for a coroutine function, so the call has to be
    made first and the result awaited only if it turns out to be awaitable.
    """
    if not callable(handler):
        raise ValueError(f"State handler is not callable: {handler}")

    result = handler(state)
    if inspect.isawaitable(result):
        result = await result
    return result

class StateBase:

    def __init__(self, machine):
        """
        Using the state manifest, this class dynamically creates a state model with the following features:
        - StateInput: A dynamically created Pydantic model based on the input_data from the manifest.
        - StateOutput: A dynamically created Pydantic model based on the output_data from the manifest.
        - Predicate: A dynamically resolved predicate function from the manifest.
        - Action: A dynamically resolved action function from the manifest.
        - Title: The title of the state extracted from the manifest.
        """        
        self.machine = machine
        self.state_name = machine.state.split('(')[0]
        self.manifest = machine.state_manifest[self.state_name]
        self.title = self.manifest.get("title")
        self.input = None
        self.output = None
    
    def assimilate_output(self, source_output_data:dict):
        """
        This method is only used when the state is a superstate or a composite state.
        Absorb the output from the source state into the current state.
        Used for transferring of substate output to parent state in composite state machine operations.
        """
        for k,v in source_output_data.items():
            self.input_data[k] = v
        self.finalize_output()

