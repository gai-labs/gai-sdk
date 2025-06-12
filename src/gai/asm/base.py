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

