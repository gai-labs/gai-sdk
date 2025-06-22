import importlib
import asyncio
from transitions.extensions.asyncio import AsyncMachine
from typing import Optional, Union
from gai.lib.logging import getLogger
from gai.asm.monologue import Monologue
from datetime import datetime

logger = getLogger(__name__)


class AsyncStateMachine:
    class StateModel:
        def __init__(
            self,
            state_manifest: dict,
            monologue: Optional[Monologue] = None,
            agent_name: str = "Assistant",
            **kwargs,
        ):
            self.state = None
            self.agent_name = agent_name
            self.state_manifest = state_manifest
            self.state_history = []
            self.state_data = {}
            self.state_bag = {}
            self.user_message = None

            if not monologue:
                monologue = Monologue(agent_name=agent_name)
            self.monologue = monologue
            self.step = 0
            self.kwargs = kwargs

        async def resolve_input(self, state):
            input_data = state.manifest.get("input_data", {})
            resolved_input_data = {}
            last_output = (
                state.machine.state_history[-1].get("output", None)
                if len(state.machine.state_history) > 0
                else None
            )

            # PASS 1: Resolve all literal values and built-in values first

            for k, v in input_data.items():
                if k == "monologue":
                    raise ValueError(
                        "input_data cannot contain reserved key `monologue`"
                    )
                if k == "user_message":
                    raise ValueError(
                        "input_data cannot contain reserved key `user_message`"
                    )

                if k == "step":
                    raise ValueError("input_data cannot contain reserved key `step`")

                if k == "time":
                    raise ValueError("input_data cannot contain reserved key `time`")

                if k == "name":
                    raise ValueError("input_data cannot contain reserved key `name`")

                if not isinstance(v, dict):
                    # This is a literal value, resolve it immediately
                    resolved_input_data[k] = v
                else:
                    # If this is a dict then resolve only if it is not a reserved type
                    if "type" in v:
                        if v["type"] == "getter":
                            continue
                        if v["type"] == "prev_state":
                            continue
                        if v["type"] == "state_bag":
                            continue
                        else:
                            resolved_input_data[k] = v

            resolved_input_data["user_message"] = self.user_message
            resolved_input_data["monologue"] = self.monologue.copy()
            resolved_input_data["step"] = self.step
            resolved_input_data["time"] = datetime.now()
            resolved_input_data["name"] = self.agent_name

            # Merge Pass 1 results into a working copy for Pass 2

            import copy

            working_input_data = copy.deepcopy(input_data)
            working_input_data.update(resolved_input_data)

            # PASS 2: Resolve dependencies and complex logic
            for k, v in working_input_data.items():
                if k in resolved_input_data:
                    # Already resolved in pass 1, skip
                    continue

                resolved = None

                if isinstance(v, dict):
                    # If item is a dependency, resolve it.

                    if v.get("type", None) == "getter":
                        # dependency refers to the name of a getter function

                        dependency = v.get("dependency", None)
                        if dependency is None:
                            raise ValueError(
                                f"'dependency' property not found for 'getter'. state: {state.machine.state} input: {k} value: {v}"
                            )

                        callable_ = state.machine.kwargs.get(dependency, None)
                        if callable_ is None:
                            raise ValueError(
                                f"Dependency {dependency} is specified in manifest but not passed into ASM builder."
                            )

                        if asyncio.iscoroutinefunction(callable_):
                            resolved = await callable_(state)
                        else:
                            resolved = callable_(state)

                    elif v.get("type", None) == "prev_state":
                        # dependency refers to the name of a previous state output

                        if last_output:
                            dependency = v.get("dependency", None)
                            if dependency:
                                if last_output.get(dependency, None) is None:
                                    raise ValueError(
                                        f"Dependency {dependency} not found in last output: {last_output}"
                                    )
                                resolved = last_output[dependency]
                            else:
                                raise ValueError(
                                    "'dependency' property not specified for 'prev_state'."
                                )
                        else:
                            raise ValueError(
                                "Last output not found in state history. Is this 'INIT' state?"
                            )

                    elif v.get("type", None) == "state_bag":
                        # dependency refers to the name of a state bag item

                        dependency = v.get("dependency", None)
                        resolved = state.machine.state_bag.get(dependency, None)
                        if resolved is None:
                            raise ValueError(
                                f"Dependency {dependency} not found in state bag: {state.machine.state_bag}"
                            )

                    if not resolved:
                        raise ValueError(
                            f"There are unresolved items from the manifest {v}. Please check the input_data section again."
                        )

                resolved_input_data[k] = resolved

            # Make sure we are not modifying the original
            resolved_input_data = copy.deepcopy(resolved_input_data)

            # Update the state bag with latest snapshot of input data.
            for k, v in resolved_input_data.items():
                state.machine.state_bag[k] = v

            return resolved_input_data

        def finalize_output(self, state):
            """
            Since state_bag data is a snapshot of the current state,
            this method finalizes the state_bag data item to be saved in history.
            The item to be saved is determined by the output_data from the manifest.
            NOTE: Data must exist in state_bag before this method is called.
            """
            import copy

            output = {}
            if "output_data" in state.manifest:
                output = {}
                for k in state.manifest["output_data"]:
                    if k == "name":
                        raise ValueError(
                            "output_data cannot contain reserved key `name`"
                        )
                    if k == "step":
                        raise ValueError(
                            "output_data cannot contain reserved key `step`"
                        )
                    if k == "monologue":
                        raise ValueError(
                            "output_data cannot contain reserved key `monologue`"
                        )
                    if k == "user_message":
                        raise ValueError(
                            "output_data cannot contain reserved key `user_message`"
                        )
                    if k == "time":
                        raise ValueError(
                            "output_data cannot contain reserved key `time`"
                        )

                    if k in self.state_bag:
                        try:
                            # deepcopy-able items will be copied.
                            output[k] = copy.deepcopy(self.state_bag.get(k))
                        except Exception as e:
                            # otherwise, keep a reference to the item.
                            output[k] = self.state_bag.get(k)
                            pass

            # Built-In State: Name
            output["name"] = self.state_bag["name"]

            # Built-In State: User Message
            output["user_message"] = self.state_bag["user_message"]

            # Built-In State: Monologues Messages
            # Keep a snapshot of the original monologue after action
            output["monologue"] = self.monologue.copy()

            # Built-In State: Step
            self.step += 1
            output["step"] = self.step

            # Built-In State: timestamp
            output["time"] = datetime.now()

            return output

        def resolve_action(self, state):
            """
            The action will override the default action of the state.
            """
            action_name = state.manifest.get("action", None)
            if action_name:
                action = state.machine.kwargs.get(action_name, None)
                if action is None:
                    raise ValueError(
                        f"Action handler `{action_name}` not found in state machine kwargs."
                    )
                return action
            return None

        def resolve_predicate(self, state):
            """
            The predicate will resolve to "True" or "False"
            """
            predicate_name = state.manifest.get("predicate", None)
            if predicate_name:
                return state.machine.kwargs.get(predicate_name, None)
            return None

        async def before_action_async(self):
            # This function is only used for configuring the "INIT" state

            logger.info(f"Leaving state: {self.state}")

            if self.state == "INIT":
                """Optional: Loads initializer by reflecting on the manifest."""

                from gai.asm.states import InitializeState

                state = InitializeState(self)
                state.manifest = self.state_manifest.get("INIT", {})
                # 'INIT' can be omitted from manifest, then create an empty one.

                state.input = await self.resolve_input(state)
                state.output = self.finalize_output(state)

                # Update history
                self.state_history.append(
                    {"state": "INIT", "input": state.input, "output": state.output}
                )

        async def action_async(self):
            # This function is only used for configuring any states other than "INIT" state

            logger.info(f"Entering state: {self.state}")

            # Get id of current state
            state_id = self.state
            state_id = state_id.split("(")[0]

            if state_id == "FINAL":
                # If the state is FINAL, we do not run any action.
                # Just log the state and input/output.

                state_input = self.state_history[-1]["output"].copy()

                state_manifest = self.state_manifest.get("FINAL", {})
                if not state_manifest:
                    output_data = self.state_bag.keys()
                else:
                    output_data = state_manifest.get("output_data", [])
                state_output = {}
                for key in output_data:
                    if key in self.state_bag:
                        state_output[key] = self.state_bag[key]
                self.state_history.append(
                    {"state": state_id, "input": state_input, "output": state_output}
                )
                return self.state

            if state_id == "INIT":
                return self.state

            """Loads a state class by reflecting on the manifest."""
            if state_id in self.state_manifest:
                if state_id == "INIT":
                    module = importlib.import_module("gai.asm")
                    StateClass = getattr(module, "InitializeState")
                else:
                    module = importlib.import_module(
                        self.state_manifest[state_id]["module_path"]
                    )
                    StateClass = getattr(
                        module, self.state_manifest[state_id]["class_name"]
                    )

                """Create a state instance with input data."""
                state = StateClass(self)

                # Merge Previous State Ouput with New State Input
                state.input = self.state_history[-1]["output"].copy()
                new_input = await self.resolve_input(state)
                state.input = {**state.input, **new_input}

                # Run state
                state.action = self.resolve_action(state)
                state.predicate = self.resolve_predicate(state)
                await state.run_async()
                state.output = self.finalize_output(state)

                self.state_history.append(
                    {"state": state_id, "input": state.input, "output": state.output}
                )
            else:
                raise ValueError(f"State {state_id} not found in state manifest.")

            return self.state

        def restart(self):
            self.state = "INIT"
            self.state_history = []
            self.state_bag = {}

    class StateMachineBuilder:
        def __init__(self, state_diagram: str):
            self.state_diagram = state_diagram

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def build(
            self,
            fsm_model: Optional[Union[dict, "AsyncStateMachine.StateModel"]] = None,
            monologue: Optional[Monologue] = None,
            agent_name: str = "Assistant",
            **kwargs,
        ) -> "AsyncStateMachine.StateModel":
            if isinstance(fsm_model, dict):
                fsm_model = AsyncStateMachine.StateModel(
                    state_manifest=fsm_model, agent_name=agent_name, monologue=monologue
                )

            if not fsm_model:
                fsm_model = AsyncStateMachine.StateModel(
                    state_manifest={}, agent_name=agent_name, monologue=monologue
                )

            fsm_model.kwargs = {**fsm_model.kwargs, **kwargs}

            states = set()
            transitions = []
            for line in self.state_diagram.strip().split("\n"):
                if not line.strip():
                    continue

                # source --> dest: condition

                parts = line.split("-->")
                source = parts[0].strip()
                dest_condition = parts[1].split(":")
                dest = dest_condition[0].strip()
                condition = (
                    dest_condition[1].strip() if len(dest_condition) > 1 else None
                )

                states.add(source)
                states.add(dest)

                transition_dict = {
                    # run_async() is used to trigger the transition
                    "trigger": "run_async",
                    "source": source,
                    "dest": dest,
                }

                # action_async() is used to run the action after the transition

                if not hasattr(fsm_model, "action_async"):
                    raise AttributeError("action_async() handler is missing.")
                transition_dict["after"] = "action_async"

                # before_action_async() is optional and is rarely used.

                if hasattr(fsm_model, "before_action_async"):
                    transition_dict["before"] = "before_action_async"

                # condition
                if condition:
                    # If the "source" state class exists, look for the condition function in it.

                    if (
                        hasattr(fsm_model, "state_manifest")
                        and source in fsm_model.state_manifest
                    ):
                        module_path = fsm_model.state_manifest[source]["module_path"]
                        module = importlib.import_module(module_path)
                        class_name = fsm_model.state_manifest[source]["class_name"]
                        StateClass = getattr(module, class_name)
                        condition_func = getattr(StateClass, condition, None)
                        if not condition_func:
                            raise AttributeError(
                                f"condition method {condition} is missing in state {source}"
                            )
                        transition_dict["conditions"] = [
                            lambda condition_func=condition_func: condition_func(
                                fsm_model
                            )
                        ]

                    # Otherwise, look for the condition function in the model.
                    elif hasattr(fsm_model, condition):
                        transition_dict["conditions"] = [condition]

                    # If the condition function is not found, raise an error.
                    else:
                        raise AttributeError(
                            f"condition method is required: {condition}"
                        )

                transitions.append(transition_dict)

            states = list(states)

            if "FINAL" not in states:
                raise ValueError("'FINAL' state is required in state_diagram.")
            if "INIT" not in states:
                raise ValueError("'INIT' state is required in state_diagram.")

            machine = AsyncMachine(
                model=fsm_model, states=states, initial="INIT", auto_transitions=False
            )

            for trans in transitions:
                machine.add_transition(**trans)

            return fsm_model
