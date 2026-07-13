#!/usr/bin/env python3

"""
Regression test for PureActionState publishing its result.

The action result used to be assigned to the state instance (`self.action_result`),
but AgenticStateMachine.finalize_output() only harvests `machine.state_bag`, so the
result was silently dropped: the FINAL output came back empty and
`fsm.state_bag.get("action_result")` returned None even after the action ran.

PureActionState.run_async() now writes to `self.machine.state_bag["action_result"]`.
"""

import pytest

from gai.asm import AgenticStateMachine

STATE_DIAGRAM = """
INIT --> PROCESS
PROCESS --> FINAL
"""


def build_fsm(action):
    with AgenticStateMachine.StateMachineBuilder(STATE_DIAGRAM) as builder:
        fsm = builder.build(
            {
                "INIT": {},
                "PROCESS": {
                    "module_path": "gai.asm.states",
                    "class_name": "PureActionState",
                    "title": "PROCESS",
                    "action": "test_action",
                    "output_data": ["action_result"],
                },
                "FINAL": {
                    "output_data": ["action_result"],
                },
            },
            test_action=action,
        )
    fsm.restart()
    return fsm


async def run_to_final(fsm):
    await fsm.run_async()  # INIT -> PROCESS
    await fsm.run_async()  # PROCESS -> FINAL
    return fsm


@pytest.mark.asyncio
async def test_action_result_is_published_to_state_bag():
    """The action's return value must land in the state bag."""

    async def test_action(state):
        return {"answer": 42}

    fsm = await run_to_final(build_fsm(test_action))

    assert fsm.state_bag["action_result"] == {"answer": 42}


@pytest.mark.asyncio
async def test_action_result_reaches_final_output():
    """FINAL projects state_bag through its output_data whitelist."""

    async def test_action(state):
        return "test_result"

    fsm = await run_to_final(build_fsm(test_action))

    final_output = fsm.state_history[-1]["output"]
    assert final_output["action_result"] == "test_result"


@pytest.mark.asyncio
async def test_action_result_is_visible_to_downstream_states():
    """The result is carried in the running state's output, not lost with the instance."""

    async def test_action(state):
        return "downstream"

    fsm = await run_to_final(build_fsm(test_action))

    process_output = fsm.state_history[-2]["output"]
    assert process_output["action_result"] == "downstream"


@pytest.mark.asyncio
async def test_action_result_defaults_to_none_without_action():
    """A state with no action still publishes the key so downstream reads are safe."""

    fsm = build_fsm(None)
    # An action of None is not resolvable from kwargs, so drop it from the manifest.
    del fsm.state_manifest["PROCESS"]["action"]

    await run_to_final(fsm)

    assert fsm.state_bag["action_result"] is None
