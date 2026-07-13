#!/usr/bin/env python3

"""
Regression test for PurePredicateState dispatching sync and async predicates.

run_async() used to branch on `callable(self.predicate)` before
`asyncio.iscoroutinefunction(self.predicate)`. A coroutine function is also
callable, so the first branch always won and an `async def` predicate was called
but never awaited: `predicate_result` held a raw coroutine, which is neither
== True nor == False, so neither condition_true nor condition_false fired and the
machine could not transition.

Both PureActionState and PurePredicateState now dispatch through
gai.asm.base.call_handler(), which awaits the result only if it is awaitable.
"""

import inspect

import pytest

from gai.asm import AgenticStateMachine

STATE_DIAGRAM = """
INIT --> CHECK
CHECK --> TRUE_BRANCH: condition_true
CHECK --> FALSE_BRANCH: condition_false
TRUE_BRANCH --> FINAL
FALSE_BRANCH --> FINAL
"""

BRANCH = {
    "module_path": "gai.asm.states",
    "class_name": "PureActionState",
    "output_data": ["action_result"],
}


def build_fsm(predicate):
    with AgenticStateMachine.StateMachineBuilder(STATE_DIAGRAM) as builder:
        fsm = builder.build(
            {
                "INIT": {},
                "CHECK": {
                    "module_path": "gai.asm.states",
                    "class_name": "PurePredicateState",
                    "title": "CHECK",
                    "predicate": "test_predicate",
                    "output_data": ["predicate_result"],
                },
                "TRUE_BRANCH": {**BRANCH, "title": "TRUE_BRANCH"},
                "FALSE_BRANCH": {**BRANCH, "title": "FALSE_BRANCH"},
                "FINAL": {
                    "output_data": ["predicate_result"],
                },
            },
            test_predicate=predicate,
        )
    fsm.restart()
    return fsm


@pytest.mark.asyncio
async def test_sync_predicate_true_takes_the_true_branch():
    def test_predicate(state):
        return True

    fsm = build_fsm(test_predicate)
    await fsm.run_async()  # INIT -> CHECK
    await fsm.run_async()  # CHECK -> ?

    assert fsm.state_bag["predicate_result"] is True
    assert fsm.state == "TRUE_BRANCH"


@pytest.mark.asyncio
async def test_sync_predicate_false_takes_the_false_branch():
    def test_predicate(state):
        return False

    fsm = build_fsm(test_predicate)
    await fsm.run_async()  # INIT -> CHECK
    await fsm.run_async()  # CHECK -> ?

    assert fsm.state_bag["predicate_result"] is False
    assert fsm.state == "FALSE_BRANCH"


@pytest.mark.asyncio
async def test_async_predicate_is_awaited_not_left_as_a_coroutine():
    """The bug: the coroutine was stored raw, so no transition condition matched."""

    async def test_predicate(state):
        return True

    fsm = build_fsm(test_predicate)
    await fsm.run_async()  # INIT -> CHECK
    await fsm.run_async()  # CHECK -> ?

    result = fsm.state_bag["predicate_result"]
    assert not inspect.isawaitable(result), f"coroutine was never awaited: {result!r}"
    assert result is True
    assert fsm.state == "TRUE_BRANCH"


@pytest.mark.asyncio
async def test_async_predicate_false_takes_the_false_branch():
    async def test_predicate(state):
        return False

    fsm = build_fsm(test_predicate)
    await fsm.run_async()  # INIT -> CHECK
    await fsm.run_async()  # CHECK -> ?

    assert fsm.state_bag["predicate_result"] is False
    assert fsm.state == "FALSE_BRANCH"


@pytest.mark.asyncio
async def test_non_callable_predicate_raises():
    fsm = build_fsm("not a function")

    # CHECK's predicate runs on entering the state, i.e. during this first trigger.
    with pytest.raises(ValueError, match="not callable"):
        await fsm.run_async()  # INIT -> CHECK
