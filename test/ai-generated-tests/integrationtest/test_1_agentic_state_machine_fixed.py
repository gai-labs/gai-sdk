"""
Fixed version of 1_agentic_state_machine.ipynb test
This test addresses the issue where fsm.run_async() returns a boolean,
not an async generator. The streamer should be accessed from the state bag.
"""

import pytest
from gai.asm import AgenticStateMachine


async def generate_action(state):
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    # Import data from state_bag
    agent_name = state.machine.state_bag.get("name", "Assistant")
    user_message = state.machine.state_bag.get(
        "user_message",
        "If you are seeing this, that means I have forgotten to add a user message. Remind me.",
    )

    # Execute
    state.machine.monologue.add_user_message(
        content=f"Your name is {agent_name}. You are a helpful assistant.{agent_name}, {user_message}"
    )

    from gai.messages import message_helper

    chat_messages = state.machine.monologue.list_chat_messages()
    chat_messages = message_helper.shrink_messages(chat_messages)

    async def streamer():
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use a valid model name
            messages=chat_messages,
            max_tokens=50,
            stream=True,
        )
        content = ""
        async for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if isinstance(chunk_content, str) and chunk_content:
                content += chunk_content
                yield chunk_content
        state.machine.monologue.add_assistant_message(content=content)

    state.machine.state_bag["streamer"] = streamer()


def sky_is_blue(state) -> bool:
    """Mock predicate function that always returns True for testing"""
    return True


async def run_if_true(state):
    print("Action: THE ANSWER IS TRUE")


async def run_if_false(state):
    print("Action: THE ANSWER IS FALSE")


@pytest.mark.asyncio
async def test_pure_action_state():
    """Test PureActionState workflow - corrected version"""
    
    # Step 1: Build FSM
    with AgenticStateMachine.StateMachineBuilder("""
        INIT --> GENERATE
        GENERATE --> FINAL
        """) as builder:
        fsm = builder.build(
            {
                "INIT": {
                    "input_data": {},
                },
                "GENERATE": {
                    "module_path": "gai.asm.states",
                    "class_name": "PureActionState",
                    "title": "GENERATE",
                    "action": "generate",
                    "output_data": ["streamer"],
                },
                "FINAL": {
                    "output_data": ["monologue"],
                },
            },
            user_message="Hello, world!",
            generate=generate_action,
        )
    
    # Step 2: Initialize
    fsm.restart()

    # Step 3: INIT --> GENERATE
    # The key fix: fsm.run_async() returns a boolean, not a streamer
    success = await fsm.run_async()
    assert success is True or success is None  # Success indicator
    
    # Get the streamer from the state bag (this is the fix!)
    streamer = fsm.state_bag.get("streamer")
    assert streamer is not None, "Streamer should be available in state bag"
    
    # Now iterate through the streamer
    chunks = []
    async for chunk in streamer:
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print("\n")
    
    # Verify we got some content
    assert len(chunks) > 0, "Should have received some streaming chunks"
    
    # Step 4: GENERATE --> FINAL
    await fsm.run_async()
    
    # Step 5: Verify state history
    print("State History:")
    for state in fsm.state_history:
        print(f"State: {state['state']}")
        print(f"- input: {state['input']}")
        print(f"- output: {state['output']}")
        print("-" * 20)
    
    # Verify final state
    assert fsm.state == "FINAL"
    assert len(fsm.state_history) >= 2  # Should have at least INIT and GENERATE states


@pytest.mark.asyncio  
async def test_pure_predicate_state():
    """Test PurePredicateState workflow - corrected version"""
    
    with AgenticStateMachine.StateMachineBuilder("""
        INIT --> PREDICATE
        PREDICATE --> TRUE: condition_true
            TRUE --> FINAL 
        PREDICATE --> FALSE: condition_false
            FALSE --> FINAL    
        """) as builder:
        fsm = builder.build(
            {
                "PREDICATE": {
                    "module_path": "gai.asm.states",
                    "class_name": "PurePredicateState",
                    "title": "PREDICATE",
                    "predicate": "sky_is_blue",
                    "output_data": ["predicate_result"],
                    "conditions": ["condition_true", "condition_false"],
                },
                "TRUE": {
                    "module_path": "gai.asm.states",
                    "class_name": "PureActionState",
                    "input_data": {"title": "TRUE"},
                    "action": "run_if_true",
                },
                "FALSE": {
                    "module_path": "gai.asm.states",
                    "class_name": "PureActionState",
                    "input_data": {"title": "FALSE"},
                    "action": "run_if_false",
                },
            },
            sky_is_blue=sky_is_blue,
            run_if_true=run_if_true,
            run_if_false=run_if_false,
        )

    # Execute the state machine
    fsm.restart()
    
    # First run: INIT -> PREDICATE (evaluates predicate)
    await fsm.run_async()
    print("Current state is", fsm.state)
    print("Result is", fsm.state_bag["predicate_result"])
    
    # Should still be in PREDICATE state after evaluation
    assert fsm.state == "PREDICATE"
    assert fsm.state_bag["predicate_result"] is True

    # Second run: PREDICATE -> TRUE (based on condition)
    await fsm.run_async()
    print("Current state is", fsm.state)
    
    # Should be in TRUE state since sky_is_blue returns True
    assert fsm.state == "TRUE"

    # Third run: TRUE -> FINAL
    await fsm.run_async()
    print("Current state is", fsm.state)
    assert fsm.state == "FINAL"


if __name__ == "__main__":
    import asyncio
    
    # Run the tests
    print("Testing PureActionState...")
    asyncio.run(test_pure_action_state())
    
    print("\nTesting PurePredicateState...")  
    asyncio.run(test_pure_predicate_state())
    
    print("\nAll tests completed!")