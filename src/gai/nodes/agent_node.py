from typing import Optional, TypeVar, Generic, Protocol, AsyncGenerator, Any
from gai.sessions import SessionManager
from gai.sessions.operations.chat import ChatResponder
from gai.sessions.operations.handshake import HandshakeSender
from gai.messages.typing import MessagePydantic, OrchPlanPydantic
from gai.messages.dialogue import Dialogue
from gai.asm.agents import ChatAgent, AutoResumeError
from gai.lib.config import GaiClientConfig
from gai.lib.config import config_helper


class AgentProtocol(Protocol):
    """Protocol defining the interface that agents must implement for AgentNode."""

    fsm: Any  # StateModel with agent_name attribute

    def start(
        self, user_message: str, recap: Optional[str] = None
    ) -> AsyncGenerator[str, None]: ...

    def resume(
        self, user_message: Optional[str] = None
    ) -> AsyncGenerator[str, None]: ...


T = TypeVar("T", bound=AgentProtocol)


class AgentNode(Generic[T]):
    def __init__(
        self,
        agent_name: str,
        model_name: str,
        session_mgr: SessionManager,
        agent_class: Any = ChatAgent,
    ):
        self.agent_name = agent_name
        self.session_mgr = session_mgr
        self.agent_class = agent_class
        self.llm_config = config_helper.get_client_config(model_name)
        self.chat_responder = ChatResponder(
            node_name=agent_name, session_mgr=session_mgr
        )

    async def input_chunks_handler(self, pydantic: MessagePydantic):
        if pydantic.body.type == "chat.reply":
            # Should never handle reply messages.
            raise ValueError("input_chunks_callback should not process reply messages.")

        # Return a simulated streamer
        from gai.asm.agents import ToolUseAgent
        from gai.lib.config import config_helper

        llm_config = config_helper.get_client_config("sonnet-4")
        agent = ToolUseAgent(agent_name=self.agent_name, llm_config=llm_config)

        # Create the agent instance
        # Use keyword arguments to be flexible with different agent constructors
        agent_kwargs = {
            "agent_name": self.agent_name,
            "llm_config": self.llm_config,
        }

        agent = self.agent_class(**agent_kwargs)

        recap = self.session_mgr.dialogue.extract_recap()

        # Time to call chat completion
        async def get_streamer():
            from gai.asm.agents.tool_use_agent import AutoResumeError

            resp = agent.start(user_message=pydantic.body.content, recap=recap)
            content = ""
            # start
            async for chunk in resp:
                if isinstance(chunk, str):
                    if chunk:
                        content += chunk
                yield chunk
            content += "\n"
            # resume
            try:
                resp = agent.resume()
                async for chunk in resp:
                    if isinstance(chunk, str):
                        if chunk:
                            content += chunk
                        yield chunk
            except AutoResumeError:
                # conversation is over.
                if not content:
                    content = "My task is completed and I have nothing to resume from."
                self.session_mgr.dialogue.add_user_message(
                    recipient=agent.fsm.agent_name, content=pydantic.body.content
                )
                self.session_mgr.dialogue.add_assistant_message(
                    sender=agent.fsm.agent_name, chunk="<eom>", content=content
                )
                return

        return get_streamer()

    async def completed_content_handler(self, pydantic: MessagePydantic):
        self.session_mgr.log_message(pydantic)

    async def subscribe(self, flow_plan: str):
        # This method should be implemented in subclasses
        await self.chat_responder.subscribe(
            input_chunks_callback=self.input_chunks_handler,
            completed_content_callback=self.completed_content_handler,
        )
        self.chat_responder.plans[self.session_mgr.dialogue_id] = (
            HandshakeSender.create_plan(flow_plan)
        )
