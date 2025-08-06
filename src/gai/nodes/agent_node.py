from typing import Optional, TypeVar, Generic, Protocol, AsyncGenerator, Any
from gai.sessions import SessionManager
from gai.sessions.operations.chat import ChatResponder
from gai.sessions.operations.handshake import HandshakeSender
from gai.messages.typing import MessagePydantic, OrchPlanPydantic
from gai.messages.dialogue import Dialogue
from gai.asm.agents import ChatAgent, AutoResumeError
from gai.lib.config import GaiClientConfig


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
    """
    A simplified wrapper around ChatResponder that creates agent instances
    to handle chat messages in multi-agent sessions.
    """

    def __init__(
        self,
        agent_name: str,
        session_mgr: SessionManager,
        llm_config: GaiClientConfig,
        agent_class: Any = ChatAgent,
        dialogue: Optional[Dialogue] = None,
        aggregated_client=None,
        monologue=None,
    ):
        self.agent_name = agent_name
        self.session_mgr = session_mgr
        self.llm_config = llm_config
        self.agent_class = agent_class
        self.dialogue = dialogue or Dialogue(agent_name="User")
        self.aggregated_client = aggregated_client
        self.monologue = monologue
        self.responder = ChatResponder(node_name=agent_name, session_mgr=session_mgr)

    async def subscribe(self, flow_plan: str):
        """
        Subscribe the agent node to the session with the given plan.
        """
        await self.responder.subscribe(
            input_chunks_callback=self._input_handler,
            completed_content_callback=self._output_handler,
        )
        # self.responder.plans[plan.dialogue_id] = plan.model_copy()
        self.responder.plans[self.dialogue.dialogue_id] = HandshakeSender.create_plan(
            flow_plan
        )

    async def _input_handler(self, pydantic: MessagePydantic):
        """
        Handle incoming chat messages by creating a ToolUseAgent and generating response.
        """
        if pydantic.body.type == "chat.reply":
            raise ValueError("Should not process reply messages.")

        # Create the agent instance
        # Use keyword arguments to be flexible with different agent constructors
        agent_kwargs = {
            "agent_name": self.agent_name,
            "llm_config": self.llm_config,
        }
        if self.aggregated_client is not None:
            agent_kwargs["aggregated_client"] = self.aggregated_client
        if self.monologue is not None:
            agent_kwargs["monologue"] = self.monologue

        agent = self.agent_class(**agent_kwargs)

        # Get conversation history
        recap = self.dialogue.extract_recap()

        async def get_response():
            # Start the agent's response
            resp = agent.start(user_message=pydantic.body.content, recap=recap)
            content = ""

            async for chunk in resp:
                if isinstance(chunk, str) and chunk:
                    content += chunk
                yield chunk

            content += "\n"

            # Try to resume if needed
            try:
                resp = agent.resume()
                async for chunk in resp:
                    if isinstance(chunk, str) and chunk:
                        content += chunk
                    yield chunk
            except AutoResumeError:
                # Conversation completed - this is expected
                pass
            
            # Always save to dialogue after completion
            if not content:
                content = "Task completed."

            # Save to dialogue
            self.dialogue.add_user_message(
                recipient=agent.fsm.agent_name, content=pydantic.body.content
            )
            self.dialogue.add_assistant_message(
                sender=agent.fsm.agent_name, chunk="<eom>", content=content
            )

        return get_response()

    async def _output_handler(self, pydantic: MessagePydantic):
        """
        Handle completed content by logging the message.
        """
        self.session_mgr.log_message(pydantic)

    async def unsubscribe(self):
        """
        Unsubscribe from the session.
        """
        await self.responder.unsubscribe()
