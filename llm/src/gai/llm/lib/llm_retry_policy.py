import asyncio
from gai.lib.logging import getLogger

logger = getLogger(__name__)


class LLMRetryPolicy:
    def __init__(self, machine, *, max_retries=5, base_delay=5):
        self.machine = machine
        self.max_retries = max_retries
        self.base_delay = base_delay

    def should_retry(self, exception: Exception) -> bool:
        msg = str(exception)
        return (
            "`tool_use` ids were found without `tool_result`" in msg
            or "overloaded_error" in msg
        )

    async def on_retry(self, exception: Exception, retries: int, delay: int):
        msg = str(exception)

        if "`tool_use` ids were found without `tool_result`" in msg:
            # We need to traverse in reverse the monologue list and pop off each message until we found the message containing tool_result,
            # then pop off that one as well before saving the data.
            messages = self.machine.monologue.list_messages()
            for i in range(len(messages) - 1, -1, -1):
                message = messages.pop(i)
                if isinstance(message.body.content, list):
                    for content_block in message.body.content:
                        if content_block["type"] == "tool_use":
                            logger.warning(
                                f"[tool_use] Removed ToolUseContentBlock with unmatched `tool_use` ids."
                            )
                            break
            self.machine.monologue.update(messages)

            logger.warning(
                f"[tool_use] {msg}. Retry in {delay} seconds ({retries}/{self.max_retries})."
            )
        elif "overloaded_error" in msg:
            logger.warning(
                f"[overload] LLM service overloaded. Retry in {delay} seconds ({retries}/{self.max_retries})."
            )
        else:
            logger.error(f"[unhandled] {exception}")
            raise exception

    async def run(self, func):
        retries = 0
        delay = self.base_delay

        while True:
            try:
                return await func()
            except Exception as e:
                if retries >= self.max_retries or not self.should_retry(e):
                    raise e

                retries += 1
                await self.on_retry(e, retries, delay)
                await asyncio.sleep(delay)
                delay *= 2


class LLMGeneratorRetryPolicy:
    def __init__(self, machine, *, max_retries=5, base_delay=5):
        self.machine = machine
        self.max_retries = max_retries
        self.base_delay = base_delay

    def should_retry(self, exception: Exception) -> bool:
        msg = str(exception)
        return (
            "`tool_use` ids were found without `tool_result`" in msg
            or "overloaded_error" in msg
        )

    async def on_retry(self, exception: Exception, retries: int, delay: int):
        msg = str(exception)

        if "`tool_use` ids were found without `tool_result`" in msg:
            # We need to traverse in reverse the monologue list and pop off each message until we found the message containing tool_result,
            # then pop off that one as well before saving the data.
            messages = self.machine.monologue.list_messages()
            for i in range(len(messages) - 1, -1, -1):
                message = messages.pop(i)
                if isinstance(message.body.content, list):
                    for content_block in message.body.content:
                        if content_block["type"] == "tool_use":
                            logger.warning(
                                f"[tool_use] Removed ToolUseContentBlock with unmatched `tool_use` ids."
                            )
                            break
            self.machine.monologue.update(messages)

            logger.warning(
                f"[tool_use] {msg}. Retry in {delay} seconds ({retries}/{self.max_retries})."
            )
        elif "overloaded_error" in msg:
            logger.warning(
                f"[overload] LLM service overloaded. Retry in {delay} seconds ({retries}/{self.max_retries})."
            )
        else:
            logger.error(f"[unhandled] {exception}")
            raise exception

    async def run(self, func):
        retries = 0
        delay = self.base_delay

        while True:
            try:
                async for chunk in func():
                    yield chunk
            except Exception as e:
                if retries >= self.max_retries or not self.should_retry(e):
                    raise e

                retries += 1
                await self.on_retry(e, retries, delay)
                await asyncio.sleep(delay)
                delay *= 2
