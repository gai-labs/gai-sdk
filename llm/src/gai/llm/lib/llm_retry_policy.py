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
            or "Found multiple `tool_result` blocks with id:" in msg
            or "overloaded_error" in msg
        )

    async def on_retry(self, exception: Exception, retries: int, delay: int):
        msg = str(exception)

        if "`tool_use` ids were found without `tool_result`" in msg:
            # Find unmatched tool_use blocks and remove them along with their messages
            messages = self.machine.monologue.list_messages()

            # First, collect all tool_result ids to know which tool_use blocks have matches
            tool_result_ids = set()
            for message in messages:
                if isinstance(message.body.content, list):
                    for content_block in message.body.content:
                        if content_block.get("type") == "tool_result":
                            tool_result_ids.add(content_block.get("tool_use_id"))

            # Now traverse in reverse and remove messages with unmatched tool_use blocks
            for i in range(len(messages) - 1, -1, -1):
                message = messages[i]
                should_remove_message = False

                if isinstance(message.body.content, list):
                    for content_block in message.body.content:
                        if content_block.get("type") == "tool_use":
                            tool_use_id = content_block.get("id")
                            if tool_use_id not in tool_result_ids:
                                # This tool_use has no matching tool_result
                                should_remove_message = True
                                logger.warning(
                                    f"[tool_use] Removing message with unmatched tool_use id: {tool_use_id}"
                                )
                                break

                if should_remove_message:
                    messages.pop(i)

            self.machine.monologue.update(messages)

            logger.warning(
                f"[tool_use] {msg}. Retry in {delay} seconds ({retries}/{self.max_retries})."
            )
        elif "Found multiple `tool_result` blocks with id:" in msg:
            # Remove duplicate tool_result blocks, keeping only the first occurrence
            messages = self.machine.monologue.list_messages()
            seen_tool_use_ids = set()

            # Process messages in forward order to keep first occurrence
            for message in messages:
                if isinstance(message.body.content, list):
                    # Filter content blocks to remove duplicates
                    filtered_content = []
                    for content_block in message.body.content:
                        if content_block.get("type") == "tool_result":
                            tool_use_id = content_block.get("tool_use_id")
                            if tool_use_id not in seen_tool_use_ids:
                                # Keep first occurrence
                                seen_tool_use_ids.add(tool_use_id)
                                filtered_content.append(content_block)
                            else:
                                # Remove duplicate
                                logger.warning(
                                    f"[tool_result] Removing duplicate tool_result block with tool_use_id: {tool_use_id}"
                                )
                        else:
                            # Keep non-tool_result content
                            filtered_content.append(content_block)

                    # Update message content
                    message.body.content = filtered_content

            self.machine.monologue.update(messages)

            logger.warning(
                f"[tool_result] {msg}. Retry in {delay} seconds ({retries}/{self.max_retries})."
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
            or "Found multiple `tool_result` blocks with id:" in msg
            or "overloaded_error" in msg
        )

    async def remove_duplicated_tool_results(self):
        messages = self.machine.monologue.list_messages()

        # Find the last message
        last_message = messages[-1] if messages else None
        if not last_message:
            return messages

        # Find the tool_result blocks in the last message
        tool_result_ids = set()
        for content_block in last_message.body.content:
            if (
                isinstance(content_block, dict)
                and content_block.get("type") == "tool_result"
            ):
                tool_result_ids.add(content_block.get("tool_use_id"))
        if not tool_result_ids:
            return messages

        # Filter out all messages that have tool_result blocks with same tool_use_id
        final_messages = [last_message]
        for i in range(len(messages) - 2, -1, -1):
            message = messages[i]
            if not isinstance(message.body.content, list):
                final_messages.append(message)
                continue

            found = False
            for content_block in message.body.content:
                if (
                    content_block.get("type") == "tool_result"
                    and content_block.get("tool_use_id") in tool_result_ids
                ):
                    found = True
            if not found:
                final_messages.append(message)

        # Reverse the final messages to maintain original order
        final_messages.reverse()
        return final_messages

    async def remove_unmatched_tool_uses(self):
        messages = self.machine.monologue.list_messages()

        # Traverse in reverse and remove all messages until we find a message with tool_use block
        found = -1
        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]

            if isinstance(message.body.content, list):
                has_tool_use = False
                for content_block in message.body.content:
                    if content_block.get("type") == "tool_use":
                        has_tool_use = True
                        break
                if has_tool_use:
                    found = i
                    break

        if found > -1:
            # Include all messsages up to but not including the found message
            messages = messages[:found].copy()
        return messages

    async def on_retry(self, exception: Exception, retries: int, delay: int):
        msg = str(exception)

        if "`tool_use` ids were found without `tool_result`" in msg:
            # Find unmatched tool_use blocks and remove them along with their messages

            messages = await self.remove_unmatched_tool_uses()
            self.machine.monologue.update(messages)

            logger.warning(
                f"[tool_use] {msg}. Retry in {delay} seconds ({retries}/{self.max_retries})."
            )
        elif "Found multiple `tool_result` blocks with id:" in msg:
            messages = await self.remove_duplicated_tool_results()
            self.machine.monologue.update(messages)

            logger.warning(
                f"[tool_result] {msg}. Retry in {delay} seconds ({retries}/{self.max_retries})."
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
