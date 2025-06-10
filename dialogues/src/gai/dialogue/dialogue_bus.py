import os
import json
import copy
import inspect
import asyncio
import uuid
import time
from collections import defaultdict
from typing import Any, Callable, Awaitable, Dict, Union, overload, Optional
from threading import Lock
from pydantic import BaseModel

from gai.lib.constants import DEFAULT_GUID
from gai.lib.utils import get_app_path
from gai.lib.logging import getLogger
logger = getLogger(__name__)

# DialogueBus

from gai.messages import MessagePydantic, AsyncMessageBus

class DialogueBus:

    def __init__(self, max_recap_size = 4096):
        self.max_recap_size = max_recap_size   # This represents the maximum kilobyte size of message history text content before truncation
        self._amb = AsyncMessageBus() # <- single-process
        self._amb_task = None
        self.messages: list[MessagePydantic] = []

    def list_messages(self) -> list[MessagePydantic]:
        return self.messages

    async def __aenter__(self):
        # self._amb_task = asyncio.create_task(self._amb.run())
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        logger.warning(f"DialogueBus: __aexit__ triggered. exc_type={exc_type}, exc={exc}")
        await self.stop()
       
    async def start(self):
        
        """amb is idempotent."""

        if self._amb_task:
            # already running
            return
        
        await self._amb.start()
        logger.info("DialogueBus: ready.")

    async def stop(self):

        """amb is idempotent."""
        
        await self._amb.stop()
        if self._amb_task:
            self._amb_task.cancel()
            try:
                await self._amb_task
            except asyncio.CancelledError:
                pass
            self._amb_task = None
        logger.info("DialogueBus: stopped.")
        
    async def subscribe(self, subject: str, callback: dict[str,Callable]):

        if not isinstance(callback, dict):
            raise TypeError("DialogueBus: subscribe() requires a dictionary of callbacks.")

        if self._amb:
            # self._amb can be disabled if _amb is not used.
            await self._amb.subscribe(subject, callback)

    async def unsubscribe(self, subject: str, subscriber_name: str):
        """Unsubscribe a subscriber from a subject."""
        if self._amb:
            # self._amb can be disabled if _amb is not used.
            await self._amb.unsubscribe(subject, subscriber_name)
            
    async def unsubscribe_all(self):
        if self._amb:
            # self._amb can be disabled if _amb is not used.
            await self._amb.unsubscribe_all()


    @overload
    async def publish(self, pydantic: MessagePydantic) -> MessagePydantic: ...
    
    @overload
    async def publish(self, type: str, body: str, sender: str, recipient: str) -> MessagePydantic: ...

    async def publish(self, *, pydantic: Optional[MessagePydantic] = None, type: Optional[str] = None, body: Optional[str] = None, sender: Optional[str] = None, recipient: Optional[str] = None) -> MessagePydantic:

        pydantic = self._validate_message_copy(pydantic,type,body,sender,recipient)

        # do not save the message if it is a chunk
        try:
            
            if pydantic.body.type == "reply" and pydantic.body.chunk != "<eom>":
                # Do not save message of a mid-stream chunk
                pass
            else:
                # Save the final chunk
                self.log_message(pydantic)
            
            if self._amb:
                # self._amb can be disabled if _amb is not used.
                await self._amb.publish(pydantic)
        
            return pydantic
        
        except Exception as e:
            logger.error(f"DialogueBus: Error during publish: {e}")
            raise

    def is_started(self) -> bool:
        """Check if the dialogue bus is started."""
        return self._amb.is_started if self._amb else False

    def log_message(self, message: MessagePydantic):

        """Add a message to dialogue history. """
       
        logger.debug(f"DialogueBus: message={message}")
        self.messages.append(message)
    
    def extract_recap(self, last_n: int=0) -> str:
        # Lazy import to avoid circular dependency
        from gai.messages.message_helper import extract_recap
        return extract_recap(self.messages, last_n=last_n, max_recap_size=self.max_recap_size)

    def _validate_message_copy(self, pydantic: MessagePydantic, type: str, body:str, sender:str, recipient:str) -> MessagePydantic:
        if pydantic:
            m = copy.deepcopy(pydantic)
        elif all([type, body, sender, recipient]):
            m = MessagePydantic(
                id=str(uuid.uuid4()),
                header={
                    "sender": sender,
                    "recipient": recipient,
                    "timestamp": time.time()
                },
                body={
                    "type": type,
                    "content": body
                }
            )
        else:            
            raise TypeError("DialogueBus: publish() requires either `message` or all of `type`, `body`, `sender`, and `recipient` as keyword arguments.")
        return m
            
    def is_subscribed(self, subject: str, subscriber_name: str) -> bool:
        """Check if a subscriber is subscribed to a subject."""
        if self._amb:
            # self._amb can be disabled if _amb is not used.
            return self._amb.is_subscribed(subject, subscriber_name)
        return False
        
    def reset(self):
        """Reset the dialogue by clearing all messages."""
        self.messages.clear()
        logger.debug("DialogueBus: Dialogue reset. all messages cleared.")
        
    def delete_message(self,message_id):
        """Delete a message by its ID."""
        self.messages = [m for m in self.messages if m.id != message_id]
        logger.debug(f"DialogueBus: Message deleted. message_id= {message_id}")

#-----------------------------------------------------------------------------------------------------------------------------------------

class FileDialogueBus(DialogueBus):
    
    # Performs CRUD operations on dialogue file.
    # The file is stored in the app_dir/data/<caller_id>/<logger_name>/dialogue/<dialogue_id>.json

    class DialogueFileStorage:
        
        class InternalStructure(BaseModel):
            last_message_order: int = 0
            messages: list[MessagePydantic] = []
        
        #file lock
        file_lock = Lock()

        def __init__(self, logger_name: str, app_dir: str = None, dialogue_id: str = DEFAULT_GUID):
            self.caller_id = DEFAULT_GUID
            self.dialogue_id = dialogue_id
            self.logger_name = logger_name
            self.app_dir = app_dir or get_app_path()
            if self.logger_name == "User":
                self.dialogue_dir = os.path.join(self.app_dir, "data", self.caller_id, "user", "dialogue")
            else:
                self.dialogue_dir = os.path.join(self.app_dir, "data", self.caller_id, "agents",logger_name.lower(), "dialogue")
            os.makedirs(self.dialogue_dir, exist_ok=True)

            # Ensure file is created with valid structure
            dialogue_path = self.get_dialogue_path()
            if not os.path.exists(dialogue_path):
                self.reset()

        def get_dialogue_path(self) -> str:
            return os.path.join(self.dialogue_dir, f"{self.dialogue_id}.json")
        
        def get_message(self, id: str) -> Optional[MessagePydantic]:
            """Get a message from the dialogue file by its ID."""
            with FileDialogueBus.DialogueFileStorage.file_lock:
                dialogue_path = self.get_dialogue_path()
                if not os.path.exists(dialogue_path):
                    logger.error(f"DialogueFileStorage: Dialogue file not found. path={dialogue_path}")
                    raise FileNotFoundError(f"DialogueFileStorage: Dialogue file not found. path={dialogue_path}")

                with open(dialogue_path, "r") as f:
                    try:
                        internal_structure = FileDialogueBus.DialogueFileStorage.InternalStructure(**json.load(f))
                    except json.JSONDecodeError:
                        logger.warning(f"DialogueFileStorage: Failed to load internal structure from {dialogue_path}. Creating new one.")
                        internal_structure = FileDialogueBus.DialogueFileStorage.InternalStructure()
                    
                for message in internal_structure.messages:
                    if message.id == id:
                        return MessagePydantic(**message.model_dump())
                return None
        
        def list_messages(self) -> list[MessagePydantic]:
            """List all messages in the dialogue file."""
            with FileDialogueBus.DialogueFileStorage.file_lock:
                dialogue_path = self.get_dialogue_path()
                if not os.path.exists(dialogue_path):
                    logger.warning(f"DialogueFileStorage: Dialogue file not found. path={dialogue_path}")
                    raise FileNotFoundError(f"DialogueFileStorage: Dialogue file not found path={dialogue_path}")
                
                with open(dialogue_path, "r") as f:
                    try:
                        internal_structure = FileDialogueBus.DialogueFileStorage.InternalStructure(**json.load(f))
                    except json.JSONDecodeError:
                        logger.error(f"DialogueFileStorage: Failed to load internal structure from {dialogue_path}. Creating new one.")
                        raise Exception(f"DialogueFileStorage: Failed to load internal structure from {dialogue_path}. Creating new one.")
                    
                return internal_structure.messages
        
        def insert_message(self, message: MessagePydantic):
            """Insert a message into the dialogue file."""
            with FileDialogueBus.DialogueFileStorage.file_lock:
                dialogue_path = self.get_dialogue_path()
                
                # Read existing data from file
                
                with open(dialogue_path, "r") as f:
                    try:
                        internal_structure = FileDialogueBus.DialogueFileStorage.InternalStructure(**json.load(f))
                    except json.JSONDecodeError:
                        logger.warning(f"DialogueFileStorage: Failed to load internal structure from {dialogue_path}. Creating new one.")
                        internal_structure = FileDialogueBus.DialogueFileStorage.InternalStructure()
                
                # Update internal structure
                
                internal_structure.messages.append(message)
                internal_structure.last_message_order += 1
                
                # Save internal structure back to file
                with open(dialogue_path, "w") as f:
                    jsoned = internal_structure.model_dump()
                    f.write(json.dumps(jsoned, indent=4))
                
                logger.debug(f"DialogueFileStorage: Message added to file. message={message}")
            
        def delete_message(self, id: str):
            """Delete a message from the dialogue file."""
            with FileDialogueBus.DialogueFileStorage.file_lock:
                dialogue_path = self.get_dialogue_path()
                with open(dialogue_path, "r") as f:
                    messages = f.readlines()
                with open(dialogue_path, "w") as f:
                    for line in messages:
                        message = json.loads(line)
                        if message["id"] != id:
                            f.write(line)
                logger.debug(f"DialogueFileStorage: Message deleted from file. id={id}")

        def update_message(self, message: MessagePydantic):
            """Update a message in the dialogue file."""
            with FileDialogueBus.DialogueFileStorage.file_lock:
                dialogue_path = self.get_dialogue_path()

                # Read existing data from file
                
                with open(dialogue_path, "r") as f:
                    try:
                        internal_structure = FileDialogueBus.DialogueFileStorage.InternalStructure(**json.load(f))
                    except json.JSONDecodeError:
                        logger.warning(f"DialogueFileStorage: Failed to load internal structure from {dialogue_path}. Creating new one.")
                        internal_structure = FileDialogueBus.DialogueFileStorage.InternalStructure()

                updated=False
                for i, msg in enumerate(internal_structure.messages):
                    if msg.id == message.id:
                        internal_structure.messages[i] = message  # ✅ REPLACE instead of mutating
                        updated = True
                        break
                    
                if updated:
                    # Save updated internal structure back to file
                    with open(dialogue_path, "w") as f:
                        jsoned = internal_structure.model_dump()
                        f.write(json.dumps(jsoned, indent=4))
                    logger.debug(f"DialogueFileStorage: Message updated in file. message={message}")
            
        def reset(self):
            """Reset the dialogue file by clearing all messages."""
            with FileDialogueBus.DialogueFileStorage.file_lock:
                dialogue_path = self.get_dialogue_path()
                with open(dialogue_path, "w") as f:
                    initial = FileDialogueBus.DialogueFileStorage.InternalStructure().model_dump()
                    f.write(json.dumps(initial, indent=4))        
                logger.debug("DialogueFileStorage: Dialogue file reset. all messages cleared.")
    

    def __init__(
        self,
        logger_name: str,
        app_dir: str = None,
        dialogue_id: str = DEFAULT_GUID,
        reset: bool = False,
        max_recap_size = 4096,         
    ):
        super().__init__(max_recap_size=max_recap_size)
        self.logger_name = logger_name
        self.storage = FileDialogueBus.DialogueFileStorage(
            logger_name=logger_name, 
            app_dir=app_dir, 
            dialogue_id=dialogue_id
            )
        # Load messages from file or create a new file if it doesn't exist
        if reset:
            self.storage.reset()
        self.messages = self.storage.list_messages()

    # Override add message to append message to file after adding to memory.
    def log_message(self, message: MessagePydantic):
        # Keep original message in memory (e.g., UI can render avatars)
        super().log_message(message)
        
        if message.body.type == "reply" and message.body.chunk != "<eom>":
            # Do not save the message if it is a mid-stream chunk
            return
        else:
            if message.body.type == "system.profile":
                
                # Remove images from the profile message before saving to file
                
                message.body.image_64x64 = None
                message.body.image_128x128 = None

        self.storage.insert_message(message)

    def reset(self):
        # Clear the messages in memory
        super().reset()
        
        # Clear the messages in the file
        self.storage.reset()
        
    def delete_message(self,message_id):
        # Delete the message in memory
        super().delete_message(message_id)
        
        # Delete the message in the file
        self.storage.delete_message(message_id)

