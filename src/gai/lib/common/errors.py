import uuid
from fastapi import HTTPException


# 404 Errors - Not Found

class MessageNotFoundException(HTTPException):
    def __init__(self,message_id=None):
        self.message = "Message not found"
        self.code="message_not_found"
        if (message_id):
            self.message = f"Message {message_id} not found"
        super().__init__(status_code=404, detail={
            "code": self.code,
            "message": self.message
        })

class CollectionNotFoundException(HTTPException):
    def __init__(self,collection_name=None):
        self.message = "Collection not found"
        self.code="collections_not_found"
        if (collection_name):
            self.message = f"Collection {collection_name} not found"
        super().__init__(status_code=404, detail={
            "code": self.code,
            "message": self.message
        })

class DocumentNotFoundException(HTTPException):
    def __init__(self,document_id=None):
        self.message = "Document not found"
        self.code="document_not_found"
        if (document_id):
            self.message = f"Document {document_id} not found"
        super().__init__(status_code=404, detail={
            "code": self.code,
            "message": self.message
        })

class ChunkgroupNotFoundException(HTTPException):
    def __init__(self,chunkgroup_id=None):
        self.message = "Chunk group not found"
        self.code="chunkgroup_not_found"
        if (chunkgroup_id):
            self.message = f"Chunk group {chunkgroup_id} not found"
        super().__init__(status_code=404, detail={
            "code": self.code,
            "message": self.message
        })

class UserNotFoundException(HTTPException):
    def __init__(self,user_id=None):
        self.code="user_not_found"
        self.message = "User not found"
        if (user_id):
            self.message = f"User {user_id} not found"
        super().__init__(status_code=404, detail={
            "code": self.code,
            "message": self.message
        })

# 409 Errors - Application Conflict

class ContextLengthExceededException(HTTPException):
    def __init__(self):
        self.code="context_length_exceeded"
        self.message="The context length exceeded the maximum allowed length."
        super().__init__(status_code=409, detail={
            "code": self.code,
            "message": self.message
        })

class DuplicatedDocumentException(HTTPException):
    def __init__(self,doc_id=None):
        self.code = "duplicate_document"
        self.message = "Document already exists in the database."
        if (doc_id):
            self.message = f"Document {doc_id} already exists in the database."
        super().__init__(status_code=409, detail={
            "code": self.code,
            "message": self.message
        })

class GeneratorMismatchException(HTTPException):
    def __init__(self):
        self.code="generator_mismatch"
        self.message="The model does not correspond to this service."
        super().__init__(status_code=409, detail={
            "code": self.code,
            "message": self.message
        })

# 500 Errors - Internal Server Error

class InternalException(HTTPException):
    def __init__(self, error_id_or_uuid):
        self.code="unknown_error"
        self.message="An unexpected error occurred. Please try again later."

        # Check if the input is a valid UUID instance or a string
        if isinstance(error_id_or_uuid, uuid.UUID):
            error_id = str(error_id_or_uuid)  # Convert UUID to string
        elif isinstance(error_id_or_uuid, str):
            error_id = error_id_or_uuid

        detail = {
            "code": self.code,
            "message": self.message,
            "id": error_id,
        }
        super().__init__(status_code=500, detail=detail)


# General Errors that are not 500 Internal Server Errors

class ApiException(HTTPException):
    def __init__(self, status_code, code, message):
        self.code=code
        self.message=message
        super().__init__(status_code=status_code, detail = {
            "code": self.code,
            "message": self.message
        })
