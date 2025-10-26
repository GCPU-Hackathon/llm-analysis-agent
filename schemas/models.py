from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class StartConversationRequest(BaseModel):
    study_id: int
    study_code: str 
    system_prompt: Optional[str] = None

class ContinueConversationRequest(BaseModel):
    question: str
    study_id: int
    study_code: str

class StartConversationResponse(BaseModel):
    conversation_id: str
    study_id: int

class ConversationResponse(BaseModel):
    conversation_id: str
    response: str
    messages: List[Message]
    study_id: int

class ConversationHistory(BaseModel):
    conversation_id: str
    study_id: int
    messages: List[Message]
